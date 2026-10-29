"""Unit tests for pure helpers: chunking, scoring, serialisation and config."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from app.config import Settings
from app.core.security import parse_api_keys
from app.core.serialization import benchmark_from_dict, benchmark_to_dict
from app.core.utils import content_tokens, fit_dimensions, parse_json_object, token_count
from app.domain import (
    BenchmarkResult,
    BenchmarkRunReport,
    ChunkRecord,
    Citation,
    DocumentPage,
    DocumentRecord,
    to_summary,
)
from app.rag.bm25 import BM25Index
from app.rag.chunking import split_pages, to_chunk_records
from app.rag.citations import (
    build_extractive_answer,
    select_citations,
    split_sentences,
    verify_answer,
)
from app.rag.entities import extract_entities
from app.rag.graph import EntityGraph
from app.rag.query import deterministic_rewrites, normalise_question
from app.repositories.memory import MemoryDocumentRepository

from tests.conftest import make_settings


def _settings(**overrides):
    return make_settings(**overrides)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    entities: list[str] | None = None,
    page: int = 1,
    document_id: str = "doc-1",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        tenant_id="default",
        document_id=document_id,
        document_title="Test Document",
        page_number=page,
        chunk_index=int(chunk_id.rsplit(":", 1)[-1]) if ":" in chunk_id else 0,
        text=text,
        embedding=[0.1, 0.2, 0.3],
        entities=entities or [],
        token_count=token_count(text),
    )


# Configuration ---------------------------------------------------------
def test_production_requires_database_url():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings(environment="production", auth_mode="none")


def test_production_can_opt_into_memory_backend():
    settings = Settings(
        environment="production",
        allow_memory_backend=True,
        auth_mode="none",
    )
    assert settings.uses_postgres is False
    assert settings.resolved_auth_mode == "none"


def test_production_requires_api_keys_when_api_key_auth():
    with pytest.raises(ValueError, match="API_KEYS"):
        Settings(
            environment="production",
            database_url="postgresql://user:pass@localhost:5432/documind",
            api_keys="",
        )


def test_production_defaults_to_api_key_auth():
    settings = Settings(
        environment="production",
        database_url="postgresql://user:pass@localhost:5432/documind",
        api_keys="k1:tenant-a",
    )
    assert settings.resolved_auth_mode == "api-key"


def test_development_defaults_to_no_auth():
    assert make_settings().resolved_auth_mode == "none"


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=200, chunk_overlap=200)


def test_embedding_dimensions_are_bounded():
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        Settings(embedding_dimensions=8)


def test_async_database_url_rewrites_scheme():
    settings = Settings(database_url="postgres://u:p@localhost:5432/db")
    assert settings.async_database_url == "postgresql+asyncpg://u:p@localhost:5432/db"


def test_cors_origins_accept_csv_string():
    settings = Settings(cors_origins="http://a.test,http://b.test")  # type: ignore[arg-type]
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


# Auth ------------------------------------------------------------------
def test_parse_api_keys_supports_tenants():
    parsed = parse_api_keys("key-a:tenant-a, key-b , key-c:tenant-c")
    assert parsed == {
        "key-a": "tenant-a",
        "key-b": "default",
        "key-c": "tenant-c",
    }


def test_parse_api_keys_ignores_empty_entries():
    assert parse_api_keys("  ,  ,k:v") == {"k": "v"}


# Chunking --------------------------------------------------------------
def test_split_text_into_pages_honours_form_feed():
    pages = split_pages(
        [
            DocumentPage(page_number=1, text="Alpha paragraph. " * 20, char_count=0),
            DocumentPage(page_number=2, text="Beta paragraph. " * 20, char_count=0),
        ],
        _settings(chunk_size=200, chunk_overlap=20),
    )
    assert pages
    assert {draft.page_number for draft in pages} == {1, 2}


def test_chunk_offsets_point_at_source_text():
    text = "Sentence one is here. Sentence two follows. Sentence three closes the page."
    pages = [DocumentPage(page_number=1, text=text, char_count=len(text))]
    drafts = split_pages(pages, _settings(chunk_size=60, chunk_overlap=0))
    assert drafts
    for draft in drafts:
        assert text[draft.start_char : draft.end_char] == draft.text


def test_chunk_ids_are_page_aware_and_ordered():
    drafts = split_pages(
        [DocumentPage(page_number=3, text="Hello world. " * 40, char_count=0)],
        _settings(chunk_size=100, chunk_overlap=10),
    )
    records = to_chunk_records(
        drafts,
        tenant_id="default",
        document_id="doc-9",
        document_title="Doc",
    )
    assert records
    assert [record.chunk_index for record in records] == list(range(len(records)))
    assert all(record.chunk_id.startswith("doc-9:") for record in records)
    assert all(record.page_number == 3 for record in records)


# Retrieval scoring -----------------------------------------------------
def test_bm25_ranks_exact_term_first():
    index = BM25Index(
        {
            "a": "The quick brown fox jumps over the lazy dog.",
            "b": "Quarterly revenue grew by 18.4 percent year over year.",
            "c": "Encryption keys rotate every 90 days using AES-256-GCM.",
        }
    )
    ranked = index.top("AES-256-GCM key rotation", limit=3)
    assert ranked
    assert ranked[0][0] == "c"
    assert ranked[0][1] > 0


def test_bm25_ignores_stopwords_only_query():
    index = BM25Index({"a": "some text"})
    assert index.score("what is the") == {}


def test_entity_graph_links_shared_entities():
    chunks = [
        _chunk("doc-1:0", "SOC 2 policy covers encryption.", entities=["SOC 2", "encryption"]),
        _chunk("doc-1:1", "Encryption uses AES-256.", entities=["encryption", "AES-256"]),
        _chunk("doc-1:2", "Revenue grew 18%.", entities=["revenue"]),
    ]
    graph = EntityGraph(chunks)
    neighbours = graph.neighbors(["doc-1:0"], limit=5)
    assert neighbours
    assert neighbours[0].chunk.chunk_id == "doc-1:1"
    assert "encryption" in {entity.lower() for entity in neighbours[0].shared_entities}


def test_entity_graph_returns_nothing_for_unknown_seed():
    assert EntityGraph([_chunk("a:0", "text")]).neighbors(["missing"]) == []


def test_entity_extraction_finds_acronyms_and_quantities():
    entities = extract_entities(
        "Apex Holdings reported $482.6 million in Q3 2026 under the SOC 2 and AES-256-GCM controls."
    )
    lowered = {entity.lower() for entity in entities}
    assert any("apex" in entity for entity in lowered)
    assert any("aes-256" in entity for entity in lowered)
    assert len(entities) <= 12


def test_entity_extraction_handles_empty_text():
    assert extract_entities("   ") == []


# Query analysis --------------------------------------------------------
def test_normalise_question_strips_filler():
    assert normalise_question("Please tell me what was the revenue?") == "what was the revenue"


def test_deterministic_rewrites_are_deduplicated():
    rewrites = deterministic_rewrites("What was the Q3 2026 revenue?", [])
    assert rewrites
    assert len(rewrites) == len(set(rewrites))


# Verification ----------------------------------------------------------
def test_split_sentences_drops_citation_markers():
    sentences = split_sentences(
        "Revenue was 482.6 million dollars [1]. Margin reached 24.8 percent [2]."
    )
    assert sentences
    assert all("[" not in sentence for sentence in sentences)


def test_verify_answer_scores_grounded_text_highly():
    chunks = [
        _chunk(
            "doc-1:0",
            "Consolidated net revenue was 482.6 million dollars, up 18.4 percent year over year.",
        )
    ]
    result = verify_answer(
        "Consolidated net revenue was 482.6 million dollars and grew 18.4 percent [1].", chunks
    )
    assert result.grounding_score > 0.6
    assert result.unsupported_sentences == []


def test_verify_answer_flags_unsupported_claims():
    chunks = [_chunk("doc-1:0", "The policy requires encryption at rest using AES-256.")]
    result = verify_answer(
        "The company launched a lunar mining division in 2027.", chunks
    )
    assert result.unsupported_sentences


def test_extractive_answer_is_cited_and_bounded():
    chunks = [
        _chunk(
            "doc-1:0",
            "Net revenue was 482.6 million dollars. Adjusted EBITDA margin was 24.8 percent. "
            "Cash totalled 1.24 billion dollars. Headcount was 3,412 employees. "
            "Guidance implies 16 to 19 percent growth.",
        )
    ]
    answer = build_extractive_answer("net revenue and EBITDA margin", chunks)
    assert "[1]" in answer
    assert answer.count("\n- ") <= 4


def test_extractive_answer_handles_empty_corpus():
    answer = build_extractive_answer("anything", [])
    assert "cannot answer" in answer.lower()


def test_select_citations_prefers_explicit_markers():
    chunks = [_chunk(f"doc-1:{index}", f"passage {index}") for index in range(3)]
    citations = select_citations(chunks, "Only the third passage matters [3].")
    assert len(citations) == 1
    assert citations[0].chunk_id == "doc-1:2"


# Serialisation ---------------------------------------------------------
def test_benchmark_roundtrip_preserves_fields():
    report = BenchmarkRunReport(
        run_id="run-1",
        timestamp=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        total_tests=1,
        passed_count=1,
        overall_recall_at_k=100.0,
        average_faithfulness=92.5,
        hallucination_rate=0.0,
        abstention_accuracy=100.0,
        average_latency_ms=42,
        results=[
            BenchmarkResult(
                id="res-1",
                test_case_id="tc-1",
                question="Q?",
                document_title="Doc",
                is_negative_test=False,
                retrieved_chunks=[
                    Citation(
                        document_id="d1",
                        document_title="Doc",
                        page_number=2,
                        chunk_id="d1:1",
                        excerpt="text",
                        similarity_score=0.5,
                        lexical_score=0.4,
                        combined_score=0.45,
                        source="graph",
                    )
                ],
                generated_answer="Answer [1].",
                recall_hit=True,
                faithfulness_score=90.0,
                hallucination_detected=False,
                abstention_correct=True,
                latency_ms=42,
                passed=True,
            )
        ],
        tenant_id="tenant-a",
    )
    payload = benchmark_to_dict(report)
    assert payload["runId"] == "run-1"
    assert payload["results"][0]["retrievedChunks"][0]["source"] == "graph"

    restored = benchmark_from_dict(payload)
    assert restored.run_id == report.run_id
    assert restored.tenant_id == "tenant-a"
    assert restored.results[0].retrieved_chunks[0].chunk_id == "d1:1"
    assert restored.results[0].faithfulness_score == 90.0


def test_to_summary_counts_chunks():
    record = DocumentRecord(
        id="d1",
        tenant_id="default",
        title="Title",
        filename="t.txt",
        file_size=10,
        mime_type="text/plain",
        checksum="abc",
        page_count=1,
        status="ready",
        embedding_status="ready",
        uploaded_at=datetime.now(UTC),
        chunks=[_chunk("d1:0", "a"), _chunk("d1:1", "b")],
    )
    summary = to_summary(record)
    assert summary.chunk_count == 2
    assert summary.title == "Title"


# Utilities -------------------------------------------------------------
def test_fit_dimensions_normalises_and_pads():
    vector = fit_dimensions([3.0, 4.0], 8)
    assert len(vector) == 8
    assert pytest.approx(sum(value * value for value in vector), rel=1e-9) == 1.0


def test_fit_dimensions_handles_zero_vector():
    assert fit_dimensions([0.0, 0.0], 4) == [0.0, 0.0, 0.0, 0.0]


def test_parse_json_object_extracts_fenced_block():
    payload = parse_json_object('Sure!\n```json\n{"entities": ["A"]}\n```\nThanks')
    assert payload == {"entities": ["A"]}


def test_parse_json_object_returns_none_for_garbage():
    assert parse_json_object("no json here") is None


def test_content_tokens_remove_stopwords_and_short_terms():
    tokens = content_tokens("What is the AES key rotation policy?", min_length=3)
    assert "aes" in tokens
    assert "rotation" in tokens
    assert "the" not in tokens


# In-memory repository --------------------------------------------------
async def test_memory_repository_isolates_tenants():
    repository = MemoryDocumentRepository()
    record = DocumentRecord(
        id="11111111-1111-1111-1111-111111111111",
        tenant_id="tenant-a",
        title="A",
        filename="a.txt",
        file_size=1,
        mime_type="text/plain",
        checksum="x",
        page_count=1,
        status="ready",
        embedding_status="ready",
        uploaded_at=datetime.now(UTC),
        chunks=[_chunk("11111111-1111-1111-1111-111111111111:0", "secret")],
    )
    await repository.save_document(record)

    assert await repository.get_document("tenant-a", record.id) is not None
    assert await repository.get_document("tenant-b", record.id) is None

    summaries, total = await repository.list_documents("tenant-a", limit=10, offset=0)
    assert total == 1 and summaries[0].title == "A"

    assert await repository.delete_document("tenant-b", record.id) is False
    assert await repository.delete_document("tenant-a", record.id) is True
    assert await repository.count_documents("tenant-a") == 0


# --- Per-request AI overrides (BYO provider keys) --------------------------------

def test_parse_ai_overrides_accepts_camel_case_payload():
    import json

    from app.api.ai_overrides import parse_ai_overrides

    header = json.dumps(
        {
            "llmProvider": "gemini",
            "geminiApiKey": "  fake-key  ",
            "embeddingProvider": "huggingface",
            "hfApiToken": "token",
        }
    )
    overrides = parse_ai_overrides(header)

    assert overrides is not None
    assert overrides.llm_provider == "gemini"
    assert overrides.gemini_api_key == "fake-key"  # stripped
    assert overrides.embedding_provider == "huggingface"


def test_parse_ai_overrides_none_and_empty():
    from app.api.ai_overrides import parse_ai_overrides

    assert parse_ai_overrides(None) is None
    assert parse_ai_overrides("") is None
    assert parse_ai_overrides("{}") is None  # nothing to override


def test_parse_ai_overrides_rejects_bad_json_and_wrong_types():
    import json as _json

    from app.api.ai_overrides import parse_ai_overrides
    from app.core.errors import AppError

    with pytest.raises(AppError) as bad_json:
        parse_ai_overrides("not-json{")
    assert bad_json.value.code == "INVALID_AI_CONFIG"

    with pytest.raises(AppError) as wrong_type:
        parse_ai_overrides(_json.dumps({"llmProvider": "hyperloop"}))
    assert wrong_type.value.code == "INVALID_AI_CONFIG"


def test_apply_overrides_swaps_settings_fields():
    from app.api.ai_overrides import AiOverrides, apply_overrides

    base = make_settings()
    overrides = AiOverrides(
        llmProvider="groq", groqApiKey="g-key", groqChatModel="llama-3.1-8b-instant"
    )
    merged = apply_overrides(base, overrides)

    assert merged.llm_provider == "groq"
    assert merged.groq_api_key == "g-key"
    assert merged.groq_chat_model == "llama-3.1-8b-instant"
    # The original settings object is untouched.
    assert base.llm_provider == "none"
    # The base's Groq config (if any) is left as-is, even when an override
    # replaces it. (The local .env may legitimately store a real key.)
    assert base.groq_api_key != "g-key"
    assert base.groq_chat_model != "llama-3.1-8b-instant"
    # Unlisted fields are inherited from the base.
    assert merged.embedding_dimensions == base.embedding_dimensions