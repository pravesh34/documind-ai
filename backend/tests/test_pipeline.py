"""Service-level end-to-end tests for ingestion, retrieval and chat."""

from __future__ import annotations

import pytest
from app.core.errors import AppError
from app.domain import BenchmarkTestCase
from app.services.seed_data import DEFAULT_DOCUMENTS

EXPECTED_TITLES = {document.title for document in DEFAULT_DOCUMENTS}


async def test_seed_corpus(container):
    summaries, total = await container.documents.list_documents(
        "default", limit=10, offset=0
    )
    assert total == 3
    assert {summary.title for summary in summaries} == EXPECTED_TITLES
    assert all(summary.chunk_count > 0 for summary in summaries)
    assert all(summary.status == "ready" for summary in summaries)


async def test_chat_answers_with_citations(container):
    answer = await container.chat.answer(
        "default",
        "What was the consolidated net revenue and adjusted EBITDA margin for Q3 2026?",
        top_k=4,
    )

    assert answer.abstained is False
    assert answer.citations, "expected at least one citation"
    assert any("482.6" in citation.excerpt for citation in answer.citations)
    assert answer.grounding_score >= 0.55
    assert answer.retrieval_attempts >= 0
    assert answer.trace


async def test_chat_abstains_on_out_of_domain_question(container):
    answer = await container.chat.answer(
        "default",
        "What were the revenues generated from the commercial lunar space flight program in Q3?",
        top_k=4,
    )

    assert answer.abstained is True
    assert answer.citations == []
    assert "cannot answer" in answer.answer.lower()


async def test_chat_abstains_on_fabricated_premise(container):
    answer = await container.chat.answer(
        "default",
        "Which quantum encryption patent was granted to Apex Holdings in 2026?",
        top_k=4,
    )

    assert answer.abstained is True


async def test_retrieval_finds_encryption_control(container):
    result = await container.chat.retrieve(
        "default", "AES-256 key rotation and FIPS validation", top_k=4
    )

    assert result.citations
    assert any("AES-256" in citation.excerpt for citation in result.citations)
    assert result.total_candidates > 0


async def test_upload_two_page_document(container):
    document = await container.ingestion.ingest(
        "default",
        title="Zephyr Programme Brief",
        filename="zephyr.txt",
        text=(
            "The Zephyr programme budget is $9.4 million for 2027. "
            "Delivery starts in March.\n\n\fPage two lists the Zephyr milestones "
            "and acceptance criteria for the integration phase."
        ),
    )

    assert document.page_count == 2
    assert document.chunks
    assert document.embedding_status == "ready"

    stored = await container.documents.get_document("default", document.id)
    assert stored.title == "Zephyr Programme Brief"
    assert len(stored.pages) == 2

    await container.documents.delete_document("default", document.id)
    with pytest.raises(AppError):
        await container.documents.get_document("default", document.id)


async def test_evaluation_run_and_history(container):
    tests = [
        BenchmarkTestCase(
            id="t-revenue",
            document_title="Apex Holdings Q3 2026 Financial Report",
            question="What was the consolidated net revenue for Q3 2026?",
            expected_passage="consolidated net revenue of $482.6 million",
            expected_page=1,
            is_negative_test=False,
            description="Revenue extraction",
        ),
        BenchmarkTestCase(
            id="t-negative",
            document_title="Apex Holdings Q3 2026 Financial Report",
            question="What was the revenue from the lunar space flight programme?",
            expected_passage="NONE",
            expected_page=0,
            is_negative_test=True,
            description="Abstention",
        ),
    ]

    report = await container.evaluation.run("default", tests)
    assert report.total_tests == 2
    assert report.passed_count >= 1
    assert report.average_latency_ms >= 0
    assert report.tenant_id == "default"

    latest = await container.evaluation.latest("default")
    assert latest is not None
    assert latest.run_id == report.run_id
