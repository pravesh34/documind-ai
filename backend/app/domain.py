"""Framework-free domain models shared by repositories, services and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

DocumentStatus = Literal["pending", "processing", "ready", "error"]
EmbeddingStatus = Literal["pending", "ready", "unavailable"]
RetrievalSource = Literal["dense", "lexical", "graph"]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class DocumentPage:
    page_number: int
    text: str
    char_count: int


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    tenant_id: str
    document_id: str
    document_title: str
    page_number: int
    chunk_index: int
    text: str
    embedding: list[float] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass(slots=True)
class DocumentRecord:
    id: str
    tenant_id: str
    title: str
    filename: str
    file_size: int
    mime_type: str
    checksum: str
    page_count: int
    status: DocumentStatus
    embedding_status: EmbeddingStatus
    uploaded_at: datetime
    pages: list[DocumentPage] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    summary: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class DocumentSummary:
    id: str
    title: str
    filename: str
    file_size: int
    page_count: int
    chunk_count: int
    status: DocumentStatus
    embedding_status: EmbeddingStatus
    uploaded_at: datetime
    summary: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class Citation:
    document_id: str
    document_title: str
    page_number: int
    chunk_id: str
    excerpt: str
    similarity_score: float
    lexical_score: float = 0.0
    combined_score: float = 0.0
    source: RetrievalSource = "dense"


@dataclass(slots=True)
class ScoredChunk:
    chunk: ChunkRecord
    score: float
    similarity: float
    lexical: float
    source: RetrievalSource


@dataclass(slots=True)
class RetrievalOptions:
    top_k: int
    similarity_threshold: float
    target_document_id: str | None = None
    candidates: int = 24


@dataclass(slots=True)
class RetrievalResult:
    query: str
    queries: list[str]
    scored_chunks: list[ScoredChunk]
    citations: list[Citation]
    total_candidates: int
    latency_ms: int
    candidate_terms: set[str] = field(default_factory=set)


@dataclass(slots=True)
class BenchmarkTestCase:
    id: str
    document_title: str
    question: str
    expected_passage: str
    expected_page: int
    is_negative_test: bool
    description: str


@dataclass(slots=True)
class BenchmarkResult:
    id: str
    test_case_id: str
    question: str
    document_title: str
    is_negative_test: bool
    retrieved_chunks: list[Citation]
    generated_answer: str
    recall_hit: bool
    faithfulness_score: float
    hallucination_detected: bool
    abstention_correct: bool
    latency_ms: int
    passed: bool


@dataclass(slots=True)
class BenchmarkRunReport:
    run_id: str
    timestamp: datetime
    total_tests: int
    passed_count: int
    overall_recall_at_k: float
    average_faithfulness: float
    hallucination_rate: float
    abstention_accuracy: float
    average_latency_ms: int
    results: list[BenchmarkResult]
    tenant_id: str = "default"


@dataclass(slots=True)
class GroundedAnswer:
    answer: str
    citations: list[Citation]
    abstained: bool
    grounding_score: float
    latency_ms: int
    retrieval_latency_ms: int
    provider: str | None = None
    model: str | None = None
    trace: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    revision_count: int = 0
    retrieval_attempts: int = 0
    retrieved_chunks: list[Citation] = field(default_factory=list)


def to_summary(document: DocumentRecord) -> DocumentSummary:
    return DocumentSummary(
        id=document.id,
        title=document.title,
        filename=document.filename,
        file_size=document.file_size,
        page_count=document.page_count,
        chunk_count=len(document.chunks),
        status=document.status,
        embedding_status=document.embedding_status,
        uploaded_at=document.uploaded_at,
        summary=document.summary,
        error_message=document.error_message,
    )
