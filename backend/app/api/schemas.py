"""Pydantic v2 request/response contracts (camelCase on the wire)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(word.capitalize() for word in rest)


class APISchema(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class ChatTurn(APISchema):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class UploadRequest(APISchema):
    title: str = Field(min_length=1, max_length=300)
    filename: str | None = Field(default=None, max_length=255)
    file_base64: str | None = None
    text_content: str | None = Field(default=None, max_length=5_000_000)
    file_size: int | None = Field(default=None, ge=0)
    mime_type: str | None = Field(default=None, max_length=120)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title must not be blank")
        return cleaned


class ChatRequest(APISchema):
    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=4, ge=1, le=12)
    similarity_threshold: float = Field(default=0.28, ge=0.0, le=1.0)
    target_document_id: str | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=24)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class RetrievalRequestModel(APISchema):
    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=6, ge=1, le=12)
    similarity_threshold: float = Field(default=0.28, ge=0.0, le=1.0)
    target_document_id: str | None = None


class EvalTestCaseInput(APISchema):
    id: str | None = Field(default=None, max_length=120)
    document_title: str = Field(default="", max_length=300)
    question: str = Field(min_length=3, max_length=2_000)
    expected_passage: str = Field(default="", max_length=4_000)
    expected_page: int = Field(default=0, ge=0, le=10_000)
    is_negative_test: bool = False
    description: str = Field(default="", max_length=500)


class EvalRunRequest(APISchema):
    custom_tests: list[EvalTestCaseInput] | None = Field(default=None, max_length=100)


class CitationModel(APISchema):
    document_id: str
    document_title: str
    page_number: int
    chunk_id: str
    excerpt: str
    similarity_score: float
    lexical_score: float = 0.0
    combined_score: float = 0.0
    source: str = "dense"


class ChatResponse(APISchema):
    answer: str
    citations: list[CitationModel]
    abstained: bool
    grounding_score: float
    latency_ms: int
    retrieval_latency_ms: int
    queries: list[str]
    trace: list[dict]
    provider: str | None = None
    model: str | None = None
    revision_count: int = 0
    retrieval_attempts: int = 0
    retrieved_chunks: list[CitationModel] = Field(default_factory=list)


class RetrievalResponse(APISchema):
    query: str
    queries: list[str]
    citations: list[CitationModel]
    total_candidates: int
    latency_ms: int
