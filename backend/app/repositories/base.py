"""Repository contract shared by the in-memory and PostgreSQL backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain import (
    BenchmarkRunReport,
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentSummary,
    EmbeddingStatus,
)


@runtime_checkable
class DocumentRepository(Protocol):
    async def save_document(self, document: DocumentRecord) -> None: ...

    async def get_document(self, tenant_id: str, document_id: str) -> DocumentRecord | None: ...

    async def list_documents(
        self,
        tenant_id: str,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
    ) -> tuple[list[DocumentSummary], int]: ...

    async def delete_document(self, tenant_id: str, document_id: str) -> bool: ...

    async def delete_all_documents(self, tenant_id: str) -> int: ...

    async def update_status(
        self,
        tenant_id: str,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None: ...

    async def update_embedding_status(
        self,
        tenant_id: str,
        document_id: str,
        status: EmbeddingStatus,
    ) -> None: ...

    async def list_chunks(
        self,
        tenant_id: str,
        document_id: str | None = None,
        *,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        """Return chunks, optionally pre-filtered by dense/lexical relevance.

        Postgres uses the hints to run a bounded ANN + full-text UNION so the
        application never scans an unbounded corpus. The in-memory backend
        ignores the hints and returns everything up to ``limit``.
        """
        ...

    async def chunks_by_entities(
        self,
        tenant_id: str,
        entities: list[str],
        *,
        exclude_chunk_ids: set[str] | None = None,
        document_id: str | None = None,
        limit: int = 8,
    ) -> list[ChunkRecord]: ...

    async def replace_chunks(
        self,
        tenant_id: str,
        document_id: str,
        chunks: list[ChunkRecord],
    ) -> None: ...

    async def save_benchmark(self, report: BenchmarkRunReport) -> None: ...

    async def latest_benchmark(self, tenant_id: str) -> BenchmarkRunReport | None: ...

    async def list_benchmarks(
        self, tenant_id: str, limit: int = 10
    ) -> list[BenchmarkRunReport]: ...

    async def count_documents(self, tenant_id: str) -> int: ...

    async def save_tenant(self, tenant_id: str) -> None: ...

    async def ping(self) -> None: ...

    async def close(self) -> None: ...
