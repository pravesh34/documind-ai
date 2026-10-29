"""In-memory repository for local development and tests.

Production deployments set ``DATABASE_URL`` and use the PostgreSQL backend. The
in-memory store keeps the exact same semantics so the test suite exercises the
real service layer.
"""

from __future__ import annotations

import asyncio

from app.domain import (
    BenchmarkRunReport,
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentSummary,
    EmbeddingStatus,
    to_summary,
)


class MemoryDocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[str, dict[str, DocumentRecord]] = {}
        self._benchmarks: dict[str, list[BenchmarkRunReport]] = {}
        self._tenants: set[str] = {"default"}
        self._lock = asyncio.Lock()

    # Documents ----------------------------------------------------------
    async def save_document(self, document: DocumentRecord) -> None:
        async with self._lock:
            tenant = self._documents.setdefault(document.tenant_id, {})
            existing = tenant.get(document.id)
            if existing is not None:
                document.uploaded_at = existing.uploaded_at
            tenant[document.id] = document

    async def get_document(self, tenant_id: str, document_id: str) -> DocumentRecord | None:
        return self._documents.get(tenant_id, {}).get(document_id)

    async def list_documents(
        self,
        tenant_id: str,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
    ) -> tuple[list[DocumentSummary], int]:
        needle = (query or "").strip().lower()
        documents = [
            document
            for document in self._documents.get(tenant_id, {}).values()
            if not needle
            or needle in document.title.lower()
            or needle in document.filename.lower()
        ]
        documents.sort(key=lambda item: item.uploaded_at, reverse=True)
        total = len(documents)
        page = documents[offset : offset + limit]
        return [to_summary(document) for document in page], total

    async def delete_document(self, tenant_id: str, document_id: str) -> bool:
        async with self._lock:
            return self._documents.get(tenant_id, {}).pop(document_id, None) is not None

    async def delete_all_documents(self, tenant_id: str) -> int:
        async with self._lock:
            documents = self._documents.setdefault(tenant_id, {})
            count = len(documents)
            documents.clear()
            return count

    async def update_status(
        self,
        tenant_id: str,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None:
        document = self._documents.get(tenant_id, {}).get(document_id)
        if document is not None:
            document.status = status
            document.error_message = error_message

    async def update_embedding_status(
        self,
        tenant_id: str,
        document_id: str,
        status: EmbeddingStatus,
    ) -> None:
        document = self._documents.get(tenant_id, {}).get(document_id)
        if document is not None:
            document.embedding_status = status

    # Chunks -------------------------------------------------------------
    async def list_chunks(
        self,
        tenant_id: str,
        document_id: str | None = None,
        *,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for document in self._documents.get(tenant_id, {}).values():
            if document.status != "ready":
                continue
            if document_id and document.id != document_id:
                continue
            chunks.extend(document.chunks)
        chunks.sort(key=lambda chunk: (chunk.document_id, chunk.chunk_index))
        # The in-memory backend cannot pre-filter dense/lexical candidates, so
        # it returns the whole (dev-sized) pool when search hints are present.
        if query_text or query_embedding:
            return chunks
        return chunks[:limit] if limit else chunks

    async def chunks_by_entities(
        self,
        tenant_id: str,
        entities: list[str],
        *,
        exclude_chunk_ids: set[str] | None = None,
        document_id: str | None = None,
        limit: int = 8,
    ) -> list[ChunkRecord]:
        wanted = {entity.strip().lower() for entity in entities if entity.strip()}
        if not wanted:
            return []

        excluded = exclude_chunk_ids or set()
        matches: list[tuple[int, ChunkRecord]] = []

        for document in self._documents.get(tenant_id, {}).values():
            if document.status != "ready":
                continue
            if document_id and document.id != document_id:
                continue
            for chunk in document.chunks:
                if chunk.chunk_id in excluded:
                    continue
                overlap = len(wanted & {entity.lower() for entity in chunk.entities})
                if overlap:
                    matches.append((overlap, chunk))

        matches.sort(key=lambda item: (-item[0], item[1].chunk_index))
        return [chunk for _, chunk in matches[:limit]]

    async def replace_chunks(
        self,
        tenant_id: str,
        document_id: str,
        chunks: list[ChunkRecord],
    ) -> None:
        document = self._documents.get(tenant_id, {}).get(document_id)
        if document is not None:
            document.chunks = chunks

    # Benchmarks ---------------------------------------------------------
    async def save_benchmark(self, report: BenchmarkRunReport) -> None:
        history = self._benchmarks.setdefault(report.tenant_id, [])
        history.insert(0, report)
        del history[20:]

    async def latest_benchmark(self, tenant_id: str) -> BenchmarkRunReport | None:
        history = self._benchmarks.get(tenant_id, [])
        return history[0] if history else None

    async def list_benchmarks(
        self, tenant_id: str, limit: int = 10
    ) -> list[BenchmarkRunReport]:
        return self._benchmarks.get(tenant_id, [])[:limit]

    # Misc ---------------------------------------------------------------
    async def count_documents(self, tenant_id: str) -> int:
        return len(self._documents.get(tenant_id, {}))

    async def save_tenant(self, tenant_id: str) -> None:
        self._tenants.add(tenant_id)

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        self._documents.clear()
        self._benchmarks.clear()
