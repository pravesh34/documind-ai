"""Document library service: listing, deletion and default-corpus seeding."""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.domain import DocumentRecord, DocumentSummary
from app.repositories.base import DocumentRepository
from app.services.ingestion import IngestionService
from app.services.seed_data import DEFAULT_DOCUMENTS

logger = get_logger("services.documents")


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        ingestion: IngestionService,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.ingestion = ingestion
        self.settings = settings
        self._seed_lock = asyncio.Lock()

    async def list_documents(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> tuple[list[DocumentSummary], int]:
        return await self.repository.list_documents(
            tenant_id, limit=limit, offset=offset, query=query
        )

    async def get_document(self, tenant_id: str, document_id: str) -> DocumentRecord:
        document = await self.repository.get_document(tenant_id, document_id)
        if document is None:
            raise AppError(404, "DOCUMENT_NOT_FOUND", "Document not found.")
        return document

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        deleted = await self.repository.delete_document(tenant_id, document_id)
        if not deleted:
            raise AppError(404, "DOCUMENT_NOT_FOUND", "Document not found.")

    async def reset_defaults(self, tenant_id: str) -> int:
        async with self._seed_lock:
            await self.repository.delete_all_documents(tenant_id)
            return await self._seed(tenant_id, replace=True)

    async def ensure_seed(self, tenant_id: str) -> int:
        if not self.settings.seed_default_documents:
            return 0
        async with self._seed_lock:
            if await self.repository.count_documents(tenant_id) > 0:
                return 0
            return await self._seed(tenant_id, replace=False)

    async def seed_if_missing(self, tenant_id: str) -> int:
        return await self.ensure_seed(tenant_id)

    async def _seed(self, tenant_id: str, *, replace: bool) -> int:
        existing_titles: set[str] = set()
        if not replace:
            existing, _ = await self.repository.list_documents(
                tenant_id, limit=200, offset=0
            )
            existing_titles = {item.title.lower() for item in existing}

        created = 0
        for spec in DEFAULT_DOCUMENTS:
            if spec.title.lower() in existing_titles:
                continue
            try:
                await self.ingestion.ingest(
                    tenant_id,
                    title=spec.title,
                    filename=spec.filename,
                    text=spec.text,
                    mime_type=spec.mime_type,
                )
                created += 1
            except Exception as error:  # noqa: BLE001 - never block startup
                logger.error(
                    "failed to seed default document",
                    title=spec.title,
                    error=str(error),
                )

        if created:
            logger.info("seeded default corpus", documents=created, tenant=tenant_id)
        return created
