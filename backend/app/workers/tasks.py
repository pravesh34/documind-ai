"""Celery tasks. Each task builds its own short-lived dependency graph."""

from __future__ import annotations

import asyncio

from app.ai.embeddings import build_embeddings
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import Database
from app.rag.entities import EntityExtractor
from app.repositories.postgres import PostgresDocumentRepository
from app.services.ingestion import IngestionService
from app.workers.celery_app import celery_app

logger = get_logger("workers.tasks")


async def _reembed(tenant_id: str, document_id: str) -> None:
    settings = get_settings()
    if not settings.database_url:
        logger.warning("reembed skipped: DATABASE_URL is not configured")
        return

    database = Database(settings)
    try:
        repository = PostgresDocumentRepository(database)
        ingestion = IngestionService(
            repository,
            build_embeddings(settings),
            EntityExtractor(settings),
            settings,
        )
        await ingestion.reembed_document(tenant_id, document_id)
    finally:
        await database.dispose()


@celery_app.task(
    name="documind.reembed_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def reembed_document(self, tenant_id: str, document_id: str) -> dict[str, str]:
    configure_logging(get_settings().log_level, get_settings().environment)
    logger.info(
        "reembed task started",
        tenant_id=tenant_id,
        document_id=document_id,
        attempt=self.request.retries,
    )
    asyncio.run(_reembed(tenant_id, document_id))
    return {"status": "ok", "document_id": document_id}
