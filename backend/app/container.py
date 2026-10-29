"""Dependency container shared by the API and workers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.ai.embeddings import FallbackEmbeddings, build_embeddings
from app.ai.llm import LLMService
from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.init_db import initialize_database
from app.db.session import Database
from app.rag.entities import EntityExtractor
from app.rag.retrieval import HybridRetriever
from app.rag.workflow import RAGWorkflow
from app.repositories.base import DocumentRepository
from app.repositories.memory import MemoryDocumentRepository
from app.repositories.postgres import PostgresDocumentRepository
from app.services.chat import ChatService
from app.services.documents import DocumentService
from app.services.evaluation import EvaluationService
from app.services.export import ExportService
from app.services.ingestion import IngestionService

logger = get_logger("container")


@dataclass
class Container:
    settings: Settings
    repository: DocumentRepository
    embeddings: FallbackEmbeddings
    llm: LLMService
    entity_extractor: EntityExtractor
    ingestion: IngestionService
    retriever: HybridRetriever
    workflow: RAGWorkflow
    documents: DocumentService
    chat: ChatService
    evaluation: EvaluationService
    exports: ExportService
    database: Database | None = None
    ready: bool = False
    started_at: float = field(default_factory=time.time)

    @classmethod
    async def create(cls, settings: Settings | None = None) -> Container:
        settings = settings or get_settings()

        database: Database | None = None
        repository: DocumentRepository
        if settings.uses_postgres:
            database = Database(settings)
            repository = PostgresDocumentRepository(database)
        else:
            repository = MemoryDocumentRepository()

        embeddings = build_embeddings(settings)
        llm = LLMService(settings)
        entity_extractor = EntityExtractor(settings, llm)

        ingestion = IngestionService(repository, embeddings, entity_extractor, settings)
        ingestion.set_background_handler(ingestion.reembed_document)
        retriever = HybridRetriever(repository, embeddings, settings)
        workflow = RAGWorkflow(repository, retriever, llm, settings)

        return cls(
            settings=settings,
            repository=repository,
            embeddings=embeddings,
            llm=llm,
            entity_extractor=entity_extractor,
            ingestion=ingestion,
            retriever=retriever,
            workflow=workflow,
            documents=DocumentService(repository, ingestion, settings),
            chat=ChatService(workflow, retriever, settings),
            evaluation=EvaluationService(repository, workflow, settings),
            exports=ExportService(settings),
            database=database,
        )

    async def startup(self) -> None:
        if self.database is not None and self.settings.auto_create_schema:
            await initialize_database(self.database, self.settings)

        await self.repository.save_tenant("default")
        await self.documents.ensure_seed("default")
        self.ready = True

        logger.info(
            "container ready",
            backend="postgres" if self.database else "memory",
            chat=self.llm.provider_chain or ["extractive"],
            embeddings=self.embeddings.provider_chain or ["deterministic"],
            documents=await self.repository.count_documents("default"),
        )

    async def shutdown(self) -> None:
        self.ready = False
        try:
            await self.ingestion.drain()
        finally:
            await self.repository.close()

    async def healthy(self) -> bool:
        try:
            await self.repository.ping()
            return True
        except Exception as error:  # noqa: BLE001 - readiness probe
            logger.warning("readiness check failed", error=str(error))
            return False
