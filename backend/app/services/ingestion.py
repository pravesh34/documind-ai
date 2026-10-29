"""Document ingestion: extraction, chunking, entity extraction, embeddings."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from app.ai.embeddings import FallbackEmbeddings, embed_in_batches
from app.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.core.utils import sha256_bytes, sha256_text, truncate
from app.domain import ChunkRecord, DocumentPage, DocumentRecord, utc_now
from app.rag.chunking import split_pages, to_chunk_records
from app.rag.entities import EntityExtractor
from app.rag.extraction import detect_mime_type, extract_pdf_pages, split_text_into_pages
from app.repositories.base import DocumentRepository

logger = get_logger("services.ingestion")

ReembedHandler = Callable[[str, str], Awaitable[None]]


class IngestionService:
    def __init__(
        self,
        repository: DocumentRepository,
        embeddings: FallbackEmbeddings,
        entity_extractor: EntityExtractor,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self.entity_extractor = entity_extractor
        self.settings = settings
        self._on_ingested: ReembedHandler | None = None
        self._background_tasks: set[asyncio.Task] = set()

    def set_background_handler(self, handler: ReembedHandler) -> None:
        self._on_ingested = handler

    async def ingest(
        self,
        tenant_id: str,
        *,
        title: str,
        filename: str,
        text: str | None = None,
        data: bytes | None = None,
        mime_type: str | None = None,
        file_size: int | None = None,
    ) -> DocumentRecord:
        title = title.strip()
        if not title:
            raise AppError(422, "VALIDATION_ERROR", "Document title is required.")
        if len(title) > 300:
            raise AppError(422, "VALIDATION_ERROR", "Document title must be <= 300 characters.")

        filename = (filename or f"{title}.txt").strip() or f"{title}.txt"
        filename = filename.replace("\\", "/").split("/")[-1][:255]

        if data is None and not (text and text.strip()):
            raise AppError(
                422,
                "VALIDATION_ERROR",
                "Provide a PDF file or non-empty text content.",
            )

        if text and len(text) > self.settings.max_text_chars:
            raise AppError(
                413,
                "PAYLOAD_TOO_LARGE",
                f"Text content exceeds {self.settings.max_text_chars} characters.",
            )

        resolved_mime = detect_mime_type(filename, mime_type)
        if data is not None:
            checksum = sha256_bytes(data)
            actual_size = len(data)
        else:
            checksum = sha256_text(text or "")
            actual_size = file_size or len((text or "").encode("utf-8"))

        pages = await self._extract_pages(filename, resolved_mime, data, text)
        drafts = split_pages(pages, self.settings)
        if not drafts:
            raise AppError(
                422,
                "EMPTY_DOCUMENT",
                "No extractable text was found in the uploaded document.",
            )

        document_id = str(uuid.uuid4())
        chunks = to_chunk_records(
            drafts,
            tenant_id=tenant_id,
            document_id=document_id,
            document_title=title,
        )

        await self._enrich_entities(chunks)
        for chunk in chunks:
            chunk.embedding = self.embeddings.deterministic.embed_query(chunk.text)

        document = DocumentRecord(
            id=document_id,
            tenant_id=tenant_id,
            title=title,
            filename=filename,
            file_size=actual_size,
            mime_type=resolved_mime,
            checksum=checksum,
            page_count=len([page for page in pages if page.text.strip()]) or len(pages),
            status="ready",
            embedding_status="pending" if self.embeddings.remote_enabled else "ready",
            uploaded_at=utc_now(),
            pages=pages,
            chunks=chunks,
            summary=summarise(pages),
        )

        await self.repository.save_document(document)

        logger.info(
            "document ingested",
            document_id=document.id,
            chunks=len(chunks),
            pages=document.page_count,
            tenant=tenant_id,
        )

        if (
            self.settings.background_enrichment
            and self.embeddings.remote_enabled
            and self._on_ingested is not None
        ):
            await self._schedule_reembed(tenant_id, document.id)

        return document

    async def reembed_document(self, tenant_id: str, document_id: str) -> None:
        document = await self.repository.get_document(tenant_id, document_id)
        if document is None or not document.chunks:
            return

        try:
            vectors = await embed_in_batches(
                self.embeddings,
                [chunk.text for chunk in document.chunks],
                batch_size=self.settings.embedding_batch_size,
                concurrency=self.settings.embedding_concurrency,
            )
            for chunk, vector in zip(document.chunks, vectors, strict=False):
                if vector and any(vector):
                    chunk.embedding = vector
            await self.repository.replace_chunks(tenant_id, document_id, document.chunks)
            await self.repository.update_embedding_status(tenant_id, document_id, "ready")
            logger.info(
                "document re-embedded with remote providers",
                document_id=document_id,
                chunks=len(vectors),
            )
        except Exception as error:  # noqa: BLE001 - fall back to deterministic vectors
            await self.repository.update_embedding_status(
                tenant_id, document_id, "unavailable"
            )
            logger.warning(
                "remote re-embedding failed; keeping deterministic vectors",
                document_id=document_id,
                error=str(error),
            )

    async def _extract_pages(
        self,
        filename: str,
        mime_type: str,
        data: bytes | None,
        text: str | None,
    ) -> list[DocumentPage]:
        if data is not None:
            if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
                return await asyncio.to_thread(extract_pdf_pages, data)
            try:
                decoded = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise AppError(
                    415,
                    "UNSUPPORTED_MEDIA_TYPE",
                    "Only UTF-8 text and PDF uploads are supported.",
                ) from error
            return split_text_into_pages(decoded)

        return split_text_into_pages(text or "")

    async def _enrich_entities(self, chunks: list[ChunkRecord]) -> None:
        if not self.settings.entity_extraction_enabled or not chunks:
            return

        if self.settings.entity_extraction_mode != "llm":
            for chunk in chunks:
                chunk.entities = await self.entity_extractor.extract(chunk.text)
            return

        semaphore = asyncio.Semaphore(self.settings.embedding_concurrency)

        async def run(chunk: ChunkRecord) -> None:
            async with semaphore:
                chunk.entities = await self.entity_extractor.extract(chunk.text)

        await asyncio.gather(*(run(chunk) for chunk in chunks))

    async def _schedule_reembed(self, tenant_id: str, document_id: str) -> None:
        handler = self._on_ingested
        if handler is None:
            return

        if self.settings.celery_broker_url:
            try:
                from app.workers.celery_app import celery_app

                celery_app.send_task(
                    "documind.reembed_document",
                    args=[tenant_id, document_id],
                    queue="enrichment",
                )
                return
            except Exception as error:  # noqa: BLE001 - fall back to in-process task
                logger.warning("celery enqueue failed", error=str(error))

        task = asyncio.create_task(handler(tenant_id, document_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def drain(self, max_wait: float = 10.0) -> None:
        if not self._background_tasks:
            return
        await asyncio.wait(set(self._background_tasks), timeout=max_wait)


def summarise(pages: list[DocumentPage]) -> str:
    for page in pages:
        for paragraph in page.text.split("\n\n"):
            cleaned = " ".join(paragraph.split())
            if len(cleaned) >= 80:
                return truncate(cleaned, 280)
    for page in pages:
        if page.text.strip():
            return truncate(page.text, 280)
    return "No extractable summary."
