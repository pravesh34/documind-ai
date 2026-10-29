"""PostgreSQL + pgvector repository.

Hybrid candidate prefetch runs entirely inside PostgreSQL: an HNSW cosine
search and a GIN full-text search are UNION-ed before the application reranks.
Graph expansion joins the ``chunk_entities`` table.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.serialization import benchmark_from_dict, benchmark_to_dict
from app.db.session import Database
from app.domain import (
    BenchmarkRunReport,
    ChunkRecord,
    DocumentPage,
    DocumentRecord,
    DocumentStatus,
    DocumentSummary,
    EmbeddingStatus,
    to_summary,
)

logger = get_logger("db.postgres")

CHUNK_COLUMNS = """
  c.chunk_id, c.tenant_id, c.document_id::text AS document_id, c.document_title,
  c.page_number, c.chunk_index, c.text, c.embedding::text AS embedding,
  c.token_count, c.start_char, c.end_char
"""

DOCUMENT_COLUMNS = """
  d.id::text AS id, d.tenant_id, d.title, d.filename, d.file_size, d.mime_type,
  d.checksum, d.page_count, d.status, d.embedding_status, d.summary,
  d.error_message, d.pages, d.created_at
"""


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _parse_embedding(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    return []


class PostgresDocumentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    # Documents ----------------------------------------------------------
    async def save_document(self, document: DocumentRecord) -> None:
        async with self.database.session() as session:
            await self._ensure_tenant(session, document.tenant_id)
            await session.execute(
                text(
                    """
                    INSERT INTO documents (
                      id, tenant_id, title, filename, file_size, mime_type, checksum,
                      page_count, status, embedding_status, summary, error_message, pages
                    ) VALUES (
                      CAST(:id AS uuid), :tenant_id, :title, :filename, :file_size,
                      :mime_type, :checksum, :page_count, :status, :embedding_status,
                      :summary, :error_message, CAST(:pages AS jsonb)
                    )
                    ON CONFLICT (id) DO UPDATE SET
                      title = EXCLUDED.title,
                      filename = EXCLUDED.filename,
                      file_size = EXCLUDED.file_size,
                      mime_type = EXCLUDED.mime_type,
                      checksum = EXCLUDED.checksum,
                      page_count = EXCLUDED.page_count,
                      status = EXCLUDED.status,
                      embedding_status = EXCLUDED.embedding_status,
                      summary = EXCLUDED.summary,
                      error_message = EXCLUDED.error_message,
                      pages = EXCLUDED.pages,
                      updated_at = now()
                    """
                ),
                {
                    "id": document.id,
                    "tenant_id": document.tenant_id,
                    "title": document.title,
                    "filename": document.filename,
                    "file_size": document.file_size,
                    "mime_type": document.mime_type,
                    "checksum": document.checksum,
                    "page_count": document.page_count,
                    "status": document.status,
                    "embedding_status": document.embedding_status,
                    "summary": document.summary,
                    "error_message": document.error_message,
                    "pages": json.dumps(
                        [
                            {
                                "pageNumber": page.page_number,
                                "text": page.text,
                                "charCount": page.char_count,
                            }
                            for page in document.pages
                        ]
                    ),
                },
            )
            await session.execute(
                text("DELETE FROM chunks WHERE document_id = CAST(:id AS uuid)"),
                {"id": document.id},
            )
            await self._insert_chunks(session, document.chunks)
            await session.execute(
                text("UPDATE documents SET updated_at = now() WHERE id = CAST(:id AS uuid)"),
                {"id": document.id},
            )

    async def get_document(self, tenant_id: str, document_id: str) -> DocumentRecord | None:
        async with self.database.session() as session:
            result = await session.execute(
                text(
                    f"SELECT {DOCUMENT_COLUMNS} FROM documents d "
                    "WHERE d.tenant_id = :tenant AND d.id = CAST(:id AS uuid)"
                ),
                {"tenant": tenant_id, "id": document_id},
            )
            row = result.mappings().first()
            if row is None:
                return None

            chunks = await self._load_chunks(
                session,
                f"""
                SELECT {CHUNK_COLUMNS} FROM chunks c
                WHERE c.tenant_id = :tenant AND c.document_id = CAST(:id AS uuid)
                ORDER BY c.chunk_index
                """,
                {"tenant": tenant_id, "id": document_id},
            )
            return _document_from_row(row, chunks)

    async def list_documents(
        self,
        tenant_id: str,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
    ) -> tuple[list[DocumentSummary], int]:
        search = (query or "").strip()
        pattern = f"%{search.replace('%', r'\%').replace('_', r'\_')}%"
        filter_sql = ""
        params: dict[str, Any] = {"tenant": tenant_id}
        if search:
            filter_sql = "AND (d.title ILIKE :pattern OR d.filename ILIKE :pattern)"
            params["pattern"] = pattern

        async with self.database.session() as session:
            rows = await session.execute(
                text(
                    f"""
                    SELECT {DOCUMENT_COLUMNS},
                      (SELECT count(*)::int FROM chunks c WHERE c.document_id = d.id) AS chunk_count
                    FROM documents d
                    WHERE d.tenant_id = :tenant {filter_sql}
                    ORDER BY d.created_at DESC
                    LIMIT :limit OFFSET :offset
                    """
                ),
                {**params, "limit": limit, "offset": offset},
            )
            total = await session.execute(
                text(
                    f"SELECT count(*)::int AS total FROM documents d "
                    f"WHERE d.tenant_id = :tenant {filter_sql}"
                ),
                params,
            )

            summaries = []
            for row in rows.mappings():
                document = _document_from_row(row, [])
                summary = to_summary(document)
                summary.chunk_count = int(row.get("chunk_count") or 0)
                summaries.append(summary)

            return summaries, int(total.scalar_one() or 0)

    async def delete_document(self, tenant_id: str, document_id: str) -> bool:
        async with self.database.session() as session:
            result = await session.execute(
                text(
                    "DELETE FROM documents "
                    "WHERE tenant_id = :tenant AND id = CAST(:id AS uuid)"
                ),
                {"tenant": tenant_id, "id": document_id},
            )
            return (result.rowcount or 0) > 0

    async def delete_all_documents(self, tenant_id: str) -> int:
        async with self.database.session() as session:
            result = await session.execute(
                text("DELETE FROM documents WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
            return result.rowcount or 0

    async def update_status(
        self,
        tenant_id: str,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None:
        async with self.database.session() as session:
            await session.execute(
                text(
                    """
                    UPDATE documents
                    SET status = :status, error_message = :error_message, updated_at = now()
                    WHERE tenant_id = :tenant AND id = CAST(:id AS uuid)
                    """
                ),
                {
                    "tenant": tenant_id,
                    "id": document_id,
                    "status": status,
                    "error_message": error_message,
                },
            )

    async def update_embedding_status(
        self,
        tenant_id: str,
        document_id: str,
        status: EmbeddingStatus,
    ) -> None:
        async with self.database.session() as session:
            await session.execute(
                text(
                    """
                    UPDATE documents
                    SET embedding_status = :status, updated_at = now()
                    WHERE tenant_id = :tenant AND id = CAST(:id AS uuid)
                    """
                ),
                {"tenant": tenant_id, "id": document_id, "status": status},
            )

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
        branch_limit = max(1, limit or 32)
        params: dict[str, Any] = {"tenant": tenant_id}
        doc_filter = ""
        if document_id:
            doc_filter = "AND document_id = CAST(:document_id AS uuid)"
            params["document_id"] = document_id

        async with self.database.session() as session:
            if query_embedding or query_text:
                ctes: list[str] = []
                if query_embedding:
                    params["embedding"] = vector_literal(query_embedding)
                    params["branch_limit"] = branch_limit
                    ctes.append(
                        f"""
                        dense AS (
                          SELECT chunk_id FROM chunks
                          WHERE tenant_id = :tenant {doc_filter}
                          ORDER BY embedding <=> CAST(:embedding AS vector)
                          LIMIT :branch_limit
                        )
                        """
                    )
                if query_text and query_text.strip():
                    params["query_text"] = query_text.strip()
                    params["branch_limit"] = branch_limit
                    ctes.append(
                        f"""
                        lexical AS (
                          SELECT chunk_id FROM chunks
                          WHERE tenant_id = :tenant {doc_filter}
                            AND tsv @@ plainto_tsquery('english', :query_text)
                          ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', :query_text)) DESC
                          LIMIT :branch_limit
                        )
                        """
                    )

                union = " UNION ".join(f"SELECT chunk_id FROM {name}" for name in (
                    "dense" if query_embedding else None,
                    "lexical" if query_text and query_text.strip() else None,
                ) if name)
                params["overall_limit"] = branch_limit * 2

                sql = (
                    f"WITH {', '.join(ctes)}, ids AS ({union}) "
                    f"SELECT {CHUNK_COLUMNS} FROM chunks c "
                    "JOIN ids ON ids.chunk_id = c.chunk_id "
                    "ORDER BY c.document_id, c.chunk_index "
                    "LIMIT :overall_limit"
                )
                return await self._load_chunks(session, sql, params)

            params["limit"] = limit or 1_000
            doc_filter_plain = (
                "AND c.document_id = CAST(:document_id AS uuid)" if document_id else ""
            )
            sql = (
                f"SELECT {CHUNK_COLUMNS} FROM chunks c "
                f"WHERE c.tenant_id = :tenant {doc_filter_plain} "
                "ORDER BY c.document_id, c.chunk_index LIMIT :limit"
            )
            return await self._load_chunks(session, sql, params)

    async def chunks_by_entities(
        self,
        tenant_id: str,
        entities: list[str],
        *,
        exclude_chunk_ids: set[str] | None = None,
        document_id: str | None = None,
        limit: int = 8,
    ) -> list[ChunkRecord]:
        wanted = [entity.strip() for entity in entities if entity.strip()]
        if not wanted:
            return []

        params: dict[str, Any] = {
            "tenant": tenant_id,
            "entities": json.dumps([entity.lower() for entity in wanted]),
            "exclude": json.dumps(sorted(exclude_chunk_ids or set())),
            "limit": limit,
        }
        document_filter = ""
        if document_id:
            document_filter = "AND c.document_id = CAST(:document_id AS uuid)"
            params["document_id"] = document_id

        sql = f"""
            SELECT {CHUNK_COLUMNS}, count(*) AS entity_hits
            FROM chunks c
            JOIN chunk_entities e ON e.chunk_id = c.chunk_id
            WHERE c.tenant_id = :tenant
              {document_filter}
              AND lower(e.entity) IN (
                SELECT lower(value) FROM jsonb_array_elements_text(CAST(:entities AS jsonb))
              )
              AND c.chunk_id NOT IN (
                SELECT value FROM jsonb_array_elements_text(CAST(:exclude AS jsonb))
              )
            GROUP BY c.chunk_id
            ORDER BY entity_hits DESC, c.chunk_index ASC
            LIMIT :limit
        """

        async with self.database.session() as session:
            return await self._load_chunks(session, sql, params)

    async def replace_chunks(
        self,
        tenant_id: str,
        document_id: str,
        chunks: list[ChunkRecord],
    ) -> None:
        async with self.database.session() as session:
            await session.execute(
                text("DELETE FROM chunks WHERE document_id = CAST(:id AS uuid)"),
                {"id": document_id},
            )
            await self._insert_chunks(session, chunks)

    # Benchmarks ---------------------------------------------------------
    async def save_benchmark(self, report: BenchmarkRunReport) -> None:
        async with self.database.session() as session:
            await self._ensure_tenant(session, report.tenant_id)
            await session.execute(
                text(
                    """
                    INSERT INTO benchmark_reports (run_id, tenant_id, payload)
                    VALUES (:run_id, :tenant, CAST(:payload AS jsonb))
                    ON CONFLICT (run_id) DO UPDATE SET payload = EXCLUDED.payload
                    """
                ),
                {
                    "run_id": report.run_id,
                    "tenant": report.tenant_id,
                    "payload": json.dumps(benchmark_to_dict(report)),
                },
            )

    async def latest_benchmark(self, tenant_id: str) -> BenchmarkRunReport | None:
        async with self.database.session() as session:
            result = await session.execute(
                text(
                    """
                    SELECT payload FROM benchmark_reports
                    WHERE tenant_id = :tenant ORDER BY created_at DESC LIMIT 1
                    """
                ),
                {"tenant": tenant_id},
            )
            row = result.first()
            return _report_from_row(row[0]) if row else None

    async def list_benchmarks(
        self, tenant_id: str, limit: int = 10
    ) -> list[BenchmarkRunReport]:
        async with self.database.session() as session:
            result = await session.execute(
                text(
                    """
                    SELECT payload FROM benchmark_reports
                    WHERE tenant_id = :tenant ORDER BY created_at DESC LIMIT :limit
                    """
                ),
                {"tenant": tenant_id, "limit": limit},
            )
            return [_report_from_row(row[0]) for row in result if row and row[0]]

    # Misc ---------------------------------------------------------------
    async def count_documents(self, tenant_id: str) -> int:
        async with self.database.session() as session:
            result = await session.execute(
                text("SELECT count(*)::int FROM documents WHERE tenant_id = :tenant"),
                {"tenant": tenant_id},
            )
            return int(result.scalar_one() or 0)

    async def save_tenant(self, tenant_id: str) -> None:
        async with self.database.session() as session:
            await self._ensure_tenant(session, tenant_id)

    async def ping(self) -> None:
        async with self.database.session() as session:
            await session.execute(text("SELECT 1"))

    async def close(self) -> None:
        await self.database.dispose()

    # Internals ----------------------------------------------------------
    async def _ensure_tenant(self, session: AsyncSession, tenant_id: str) -> None:
        await session.execute(
            text(
                "INSERT INTO tenants (id) VALUES (:tenant) ON CONFLICT (id) DO NOTHING"
            ),
            {"tenant": tenant_id},
        )

    async def _insert_chunks(
        self,
        session: AsyncSession,
        chunks: list[ChunkRecord],
    ) -> None:
        if not chunks:
            return

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "tenant_id": chunk.tenant_id,
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": vector_literal(chunk.embedding),
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in chunks
        ]

        for offset in range(0, len(rows), 100):
            batch = rows[offset : offset + 100]
            await session.execute(
                text(
                    """
                    INSERT INTO chunks (
                      chunk_id, tenant_id, document_id, document_title, page_number,
                      chunk_index, text, embedding, token_count, start_char, end_char
                    ) VALUES (
                      :chunk_id, :tenant_id, CAST(:document_id AS uuid), :document_title,
                      :page_number, :chunk_index, :text, CAST(:embedding AS vector),
                      :token_count, :start_char, :end_char
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      document_title = EXCLUDED.document_title,
                      page_number = EXCLUDED.page_number,
                      chunk_index = EXCLUDED.chunk_index,
                      text = EXCLUDED.text,
                      embedding = EXCLUDED.embedding,
                      token_count = EXCLUDED.token_count,
                      start_char = EXCLUDED.start_char,
                      end_char = EXCLUDED.end_char
                    """
                ),
                batch,
            )

        entity_rows = [
            {
                "chunk_id": chunk.chunk_id,
                "tenant_id": chunk.tenant_id,
                "entity": entity.strip().lower(),
            }
            for chunk in chunks
            for entity in chunk.entities
            if entity.strip()
        ]
        for offset in range(0, len(entity_rows), 1_000):
            await session.execute(
                text(
                    """
                    INSERT INTO chunk_entities (chunk_id, tenant_id, entity)
                    VALUES (:chunk_id, :tenant_id, :entity)
                    ON CONFLICT (chunk_id, entity) DO NOTHING
                    """
                ),
                entity_rows[offset : offset + 1_000],
            )

    async def _load_chunks(
        self,
        session: AsyncSession,
        sql: str,
        params: dict[str, Any],
    ) -> list[ChunkRecord]:
        result = await session.execute(text(sql), params)
        rows = result.mappings().all()
        if not rows:
            return []

        chunk_ids = [str(row["chunk_id"]) for row in rows]
        entities = await self._load_entities(session, chunk_ids)
        return [_chunk_from_row(row, entities.get(str(row["chunk_id"]), [])) for row in rows]

    async def _load_entities(
        self, session: AsyncSession, chunk_ids: list[str]
    ) -> dict[str, list[str]]:
        if not chunk_ids:
            return {}
        result = await session.execute(
            text(
                """
                SELECT chunk_id, entity FROM chunk_entities
                WHERE chunk_id IN (
                  SELECT value FROM jsonb_array_elements_text(CAST(:ids AS jsonb))
                )
                """
            ),
            {"ids": json.dumps(chunk_ids)},
        )
        mapping: dict[str, list[str]] = {}
        for chunk_id, entity in result:
            mapping.setdefault(str(chunk_id), []).append(str(entity))
        return mapping


def _document_from_row(row: Any, chunks: list[ChunkRecord]) -> DocumentRecord:
    pages_payload = row["pages"]
    if isinstance(pages_payload, str):
        try:
            pages_payload = json.loads(pages_payload)
        except json.JSONDecodeError:
            pages_payload = []

    pages = [
        DocumentPage(
            page_number=int(item.get("pageNumber", index + 1)),
            text=str(item.get("text", "")),
            char_count=int(item.get("charCount", len(str(item.get("text", ""))))),
        )
        for index, item in enumerate(pages_payload or [])
        if isinstance(item, dict)
    ]

    return DocumentRecord(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        title=str(row["title"]),
        filename=str(row["filename"]),
        file_size=int(row["file_size"] or 0),
        mime_type=str(row["mime_type"]),
        checksum=str(row["checksum"]),
        page_count=int(row["page_count"] or 0),
        status=str(row["status"]),  # type: ignore[arg-type]
        embedding_status=str(row["embedding_status"]),  # type: ignore[arg-type]
        uploaded_at=row["created_at"],
        pages=pages,
        chunks=chunks,
        summary=row["summary"],
        error_message=row["error_message"],
    )


def _chunk_from_row(row: Any, entities: list[str]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(row["chunk_id"]),
        tenant_id=str(row["tenant_id"]),
        document_id=str(row["document_id"]),
        document_title=str(row["document_title"]),
        page_number=int(row["page_number"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        embedding=_parse_embedding(row.get("embedding")),
        entities=entities,
        token_count=int(row["token_count"] or 0),
        start_char=int(row["start_char"] or 0),
        end_char=int(row["end_char"] or 0),
    )


def _report_from_row(payload: Any) -> BenchmarkRunReport:
    if isinstance(payload, str):
        payload = json.loads(payload)
    return benchmark_from_dict(payload)
