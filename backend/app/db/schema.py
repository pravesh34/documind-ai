"""Idempotent PostgreSQL + pgvector schema."""

from __future__ import annotations

from app.core.errors import AppError

SCHEMA_VERSION = 1


def validate_dimensions(dimensions: int) -> int:
    if not isinstance(dimensions, int) or not 64 <= dimensions <= 2_000:
        raise AppError(
            500,
            "INVALID_EMBEDDING_DIMENSIONS",
            f"Unsupported EMBEDDING_DIMENSIONS={dimensions}; expected 64-2000.",
            expose=False,
        )
    return dimensions


def extension_sql() -> str:
    return "CREATE EXTENSION IF NOT EXISTS vector;"


def schema_statements(dimensions: int) -> list[str]:
    dimensions = validate_dimensions(dimensions)

    return [
        """
        CREATE TABLE IF NOT EXISTS tenants (
          id         text PRIMARY KEY,
          created_at timestamptz NOT NULL DEFAULT now()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
          id               uuid PRIMARY KEY,
          tenant_id        text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
          title            text NOT NULL,
          filename         text NOT NULL,
          file_size        bigint NOT NULL DEFAULT 0,
          mime_type        text NOT NULL DEFAULT 'application/octet-stream',
          checksum         text NOT NULL DEFAULT '',
          page_count       integer NOT NULL DEFAULT 0,
          status           text NOT NULL DEFAULT 'pending'
                             CHECK (status IN ('pending', 'processing', 'ready', 'error')),
          embedding_status text NOT NULL DEFAULT 'pending'
                             CHECK (embedding_status IN ('pending', 'ready', 'unavailable')),
          summary          text,
          error_message    text,
          pages            jsonb NOT NULL DEFAULT '[]'::jsonb,
          created_at       timestamptz NOT NULL DEFAULT now(),
          updated_at       timestamptz NOT NULL DEFAULT now()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS documents_tenant_created_idx
          ON documents (tenant_id, created_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS documents_tenant_status_idx
          ON documents (tenant_id, status);
        """,
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id       text PRIMARY KEY,
          tenant_id      text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
          document_id    uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
          document_title text NOT NULL,
          page_number    integer NOT NULL,
          chunk_index    integer NOT NULL,
          text           text NOT NULL,
          embedding      vector({dimensions}) NOT NULL,
          token_count    integer NOT NULL DEFAULT 0,
          start_char     integer NOT NULL DEFAULT 0,
          end_char       integer NOT NULL DEFAULT 0,
          tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS chunks_tenant_document_idx
          ON chunks (tenant_id, document_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS chunks_tsv_idx
          ON chunks USING gin (tsv);
        """,
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
          ON chunks USING hnsw (embedding vector_cosine_ops);
        """,
        """
        CREATE TABLE IF NOT EXISTS chunk_entities (
          chunk_id  text NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
          tenant_id text NOT NULL,
          entity    text NOT NULL,
          PRIMARY KEY (chunk_id, entity)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS chunk_entities_tenant_entity_idx
          ON chunk_entities (tenant_id, lower(entity));
        """,
        """
        CREATE TABLE IF NOT EXISTS benchmark_reports (
          run_id     text PRIMARY KEY,
          tenant_id  text NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          payload    jsonb NOT NULL
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS benchmark_reports_tenant_created_idx
          ON benchmark_reports (tenant_id, created_at DESC);
        """,
    ]
