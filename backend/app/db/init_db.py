"""Schema bootstrap under a PostgreSQL advisory lock.

Multiple API replicas can boot simultaneously; the lock serialises DDL so the
``CREATE ... IF NOT EXISTS`` statements cannot race.
"""

from __future__ import annotations

from sqlalchemy import text

from app.config import Settings
from app.core.logging import get_logger
from app.db.schema import SCHEMA_VERSION, extension_sql, schema_statements
from app.db.session import Database

logger = get_logger("db.init")

ADVISORY_LOCK_KEY = 72_727_271


async def initialize_database(database: Database, settings: Settings) -> None:
    async with database.session() as session:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": ADVISORY_LOCK_KEY},
        )
        await session.execute(text(extension_sql()))
        for statement in schema_statements(settings.embedding_dimensions):
            await session.execute(text(statement))
        await session.execute(
            text(
                """
                INSERT INTO tenants (id) VALUES (:tenant)
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"tenant": "default"},
        )

    logger.info(
        "database schema ready",
        version=SCHEMA_VERSION,
        dimensions=settings.embedding_dimensions,
    )
