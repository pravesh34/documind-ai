"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any

import pytest_asyncio
from app.config import Settings
from app.container import Container


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "test",
        "database_url": None,
        "auth_mode": "none",
        "seed_default_documents": True,
        "rate_limit_enabled": False,
        "metrics_enabled": False,
        "enable_docs": True,
        "llm_provider": "none",
        "embedding_provider": "deterministic",
        "background_enrichment": False,
        "entity_extraction_mode": "heuristic",
    }
    base.update(overrides)
    return Settings(**base)


@pytest_asyncio.fixture
async def container():
    instance = await Container.create(make_settings())
    await instance.startup()
    try:
        yield instance
    finally:
        await instance.shutdown()
