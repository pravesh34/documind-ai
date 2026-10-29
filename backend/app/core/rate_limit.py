"""Shared rate limiter (Redis-backed when REDIS_URL is configured)."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import Settings


def create_limiter(settings: Settings) -> Limiter:
    kwargs: dict = {
        "key_func": get_remote_address,
        "enabled": settings.rate_limit_enabled,
        "headers_enabled": True,
        "default_limits": [settings.rate_limit_default] if settings.rate_limit_enabled else [],
    }
    if settings.redis_url:
        kwargs["storage_uri"] = settings.redis_url
    return Limiter(**kwargs)
