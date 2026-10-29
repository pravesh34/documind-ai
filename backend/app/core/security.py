"""API-key authentication with tenant resolution."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Request

from app.config import Settings, get_settings
from app.core.errors import AppError


@dataclass(frozen=True, slots=True)
class AuthContext:
    tenant_id: str


def parse_api_keys(raw: str) -> dict[str, str]:
    """Parse ``key:tenant,key2:tenant2`` into a mapping."""
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        key, separator, tenant = trimmed.partition(":")
        key = key.strip()
        if not key:
            continue
        keys[key] = (tenant.strip() or "default") if separator else "default"
    return keys


def _bearer_token(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _settings_for(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


async def require_auth(request: Request) -> AuthContext:
    settings = _settings_for(request)

    if settings.resolved_auth_mode == "none":
        context = AuthContext(tenant_id="default")
        request.state.tenant_id = context.tenant_id
        return context

    provided = request.headers.get("x-api-key") or _bearer_token(
        request.headers.get("authorization")
    )
    if not provided:
        raise AppError(
            401,
            "UNAUTHORIZED",
            "Missing API key. Send x-api-key or an Authorization: Bearer token.",
        )

    for key, tenant in parse_api_keys(settings.api_keys).items():
        if secrets.compare_digest(key, provided):
            context = AuthContext(tenant_id=tenant)
            request.state.tenant_id = context.tenant_id
            return context

    raise AppError(401, "UNAUTHORIZED", "Invalid API key.")
