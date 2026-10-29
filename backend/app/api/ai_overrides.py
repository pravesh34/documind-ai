"""Per-request AI provider overrides (sent by the frontend for BYO keys).

Keys arrive in the ``x-ai-config`` JSON header, live only for the duration of
the request, and are never persisted server-side. Requests without the header
use the statically configured provider chain from environment settings.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import Settings
from app.core.errors import AppError

LLMProviders = Literal["auto", "ollama", "gemini", "groq", "openrouter", "none"]
EmbeddingProviders = Literal[
    "auto", "ollama", "gemini", "huggingface", "deterministic"
]

MAX_KEY_LENGTH = 400
MAX_MODEL_LENGTH = 200
MAX_HEADER_LENGTH = 4_096


class AiOverrides(BaseModel):
    """Values a client may override for a single request."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    llm_provider: LLMProviders | None = Field(default=None, alias="llmProvider")
    embedding_provider: EmbeddingProviders | None = Field(
        default=None, alias="embeddingProvider"
    )
    gemini_api_key: str | None = Field(default=None, alias="geminiApiKey")
    groq_api_key: str | None = Field(default=None, alias="groqApiKey")
    openrouter_api_key: str | None = Field(default=None, alias="openrouterApiKey")
    hf_api_token: str | None = Field(default=None, alias="hfApiToken")
    ollama_base_url: str | None = Field(default=None, alias="ollamaBaseUrl")

    gemini_chat_model: str | None = Field(default=None, alias="geminiChatModel")
    groq_chat_model: str | None = Field(default=None, alias="groqChatModel")
    openrouter_chat_model: str | None = Field(
        default=None, alias="openrouterChatModel"
    )
    hf_embedding_model: str | None = Field(default=None, alias="hfEmbeddingModel")
    ollama_chat_model: str | None = Field(default=None, alias="ollamaChatModel")
    ollama_embedding_model: str | None = Field(
        default=None, alias="ollamaEmbeddingModel"
    )

    @field_validator("*", mode="before")
    @classmethod
    def _clean(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if len(value) > MAX_KEY_LENGTH:
                raise ValueError("value too long")
        return value


def parse_ai_overrides(header_value: str | None) -> AiOverrides | None:
    """Parse the ``x-ai-config`` header; ``None`` means "no override"."""
    if not header_value:
        return None
    if len(header_value) > MAX_HEADER_LENGTH:
        raise AppError(
            422, "INVALID_AI_CONFIG", "x-ai-config header exceeds the size limit."
        )
    try:
        payload = json.loads(header_value)
    except json.JSONDecodeError:
        raise AppError(
            422, "INVALID_AI_CONFIG", "x-ai-config header is not valid JSON."
        ) from None
    if not isinstance(payload, dict):
        raise AppError(
            422, "INVALID_AI_CONFIG", "x-ai-config must be a JSON object."
        )
    try:
        overrides = AiOverrides.model_validate(payload)
    except ValueError as error:
        raise AppError(
            422, "INVALID_AI_CONFIG", f"Invalid x-ai-config payload: {error}"
        ) from error
    if not overrides.model_dump(exclude_none=True, by_alias=True):
        return None
    return overrides


def apply_overrides(settings: Settings, overrides: AiOverrides) -> Settings:
    """Return a clone of ``settings`` with the override values swapped in."""
    updates: dict[str, Any] = {
        key: value
        for key, value in overrides.model_dump().items()
        if value is not None
    }
    if not updates:
        return settings
    return settings.model_copy(update=updates)
