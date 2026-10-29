"""Typed application settings (Pydantic Settings).

All AI providers are free-tier or local. The app remains fully functional with
zero credentials by using deterministic embeddings and extractive answers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AuthMode = Literal["none", "api-key"]
Environment = Literal["development", "test", "production"]
LLMProvider = Literal["auto", "ollama", "gemini", "groq", "openrouter", "none"]
EmbeddingProvider = Literal["auto", "ollama", "gemini", "huggingface", "deterministic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application ---------------------------------------------------------
    app_name: str = "DocuMind AI"
    app_version: str = "1.0.0"
    environment: Environment = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    enable_docs: bool = True
    frontend_dist: str = "dist"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8000",
        ]
    )
    trusted_hosts: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    # Persistence ---------------------------------------------------------
    # When unset the service uses an in-memory repository (great for local dev,
    # demos and tests). Production requires DATABASE_URL.
    database_url: str | None = None
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_echo: bool = False
    auto_create_schema: bool = True
    seed_default_documents: bool = True

    # Redis / queue -------------------------------------------------------
    redis_url: str | None = None
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    background_enrichment: bool = True

    # Security ------------------------------------------------------------
    auth_mode: AuthMode | None = None
    api_keys: str = ""
    rate_limit_enabled: bool = True
    rate_limit_default: str = "300/minute"
    rate_limit_chat: str = "30/minute"
    max_upload_bytes: int = 25 * 1024 * 1024
    max_text_chars: int = 5_000_000
    metrics_enabled: bool = True

    # Free AI providers ---------------------------------------------------
    llm_provider: LLMProvider = "auto"
    embedding_provider: EmbeddingProvider = "auto"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"

    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_fallback_chat_model: str = "gemini-2.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    groq_api_key: str | None = None
    groq_chat_model: str = "openai/gpt-oss-120b"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_chat_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    hf_api_token: str | None = None
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"

    ai_timeout_seconds: float = 30.0
    ai_max_retries: int = 2
    embedding_dimensions: int = 768
    embedding_batch_size: int = 32
    embedding_concurrency: int = 4

    # RAG / LangGraph -----------------------------------------------------
    chunk_size: int = 1_200
    chunk_overlap: int = 200
    rrf_k: int = 60
    retrieval_top_k: int = 6
    retrieval_candidates: int = 24
    similarity_threshold: float = 0.28
    max_retrieval_attempts: int = 2
    max_revision_attempts: int = 1
    grounding_threshold: float = 0.55
    query_analysis_enabled: bool = True
    multi_query_enabled: bool = True
    llm_grading_enabled: bool = False
    graph_expansion_enabled: bool = True
    graph_seed_chunks: int = 6
    graph_neighbor_entities: int = 16
    graph_neighbor_chunks: int = 8
    entity_extraction_enabled: bool = True
    entity_extraction_mode: Literal["heuristic", "llm"] = "heuristic"
    entity_extraction_limit: int = 12
    max_history_turns: int = 12

    # Evaluation ----------------------------------------------------------
    eval_max_tests: int = 50
    eval_concurrency: int = 2

    # Export --------------------------------------------------------------
    export_include_tests: bool = True

    @field_validator("cors_origins", "trusted_hosts", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not 64 <= self.embedding_dimensions <= 2_000:
            raise ValueError("EMBEDDING_DIMENSIONS must be between 64 and 2000")
        if self.retrieval_top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K must be >= 1")

        if self.environment == "production":
            if not self.database_url:
                raise ValueError("DATABASE_URL is required in production")
            if self.resolved_auth_mode == "api-key" and not self.api_keys.strip():
                raise ValueError("API_KEYS is required when AUTH_MODE=api-key")
            if self.auto_create_schema:
                # Allowed, but surface the trade-off in logs.
                pass

        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        if self.embedding_provider == "huggingface" and not self.hf_api_token:
            raise ValueError("HF_API_TOKEN is required when EMBEDDING_PROVIDER=huggingface")

        return self

    @property
    def resolved_auth_mode(self) -> AuthMode:
        if self.auth_mode is not None:
            return self.auth_mode
        return "api-key" if self.environment == "production" else "none"

    @property
    def async_database_url(self) -> str | None:
        url = self.database_url
        if not url:
            return None
        if url.startswith("postgres://"):
            return "postgresql+asyncpg://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            return "postgresql+asyncpg://" + url[len("postgresql://") :]
        return url

    @property
    def uses_postgres(self) -> bool:
        return bool(self.database_url)

    @property
    def uses_celery(self) -> bool:
        return bool(self.celery_broker_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
