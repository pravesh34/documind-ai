"""Free/local chat-model chains built with LangChain.

Provider order is configurable, but every chain is free-tier or local. When no
provider is reachable the RAG pipeline degrades to deterministic extractive
answers instead of failing the request.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import message_text, parse_json_object, provider_metadata, tcp_endpoint_open

logger = get_logger("ai.llm")


@dataclass(slots=True)
class ChatResult:
    text: str
    provider: str | None = None
    model: str | None = None


@dataclass(slots=True)
class ChatChain:
    runnable: Runnable | None
    providers: list[tuple[str, str]]


def _ollama(settings: Settings) -> tuple[str, BaseChatModel]:
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=0.0,
        num_predict=1_024,
        client_kwargs={"timeout": settings.ai_timeout_seconds},
    )
    return f"ollama:{settings.ollama_chat_model}", model


def _gemini(settings: Settings, model_name: str) -> tuple[str, BaseChatModel]:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_output_tokens=1_024,
    )
    return f"gemini:{model_name}", model


def _groq(settings: Settings) -> tuple[str, BaseChatModel]:
    from langchain_groq import ChatGroq

    model = ChatGroq(
        model=settings.groq_chat_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=1_024,
        timeout=settings.ai_timeout_seconds,
        max_retries=0,
    )
    return f"groq:{settings.groq_chat_model}", model


def _openrouter(settings: Settings) -> tuple[str, BaseChatModel]:
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.openrouter_chat_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0.0,
        max_tokens=1_024,
        timeout=settings.ai_timeout_seconds,
        max_retries=0,
        default_headers={
            "HTTP-Referer": "https://github.com/documind-ai",
            "X-Title": "DocuMind AI",
        },
    )
    return f"openrouter:{settings.openrouter_chat_model}", model


def build_chat_chain(settings: Settings) -> ChatChain:
    """Build an ordered chat chain with retries and fallbacks."""
    candidates: list[tuple[str, BaseChatModel]] = []

    def add(factory, *args: Any) -> None:
        try:
            candidates.append(factory(settings, *args))
        except Exception as error:  # noqa: BLE001 - optional provider/config
            logger.warning("chat provider unavailable", error=str(error))

    mode = settings.llm_provider

    if mode == "none":
        logger.info("LLM_PROVIDER=none; using extractive generation")
        return ChatChain(runnable=None, providers=[])

    if mode in {"auto", "gemini"} and settings.gemini_api_key:
        add(_gemini, settings.gemini_chat_model)
        if settings.gemini_fallback_chat_model != settings.gemini_chat_model:
            add(_gemini, settings.gemini_fallback_chat_model)
    if mode in {"auto", "groq"} and settings.groq_api_key:
        add(_groq)
    if mode in {"auto", "openrouter"} and settings.openrouter_api_key:
        add(_openrouter)
    if mode in {"auto", "ollama"}:
        if mode == "ollama" or tcp_endpoint_open(settings.ollama_base_url):
            add(_ollama)

    if not candidates:
        logger.info("no chat providers configured; using extractive generation")
        return ChatChain(runnable=None, providers=[])

    labels = [label for label, _ in candidates]
    models = [model for _, model in candidates]

    chain: Runnable = models[0].with_retry(
        stop_after_attempt=max(1, settings.ai_max_retries + 1),
        wait_exponential_jitter=True,
    )
    if len(models) > 1:
        chain = chain.with_fallbacks(
            [
                model.with_retry(
                    stop_after_attempt=max(1, settings.ai_max_retries + 1),
                    wait_exponential_jitter=True,
                )
                for model in models[1:]
            ],
            exceptions_to_handle=(Exception,),
        )

    providers = [
        (label.split(":", 1)[0], label.split(":", 1)[1]) for label in labels
    ]
    logger.info("chat providers ready", chain=labels)
    return ChatChain(runnable=chain, providers=providers)


class LLMService:
    """Thin async facade over the LangChain chat chain."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chain = build_chat_chain(settings)

    @property
    def available(self) -> bool:
        return self.chain.runnable is not None

    @property
    def provider_chain(self) -> list[str]:
        return [f"{provider}:{model}" for provider, model in self.chain.providers]

    @property
    def timeout_seconds(self) -> float:
        # Accounts for provider fallbacks; each attempt is bounded by the
        # provider client's own timeout.
        return self.settings.ai_timeout_seconds * max(1, len(self.chain.providers))

    async def complete(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        node: str = "chat",
    ) -> ChatResult | None:
        if self.chain.runnable is None:
            return None

        config: dict[str, Any] = {
            "run_name": f"documind.{node}",
            "metadata": {"node": node},
        }

        try:
            response = await asyncio.wait_for(
                self.chain.runnable.ainvoke(messages, config=config),
                timeout=self.timeout_seconds,
            )
        except Exception as error:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "chat completion failed; falling back to extractive mode",
                node=node,
                error=type(error).__name__,
                detail=str(error)[:300],
            )
            return None

        text = message_text(response)
        if not text:
            return None

        provider, model = provider_metadata(response)
        if provider is None:
            provider = getattr(response, "response_metadata", {}).get("model_provider")
        return ChatResult(text=text, provider=provider, model=model)

    async def complete_json(
        self,
        messages: list[BaseMessage],
        *,
        node: str = "json",
    ) -> dict[str, Any] | None:
        result = await self.complete(messages, temperature=0.0, node=node)
        if result is None:
            return None
        return parse_json_object(result.text)
