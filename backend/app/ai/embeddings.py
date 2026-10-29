"""Free/local embedding providers with a deterministic offline fallback."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections import OrderedDict
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import fit_dimensions, tcp_endpoint_open, tokenize

logger = get_logger("ai.embeddings")


class DeterministicEmbeddings(Embeddings):
    """Hashed bag-of-words embeddings; free forever and dependency-free."""

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = tokenize(text)
        for index, word in enumerate(words):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[position] += sign * (1.0 / (1.0 + (index % 5)))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class FallbackEmbeddings(Embeddings):
    """Try free providers in order, then fall back to deterministic vectors."""

    def __init__(
        self,
        providers: Sequence[tuple[str, Embeddings]],
        dimensions: int,
    ) -> None:
        self.providers = list(providers)
        self.dimensions = dimensions
        self.deterministic = DeterministicEmbeddings(dimensions)
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_limit = 1_024

    def _cache_get(self, text: str) -> list[float] | None:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
        return cached

    def _cache_put(self, text: str, vector: list[float]) -> None:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._query_cache[key] = vector
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._cache_limit:
            self._query_cache.popitem(last=False)

    @property
    def remote_enabled(self) -> bool:
        return bool(self.providers)

    @property
    def provider_chain(self) -> list[str]:
        return [name for name, _ in self.providers]

    def _fit(self, vector: Sequence[float]) -> list[float]:
        return fit_dimensions(vector, self.dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for name, provider in self.providers:
            try:
                vectors = provider.embed_documents(texts)
                if len(vectors) == len(texts):
                    return [self._fit(vector) for vector in vectors]
                logger.warning("embedding provider returned wrong batch size", provider=name)
            except Exception as error:  # noqa: BLE001 - fall through to next provider
                logger.warning(
                    "embedding provider failed",
                    provider=name,
                    error=str(error),
                )
        return self.deterministic.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        for name, provider in self.providers:
            try:
                vector = self._fit(provider.embed_query(text))
                self._cache_put(text, vector)
                return vector
            except Exception as error:  # noqa: BLE001
                logger.warning("query embedding failed", provider=name, error=str(error))
        fallback = self.deterministic.embed_query(text)
        self._cache_put(text, fallback)
        return fallback

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        for name, provider in self.providers:
            try:
                vectors = await provider.aembed_documents(texts)
                if len(vectors) == len(texts):
                    return [self._fit(vector) for vector in vectors]
                logger.warning("embedding provider returned wrong batch size", provider=name)
            except Exception as error:  # noqa: BLE001
                logger.warning("embedding provider failed", provider=name, error=str(error))
        return await self.deterministic.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        for name, provider in self.providers:
            try:
                vector = self._fit(await provider.aembed_query(text))
                self._cache_put(text, vector)
                return vector
            except Exception as error:  # noqa: BLE001
                logger.warning("query embedding failed", provider=name, error=str(error))
        fallback = await self.deterministic.aembed_query(text)
        self._cache_put(text, fallback)
        return fallback


def _ollama(settings: Settings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def _gemini(settings: Settings) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    kwargs = {
        "model": settings.gemini_embedding_model,
        "google_api_key": settings.gemini_api_key,
    }
    try:
        return GoogleGenerativeAIEmbeddings(
            **kwargs,
            output_dimensionality=settings.embedding_dimensions,
        )
    except TypeError:
        # Older integration versions do not expose output_dimensionality.
        return GoogleGenerativeAIEmbeddings(**kwargs)


def _huggingface(settings: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model=settings.hf_embedding_model,
        task="feature-extraction",
        huggingfacehub_api_token=settings.hf_api_token,
    )


def build_embeddings(settings: Settings) -> FallbackEmbeddings:
    providers: list[tuple[str, Embeddings]] = []

    def add(name: str, factory) -> None:
        try:
            providers.append((name, factory(settings)))
        except Exception as error:  # noqa: BLE001 - optional dependency/config
            logger.warning("embedding provider unavailable", provider=name, error=str(error))

    if settings.embedding_provider == "deterministic":
        pass
    elif settings.embedding_provider == "ollama":
        add("ollama", _ollama)
    elif settings.embedding_provider == "gemini":
        add("gemini", _gemini)
    elif settings.embedding_provider == "huggingface":
        add("huggingface", _huggingface)
    else:  # auto — free hosted providers first, then local Ollama
        if settings.gemini_api_key:
            add("gemini", _gemini)
        if settings.hf_api_token:
            add("huggingface", _huggingface)
        if tcp_endpoint_open(settings.ollama_base_url):
            add("ollama", _ollama)

    embeddings = FallbackEmbeddings(providers, settings.embedding_dimensions)
    logger.info(
        "embedding providers ready",
        chains=embeddings.provider_chain or ["deterministic"],
        dimensions=settings.embedding_dimensions,
    )
    return embeddings


async def embed_in_batches(
    embeddings: FallbackEmbeddings,
    texts: list[str],
    batch_size: int,
    concurrency: int,
) -> list[list[float]]:
    """Batch embeddings with bounded concurrency (free-tier friendly)."""
    if not texts:
        return []

    batches = [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await embeddings.aembed_documents(batch)

    results = await asyncio.gather(*(run(batch) for batch in batches))
    return [vector for batch in results for vector in batch]
