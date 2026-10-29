"""Hybrid dense + sparse + graph retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

from app.ai.embeddings import FallbackEmbeddings
from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import content_tokens, cosine_similarity, unique_ordered
from app.domain import (
    ChunkRecord,
    Citation,
    RetrievalResult,
    ScoredChunk,
)
from app.rag.bm25 import BM25Index
from app.rag.graph import EntityGraph
from app.repositories.base import DocumentRepository

logger = get_logger("rag.retrieval")


@dataclass(slots=True)
class RetrievalRequest:
    query: str
    queries: list[str] = field(default_factory=list)
    top_k: int = 6
    similarity_threshold: float = 0.28
    target_document_id: str | None = None
    candidates: int = 24


@dataclass(slots=True)
class _Signals:
    similarity: dict[str, float] = field(default_factory=dict)
    lexical: dict[str, float] = field(default_factory=dict)
    rrf: dict[str, float] = field(default_factory=dict)
    graph: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        repository: DocumentRepository,
        embeddings: FallbackEmbeddings,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self.settings = settings

    async def retrieve(self, tenant_id: str, request: RetrievalRequest) -> RetrievalResult:
        started = time.perf_counter()
        queries = unique_ordered([request.query, *request.queries])[:4]

        query_embeddings = await asyncio.gather(
            *(self.embeddings.aembed_query(question) for question in queries),
            return_exceptions=True,
        )
        usable: list[tuple[str, list[float]]] = []
        for question, vector in zip(queries, query_embeddings, strict=False):
            if isinstance(vector, Exception):
                logger.warning("query embedding failed", error=str(vector))
                continue
            usable.append((question, vector))

        pool = await self._prefetch(tenant_id, request, usable)
        if not pool:
            return RetrievalResult(
                query=request.query,
                queries=queries,
                scored_chunks=[],
                citations=[],
                total_candidates=0,
                latency_ms=int((time.perf_counter() - started) * 1_000),
            )

        signals = await self._score_pool(tenant_id, request, usable, pool)
        scored = self._fuse(pool, signals, request)

        return RetrievalResult(
            query=request.query,
            queries=queries,
            scored_chunks=scored,
            citations=[_to_citation(item) for item in scored],
            total_candidates=len(pool),
            latency_ms=int((time.perf_counter() - started) * 1_000),
            candidate_terms={
                term
                for chunk in pool
                for term in content_tokens(chunk.text, min_length=3)
            },
        )

    async def _prefetch(
        self,
        tenant_id: str,
        request: RetrievalRequest,
        queries: list[tuple[str, list[float]]],
    ) -> list[ChunkRecord]:
        batches = await asyncio.gather(
            *(
                self.repository.list_chunks(
                    tenant_id,
                    request.target_document_id,
                    query_text=question,
                    query_embedding=vector,
                    limit=max(request.candidates, request.top_k * 4),
                )
                for question, vector in queries
            ),
            return_exceptions=True,
        )

        pool: dict[str, ChunkRecord] = {}
        for batch in batches:
            if isinstance(batch, Exception):
                logger.warning("candidate prefetch failed", error=str(batch))
                continue
            for chunk in batch:
                pool.setdefault(chunk.chunk_id, chunk)

        if not pool:
            fallback = await self.repository.list_chunks(
                tenant_id,
                request.target_document_id,
                limit=request.candidates,
            )
            pool = {chunk.chunk_id: chunk for chunk in fallback}

        return list(pool.values())

    async def _score_pool(
        self,
        tenant_id: str,
        request: RetrievalRequest,
        queries: list[tuple[str, list[float]]],
        pool: list[ChunkRecord],
    ) -> _Signals:
        signals = _Signals()
        bm25 = BM25Index({chunk.chunk_id: chunk.text for chunk in pool})
        k = max(1, self.settings.rrf_k)

        for _, vector in queries:
            for chunk in pool:
                similarity = max(
                    0.0,
                    min(
                        1.0,
                        cosine_similarity(vector, chunk.embedding)
                        if chunk.embedding
                        else 0.0,
                    ),
                )
                if similarity > signals.similarity.get(chunk.chunk_id, 0.0):
                    signals.similarity[chunk.chunk_id] = similarity

        for question, _ in queries:
            lexical = bm25.score(question)
            maximum = max(lexical.values(), default=0.0)
            if maximum <= 0:
                continue
            for chunk_id, value in lexical.items():
                normalised = value / maximum
                if normalised > signals.lexical.get(chunk_id, 0.0):
                    signals.lexical[chunk_id] = normalised

        # Reciprocal Rank Fusion across every query variant.
        max_rrf = 0.0
        for question, vector in queries:
            dense_ranking = sorted(
                pool,
                key=lambda chunk: cosine_similarity(vector, chunk.embedding)
                if chunk.embedding
                else 0.0,
                reverse=True,
            )
            lexical_scores = bm25.score(question)
            lexical_ranking = sorted(
                pool,
                key=lambda chunk: lexical_scores.get(chunk.chunk_id, 0.0),
                reverse=True,
            )

            for ranking in (dense_ranking, lexical_ranking):
                for rank, chunk in enumerate(ranking, start=1):
                    contribution = 1.0 / (k + rank)
                    signals.rrf[chunk.chunk_id] = (
                        signals.rrf.get(chunk.chunk_id, 0.0) + contribution
                    )
                    max_rrf = max(max_rrf, signals.rrf[chunk.chunk_id])

        if max_rrf > 0:
            signals.rrf = {
                chunk_id: value / max_rrf for chunk_id, value in signals.rrf.items()
            }

        if self.settings.graph_expansion_enabled and pool:
            await self._expand_graph(tenant_id, request, signals, pool, bm25)

        return signals

    async def _expand_graph(
        self,
        tenant_id: str,
        request: RetrievalRequest,
        signals: _Signals,
        pool: list[ChunkRecord],
        bm25: BM25Index,
    ) -> None:
        preliminary = {
            chunk.chunk_id: (
                0.6 * signals.similarity.get(chunk.chunk_id, 0.0)
                + 0.4 * signals.lexical.get(chunk.chunk_id, 0.0)
            )
            for chunk in pool
        }
        seeds = sorted(pool, key=lambda chunk: preliminary[chunk.chunk_id], reverse=True)[
            : self.settings.graph_seed_chunks
        ]
        if not seeds:
            return

        graph = EntityGraph(pool)
        seed_entities = graph.entities_of([chunk.chunk_id for chunk in seeds])[
            : self.settings.graph_neighbor_entities
        ]
        if not seed_entities:
            return

        neighbors = await self.repository.chunks_by_entities(
            tenant_id,
            seed_entities,
            exclude_chunk_ids={chunk.chunk_id for chunk in pool},
            document_id=request.target_document_id,
            limit=self.settings.graph_neighbor_chunks,
        )
        if not neighbors:
            return

        combined = [*pool, *neighbors]
        expanded = EntityGraph(combined)
        ranked = expanded.neighbors(
            [chunk.chunk_id for chunk in seeds],
            exclude={chunk.chunk_id for chunk in seeds},
            limit=len(neighbors) + self.settings.graph_seed_chunks,
        )
        for neighbor in ranked:
            signals.graph[neighbor.chunk.chunk_id] = neighbor.score
            if all(chunk.chunk_id != neighbor.chunk.chunk_id for chunk in pool):
                pool.append(neighbor.chunk)
                signals.similarity.setdefault(
                    neighbor.chunk.chunk_id, _seed_similarity(neighbor.chunk, seeds)
                )
                signals.lexical.setdefault(neighbor.chunk.chunk_id, 0.0)

        # Graph-only chunks need a BM25-consistent baseline so fusion is not
        # penalised purely for lacking a dense score.
        for chunk in pool:
            if chunk.chunk_id not in signals.rrf:
                signals.rrf[chunk.chunk_id] = 0.0

    def _fuse(
        self,
        pool: list[ChunkRecord],
        signals: _Signals,
        request: RetrievalRequest,
    ) -> list[ScoredChunk]:
        scored: list[ScoredChunk] = []

        for chunk in pool:
            similarity = signals.similarity.get(chunk.chunk_id, 0.0)
            lexical = signals.lexical.get(chunk.chunk_id, 0.0)
            rrf = signals.rrf.get(chunk.chunk_id, 0.0)
            graph_score = signals.graph.get(chunk.chunk_id, 0.0)

            if (
                similarity < request.similarity_threshold
                and lexical < 0.08
                and graph_score < 0.10
            ):
                continue

            combined = (
                0.45 * similarity
                + 0.25 * lexical
                + 0.20 * rrf
                + 0.10 * graph_score
            )

            if graph_score >= max(similarity, lexical):
                source = "graph"
            elif lexical > similarity:
                source = "lexical"
            else:
                source = "dense"

            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=combined,
                    similarity=similarity,
                    lexical=lexical,
                    source=source,  # type: ignore[arg-type]
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, request.top_k)]


def query_coverage(query: str, chunks: list[ChunkRecord], *, top_n: int = 3) -> float:
    """Fraction of query content terms found in the best chunks."""
    query_terms = set(content_tokens(query, min_length=2))
    if not query_terms:
        return 0.0

    available: set[str] = set()
    for chunk in chunks[:top_n]:
        available.update(content_tokens(chunk.text, min_length=2))

    return len(query_terms & available) / len(query_terms)


def _seed_similarity(chunk: ChunkRecord, seeds: list[ChunkRecord]) -> float:
    return max(
        (
            cosine_similarity(chunk.embedding, seed.embedding)
            if chunk.embedding and seed.embedding
            else 0.0
        )
        for seed in seeds
    )


def _to_citation(item: ScoredChunk) -> Citation:
    return Citation(
        document_id=item.chunk.document_id,
        document_title=item.chunk.document_title,
        page_number=item.chunk.page_number,
        chunk_id=item.chunk.chunk_id,
        excerpt=item.chunk.text,
        similarity_score=round(item.similarity, 4),
        lexical_score=round(item.lexical, 4),
        combined_score=round(item.score, 4),
        source=item.source,
    )


def rrf_normalise(value: float, k: int) -> float:
    maximum = 2.0 / (k + 1)
    return min(1.0, value / maximum) if maximum > 0 and math.isfinite(value) else 0.0
