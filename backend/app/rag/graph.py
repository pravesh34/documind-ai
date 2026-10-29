"""Entity co-occurrence graph used for Graph RAG expansion.

Chunks are nodes; an edge exists when two chunks share a salient entity. Edge
weight is the IDF-weighted overlap, so rare entities (``SOC 2``, ``pgvector``)
create stronger links than ubiquitous ones.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from app.domain import ChunkRecord


@dataclass(slots=True)
class GraphNeighbor:
    chunk: ChunkRecord
    score: float
    shared_entities: list[str]


class EntityGraph:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks: dict[str, ChunkRecord] = {chunk.chunk_id: chunk for chunk in chunks}
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)

        for chunk in chunks:
            for entity in chunk.entities:
                key = entity.strip().lower()
                if key:
                    self.entity_chunks[key].add(chunk.chunk_id)

        total = max(1, len(self.chunks))
        self.idf: dict[str, float] = {
            entity: math.log(1.0 + total / (1.0 + len(chunk_ids)))
            for entity, chunk_ids in self.entity_chunks.items()
        }

    def entities_of(self, chunk_ids: list[str]) -> list[str]:
        counts: dict[str, int] = defaultdict(int)
        display: dict[str, str] = {}
        for chunk_id in chunk_ids:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            for entity in chunk.entities:
                key = entity.strip().lower()
                if key:
                    counts[key] += 1
                    display.setdefault(key, entity.strip())
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [display[key] for key, _ in ranked]

    def neighbors(
        self,
        seed_chunk_ids: list[str],
        *,
        exclude: set[str] | None = None,
        limit: int = 8,
    ) -> list[GraphNeighbor]:
        excluded = set(exclude or ()) | set(seed_chunk_ids)
        seed_entities = {
            entity.lower()
            for chunk_id in seed_chunk_ids
            for entity in (self.chunks.get(chunk_id).entities if self.chunks.get(chunk_id) else [])
            if entity.strip()
        }
        if not seed_entities:
            return []

        scores: dict[str, float] = defaultdict(float)
        shared: dict[str, list[str]] = defaultdict(list)

        for entity in seed_entities:
            chunk_ids = self.entity_chunks.get(entity)
            if not chunk_ids:
                continue
            weight = self.idf.get(entity, 1.0) / (1.0 + math.log(1.0 + len(chunk_ids)))
            for chunk_id in chunk_ids:
                if chunk_id in excluded:
                    continue
                scores[chunk_id] += weight
                shared[chunk_id].append(entity)

        if not scores:
            return []

        maximum = max(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda item: -item[1])[:limit]
        neighbors: list[GraphNeighbor] = []
        for chunk_id, score in ranked:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            neighbors.append(
                GraphNeighbor(
                    chunk=chunk,
                    score=min(1.0, score / maximum),
                    shared_entities=shared[chunk_id],
                )
            )
        return neighbors
