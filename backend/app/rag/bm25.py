"""Small BM25 implementation used for the sparse half of hybrid retrieval."""

from __future__ import annotations

import math
from collections import Counter

from app.core.utils import content_tokens


class BM25Index:
    """Okapi BM25 over an in-memory pool of chunks."""

    def __init__(
        self,
        documents: dict[str, str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        self.document_frequencies: Counter[str] = Counter()

        for chunk_id, text in documents.items():
            tokens = content_tokens(text, min_length=2)
            frequencies = Counter(tokens)
            self.term_frequencies[chunk_id] = frequencies
            self.lengths[chunk_id] = max(1, len(tokens))
            for token in frequencies:
                self.document_frequencies[token] += 1

        self.total_documents = max(1, len(documents))
        total_length = sum(self.lengths.values()) or 1
        self.average_length = total_length / self.total_documents

    def score(self, query: str) -> dict[str, float]:
        tokens = [token for token in content_tokens(query, min_length=2)]
        if not tokens:
            return {}

        unique_tokens = list(dict.fromkeys(tokens))
        scores: dict[str, float] = {}

        for chunk_id, frequencies in self.term_frequencies.items():
            length = self.lengths[chunk_id]
            score = 0.0

            for token in unique_tokens:
                frequency = frequencies.get(token, 0)
                if frequency == 0:
                    continue

                document_frequency = self.document_frequencies[token]
                idf = math.log(
                    1.0
                    + (self.total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * (length / self.average_length)
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scores[chunk_id] = score

        return scores

    def top(self, query: str, limit: int) -> list[tuple[str, float]]:
        ranked = sorted(self.score(query).items(), key=lambda item: -item[1])
        return ranked[:limit]
