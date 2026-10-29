"""Citation selection and claim-level grounding verification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.utils import content_tokens, truncate
from app.domain import ChunkRecord, Citation, ScoredChunk

MARKER_RE = re.compile(r"\[(\d{1,2})\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")

ABSTENTION_MARKERS = (
    "cannot answer",
    "insufficient evidence",
    "not contain sufficient",
    "no evidence",
    "unable to answer",
)


def _raw_sentences(text: str) -> list[str]:
    """Split while preserving ``[n]`` markers (needed for claim scoping)."""
    parts = [part.strip(" -•\t") for part in SENTENCE_RE.split(text)]
    return [part for part in parts if len(part) >= 24]


def split_sentences(text: str) -> list[str]:
    return [
        cleaned
        for part in _raw_sentences(text)
        if (cleaned := MARKER_RE.sub("", part).strip())
    ]


def citation_markers(text: str) -> list[int]:
    return [int(match) for match in MARKER_RE.findall(text)]


@dataclass(slots=True)
class VerificationResult:
    grounding_score: float
    supported_sentences: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    sentence_scores: list[float] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return bool(self.unsupported_sentences)

    @property
    def faithfulness_percent(self) -> float:
        return round(self.grounding_score * 100, 2)


def select_citations(
    chunks: list[ChunkRecord],
    answer: str,
    *,
    limit: int = 6,
) -> list[Citation]:
    markers = [marker for marker in citation_markers(answer) if 1 <= marker <= len(chunks)]

    if markers:
        selected: list[ChunkRecord] = []
        seen: set[str] = set()
        for marker in markers:
            chunk = chunks[marker - 1]
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                selected.append(chunk)
    else:
        selected = _rank_by_overlap(chunks, answer)[:limit]

    return [
        Citation(
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            excerpt=truncate(chunk.text, 900),
            similarity_score=0.0,
            lexical_score=0.0,
            combined_score=0.0,
            source="dense",
        )
        for chunk in selected[:limit]
    ]


def apply_retrieval_scores(citations: list[Citation], scored: list[ScoredChunk]) -> list[Citation]:
    lookup = {item.chunk.chunk_id: item for item in scored}
    enriched: list[Citation] = []
    for citation in citations:
        item = lookup.get(citation.chunk_id)
        if item is None:
            enriched.append(citation)
            continue
        enriched.append(
            Citation(
                document_id=citation.document_id,
                document_title=citation.document_title,
                page_number=citation.page_number,
                chunk_id=citation.chunk_id,
                excerpt=citation.excerpt,
                similarity_score=round(item.similarity, 4),
                lexical_score=round(item.lexical, 4),
                combined_score=round(item.score, 4),
                source=item.source,
            )
        )
    return enriched


def verify_answer(
    answer: str,
    chunks: list[ChunkRecord],
    *,
    support_threshold: float = 0.34,
) -> VerificationResult:
    sentences = _raw_sentences(answer)
    if not sentences:
        if not answer.strip():
            return VerificationResult(grounding_score=0.0)
        # Short answer: fall back to whole-answer coverage.
        coverage = _best_coverage(answer, chunks)
        return VerificationResult(grounding_score=coverage)

    supported: list[str] = []
    unsupported: list[str] = []
    scores: list[float] = []

    for raw in sentences:
        sentence = MARKER_RE.sub("", raw).strip()
        lowered = sentence.lower()
        if any(marker in lowered for marker in ABSTENTION_MARKERS):
            supported.append(sentence)
            scores.append(1.0)
            continue

        markers = citation_markers(raw)
        cited = [
            chunks[marker - 1]
            for marker in markers
            if 1 <= marker <= len(chunks)
        ]
        candidates = cited or chunks
        coverage = _best_coverage(sentence, candidates)
        scores.append(coverage)

        if coverage >= support_threshold:
            supported.append(sentence)
        elif coverage >= support_threshold * 0.55:
            # Partial support counts proportionally but is not a hallucination.
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    grounding = sum(scores) / len(scores) if scores else 0.0
    return VerificationResult(
        grounding_score=max(0.0, min(1.0, grounding)),
        supported_sentences=supported,
        unsupported_sentences=unsupported,
        sentence_scores=scores,
    )


def prune_unsupported(answer: str, verification: VerificationResult) -> str:
    """Deterministically remove unsupported sentences from an answer."""
    if not verification.unsupported_sentences:
        return answer

    kept: list[str] = []
    for raw in re.split(r"(?<=[.!?])\s+|\n", answer):
        sentence = raw.strip()
        if not sentence:
            continue
        if any(
            unsupported[:40].lower() in sentence.lower()
            for unsupported in verification.unsupported_sentences
        ):
            continue
        kept.append(sentence)

    pruned = " ".join(kept).strip()
    return pruned or answer


def _best_coverage(sentence: str, chunks: list[ChunkRecord]) -> float:
    terms = set(content_tokens(sentence, min_length=3))
    if not terms:
        return 0.0

    best = 0.0
    for chunk in chunks:
        chunk_terms = set(content_tokens(chunk.text, min_length=3))
        if not chunk_terms:
            continue
        best = max(best, len(terms & chunk_terms) / len(terms))
    return best


def _rank_by_overlap(chunks: list[ChunkRecord], answer: str) -> list[ChunkRecord]:
    answer_terms = set(content_tokens(answer, min_length=3))
    if not answer_terms:
        return chunks

    ranked = sorted(
        chunks,
        key=lambda chunk: len(
            answer_terms & set(content_tokens(chunk.text, min_length=3))
        ),
        reverse=True,
    )
    return ranked


def build_extractive_answer(
    question: str,
    chunks: list[ChunkRecord],
    *,
    max_bullets: int = 4,
) -> str:
    """Deterministic, zero-LLM answer composed only of source sentences."""
    if not chunks:
        return (
            "I cannot answer this from the ingested corpus because no relevant "
            "evidence was retrieved."
        )

    question_terms = set(content_tokens(question, min_length=3))
    candidates: list[tuple[float, int, str]] = []

    for index, chunk in enumerate(chunks[:6], start=1):
        for sentence in _sentences(chunk.text):
            terms = set(content_tokens(sentence, min_length=3))
            if not terms:
                continue
            overlap = len(question_terms & terms) / max(1, len(question_terms))
            score = overlap + 1.0 / (1.0 + len(sentence) / 200.0) + (0.05 if index == 1 else 0)
            candidates.append((score, index, sentence))

    if not candidates:
        return (
            "I cannot answer this from the ingested corpus because the retrieved "
            "passages do not contain sufficient evidence."
        )

    seen: set[str] = set()
    bullets: list[str] = []
    for _, index, sentence in sorted(candidates, key=lambda item: -item[0]):
        key = sentence.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {truncate(sentence, 320)} [{index}]")
        if len(bullets) >= max_bullets:
            break

    return (
        "The retrieved passages support the following:\n\n"
        + "\n".join(bullets)
        + "\n\n_Answer composed extractively from source passages "
        "(no generative model configured)._"
    )


def _sentences(text: str) -> list[str]:
    parts = [part.strip(" -•\t") for part in SENTENCE_RE.split(text)]
    return [part for part in parts if len(part) >= 30]
