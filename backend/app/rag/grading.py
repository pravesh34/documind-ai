"""Retrieval relevance grading with an optional LLM judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.ai.llm import LLMService
from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import content_tokens, parse_json_object
from app.domain import RetrievalResult
from app.rag.retrieval import query_coverage

logger = get_logger("rag.grading")

Route = Literal["generate", "rewrite", "abstain"]

JUDGE_PROMPT = """You are a retrieval relevance judge.

Question: {question}

Retrieved passages:
{passages}

Decide whether the passages contain enough evidence to answer the question.
Return JSON only: {{"relevant": true|false, "score": 0.0-1.0, "reason": "..."}}"""


@dataclass(slots=True)
class GradeResult:
    relevant: bool
    score: float
    best_similarity: float
    best_lexical: float
    coverage: float
    route: Route
    reason: str


class RelevanceGrader:
    def __init__(self, settings: Settings, llm: LLMService | None = None) -> None:
        self.settings = settings
        self.llm = llm

    async def grade(
        self,
        question: str,
        retrieval: RetrievalResult,
        *,
        attempts_remaining: int,
    ) -> GradeResult:
        chunks = [item.chunk for item in retrieval.scored_chunks]
        if not chunks:
            return GradeResult(
                relevant=False,
                score=0.0,
                best_similarity=0.0,
                best_lexical=0.0,
                coverage=0.0,
                route="rewrite" if attempts_remaining > 0 else "abstain",
                reason="no candidate chunks survived retrieval filtering",
            )

        best_similarity = max(item.similarity for item in retrieval.scored_chunks)
        best_lexical = max(item.lexical for item in retrieval.scored_chunks)
        coverage = query_coverage(question, chunks)

        candidate_terms = retrieval.candidate_terms or {
            term
            for chunk in chunks
            for term in content_tokens(chunk.text, min_length=3)
        }
        question_terms = set(content_tokens(question, min_length=3))
        unknown_terms = question_terms - candidate_terms
        unknown_ratio = len(unknown_terms) / max(1, len(question_terms))

        score = min(
            1.0,
            0.55 * coverage
            + 0.25 * min(1.0, best_similarity / max(0.01, self.settings.similarity_threshold))
            + 0.20 * best_lexical,
        )

        # A question that names entities the corpus never mentions cannot be
        # answered from it, even when generic terms (company name, year) match.
        if unknown_ratio >= 0.34 and coverage < 0.6:
            return GradeResult(
                relevant=False,
                score=score,
                best_similarity=best_similarity,
                best_lexical=best_lexical,
                coverage=coverage,
                route="rewrite" if attempts_remaining > 0 else "abstain",
                reason=(
                    "question references terms absent from the corpus: "
                    + ", ".join(sorted(unknown_terms)[:5])
                ),
            )

        if coverage >= 0.55 or (
            best_similarity >= self.settings.similarity_threshold and coverage >= 0.34
        ):
            return GradeResult(
                relevant=True,
                score=score,
                best_similarity=best_similarity,
                best_lexical=best_lexical,
                coverage=coverage,
                route="generate",
                reason="retrieved passages cover the question terms",
            )

        if coverage >= 0.30 and (best_lexical >= 0.25 or best_similarity >= 0.2):
            if attempts_remaining > 0:
                return GradeResult(
                    relevant=False,
                    score=score,
                    best_similarity=best_similarity,
                    best_lexical=best_lexical,
                    coverage=coverage,
                    route="rewrite",
                    reason="partial coverage; widening the query",
                )
            return GradeResult(
                relevant=True,
                score=score,
                best_similarity=best_similarity,
                best_lexical=best_lexical,
                coverage=coverage,
                route="generate",
                reason="partial coverage after query rewriting",
            )

        if attempts_remaining > 0 and coverage >= 0.15:
            return GradeResult(
                relevant=False,
                score=score,
                best_similarity=best_similarity,
                best_lexical=best_lexical,
                coverage=coverage,
                route="rewrite",
                reason="weak lexical overlap; retrying with rewritten queries",
            )

        return GradeResult(
            relevant=False,
            score=score,
            best_similarity=best_similarity,
            best_lexical=best_lexical,
            coverage=coverage,
            route="abstain",
            reason="the corpus has no evidence for the question",
        )

    async def llm_judge(self, question: str, retrieval: RetrievalResult) -> float | None:
        """Optional second opinion from the chat model (disabled by default)."""
        if self.llm is None or not self.llm.available or not self.settings.llm_grading_enabled:
            return None

        passages = "\n\n".join(
            f"[{index}] {item.chunk.text[:600]}"
            for index, item in enumerate(retrieval.scored_chunks, start=1)
        )
        from langchain_core.messages import HumanMessage, SystemMessage

        result = await self.llm.complete(
            [
                SystemMessage(content="You are a strict retrieval judge. Output JSON only."),
                HumanMessage(
                    content=JUDGE_PROMPT.format(
                        question=question, passages=passages or "(none)"
                    )
                ),
            ],
            temperature=0.0,
            node="relevance_grading",
        )
        if result is None:
            return None

        payload = parse_json_object(result.text)
        if not payload:
            return None
        try:
            return max(0.0, min(1.0, float(payload.get("score", 0.0))))
        except (TypeError, ValueError):
            return None
