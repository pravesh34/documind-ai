"""LangGraph orchestration for grounded, self-correcting Graph RAG.

Flow::

    analyze → retrieve → grade ─┬─ rewrite → retrieve (bounded loop)
                                ├─ generate → verify ─┬─ revise → verify
                                │                     └─ finalize
                                └─ abstain → finalize

Every node appends to an accumulated trace so the API can expose how the answer
was produced.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai.llm import LLMService
from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import parse_json_object
from app.domain import (
    ChunkRecord,
    Citation,
    GroundedAnswer,
    RetrievalResult,
)
from app.rag.citations import (
    VerificationResult,
    apply_retrieval_scores,
    build_extractive_answer,
    prune_unsupported,
    select_citations,
    verify_answer,
)
from app.rag.grading import GradeResult, RelevanceGrader
from app.rag.prompts import (
    answer_messages,
    revision_messages,
    verification_messages,
)
from app.rag.query import QueryAnalysis, QueryAnalyzer
from app.rag.retrieval import HybridRetriever, RetrievalRequest
from app.repositories.base import DocumentRepository

logger = get_logger("rag.workflow")

ABSTENTION_ANSWER = (
    "I cannot answer this from the ingested corpus because the retrieved passages "
    "do not contain sufficient evidence."
)


class RAGState(TypedDict, total=False):
    tenant_id: str
    question: str
    top_k: int
    similarity_threshold: float
    target_document_id: str | None
    analysis: QueryAnalysis | None
    queries: list[str]
    retrieval: RetrievalResult | None
    grade: GradeResult | None
    chunks: list[ChunkRecord]
    answer: str
    citations: list[Citation]
    verification: VerificationResult | None
    grounding_score: float
    abstained: bool
    retrieval_attempts: int
    revision_count: int
    max_retrieval_attempts: int
    max_revision_attempts: int
    provider: str | None
    model: str | None
    retrieval_latency_ms: int
    started_at: float
    trace: Annotated[list[dict[str, Any]], operator.add]


class RAGWorkflow:
    def __init__(
        self,
        repository: DocumentRepository,
        retriever: HybridRetriever,
        llm: LLMService,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.analyzer = QueryAnalyzer(settings, llm)
        self.grader = RelevanceGrader(settings, llm)
        self.graph = self._build_graph()

    def _default_provider(self) -> str:
        """Provider label used when no generative call produced the answer."""
        if not self.llm.provider_chain:
            return "extractive"
        return self.llm.provider_chain[0].split(":")[0]

    # Graph construction --------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(RAGState)

        builder.add_node("analyze", self._analyze)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("generate", self._generate)
        builder.add_node("verify", self._verify)
        builder.add_node("revise", self._revise)
        builder.add_node("abstain", self._abstain)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "analyze")
        builder.add_edge("analyze", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            _route_after_grade,
            {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", "verify")
        builder.add_conditional_edges(
            "verify",
            _route_after_verify,
            {"revise": "revise", "finalize": "finalize"},
        )
        builder.add_edge("revise", "verify")
        builder.add_edge("abstain", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile()

    # Entrypoint ----------------------------------------------------------
    async def run(
        self,
        tenant_id: str,
        question: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        target_document_id: str | None = None,
    ) -> GroundedAnswer:
        started_at = time.perf_counter()
        initial_state: RAGState = {
            "tenant_id": tenant_id,
            "question": question,
            "top_k": top_k or self.settings.retrieval_top_k,
            "similarity_threshold": (
                similarity_threshold
                if similarity_threshold is not None
                else self.settings.similarity_threshold
            ),
            "target_document_id": target_document_id,
            "analysis": None,
            "queries": [],
            "retrieval": None,
            "grade": None,
            "chunks": [],
            "answer": "",
            "citations": [],
            "verification": None,
            "grounding_score": 0.0,
            "abstained": False,
            "retrieval_attempts": 1,
            "revision_count": 0,
            "max_retrieval_attempts": self.settings.max_retrieval_attempts,
            "max_revision_attempts": self.settings.max_revision_attempts,
            "provider": None,
            "model": None,
            "retrieval_latency_ms": 0,
            "started_at": started_at,
            "trace": [],
        }

        try:
            final: RAGState = await self.graph.ainvoke(initial_state)
        except Exception as error:  # noqa: BLE001 - never 500 a chat request
            logger.exception("rag workflow failed", error=str(error))
            return GroundedAnswer(
                answer=(
                    "The retrieval pipeline failed while answering this question. "
                    "Please retry; no speculative answer was generated."
                ),
                citations=[],
                abstained=True,
                grounding_score=0.0,
                latency_ms=int((time.perf_counter() - started_at) * 1_000),
                retrieval_latency_ms=0,
                trace=[{"node": "error", "detail": type(error).__name__}],
            )

        analysis = final.get("analysis")
        retrieval = final.get("retrieval")
        return GroundedAnswer(
            answer=final.get("answer") or ABSTENTION_ANSWER,
            citations=final.get("citations") or [],
            abstained=bool(final.get("abstained")),
            grounding_score=round(float(final.get("grounding_score") or 0.0), 4),
            latency_ms=int((time.perf_counter() - started_at) * 1_000),
            retrieval_latency_ms=int(final.get("retrieval_latency_ms") or 0),
            provider=final.get("provider"),
            model=final.get("model"),
            trace=final.get("trace") or [],
            queries=analysis.sub_queries if analysis else final.get("queries") or [],
            revision_count=int(final.get("revision_count") or 0),
            retrieval_attempts=int(final.get("retrieval_attempts") or 0),
            retrieved_chunks=retrieval.citations if retrieval else [],
        )

    # Nodes ---------------------------------------------------------------
    async def _analyze(self, state: RAGState) -> dict[str, Any]:
        analysis = await self.analyzer.analyze(state["question"])
        return {
            "analysis": analysis,
            "queries": analysis.sub_queries,
            "trace": [
                {
                    "node": "analyze",
                    "intent": analysis.intent,
                    "keywords": analysis.keywords,
                    "queries": analysis.sub_queries,
                }
            ],
        }

    async def _retrieve(self, state: RAGState) -> dict[str, Any]:
        retrieval = await self.retriever.retrieve(
            state["tenant_id"],
            RetrievalRequest(
                query=state["question"],
                queries=state.get("queries") or [],
                top_k=state.get("top_k") or self.settings.retrieval_top_k,
                similarity_threshold=state.get("similarity_threshold")
                or self.settings.similarity_threshold,
                target_document_id=state.get("target_document_id"),
                candidates=self.settings.retrieval_candidates,
            ),
        )
        chunks = [item.chunk for item in retrieval.scored_chunks]
        return {
            "retrieval": retrieval,
            "chunks": chunks,
            "retrieval_latency_ms": retrieval.latency_ms,
            "trace": [
                {
                    "node": "retrieve",
                    "attempt": state.get("retrieval_attempts", 1),
                    "candidates": retrieval.total_candidates,
                    "selected": len(chunks),
                    "top_score": round(
                        retrieval.scored_chunks[0].score if retrieval.scored_chunks else 0.0,
                        4,
                    ),
                    "latency_ms": retrieval.latency_ms,
                }
            ],
        }

    async def _grade(self, state: RAGState) -> dict[str, Any]:
        retrieval = state.get("retrieval")
        if retrieval is None:
            retrieval = RetrievalResult(
                query=state["question"],
                queries=state.get("queries") or [],
                scored_chunks=[],
                citations=[],
                total_candidates=0,
                latency_ms=0,
            )

        attempts_remaining = max(
            0,
            self.settings.max_retrieval_attempts - state.get("retrieval_attempts", 1),
        )
        grade = await self.grader.grade(
            state["question"], retrieval, attempts_remaining=attempts_remaining
        )

        if grade.route == "generate":
            judge = await self.grader.llm_judge(state["question"], retrieval)
            if judge is not None and judge < 0.2 and grade.coverage < 0.45:
                grade = GradeResult(
                    relevant=False,
                    score=grade.score,
                    best_similarity=grade.best_similarity,
                    best_lexical=grade.best_lexical,
                    coverage=grade.coverage,
                    route="abstain",
                    reason="LLM judge found the evidence insufficient",
                )

        return {
            "grade": grade,
            "trace": [
                {
                    "node": "grade",
                    "relevant": grade.relevant,
                    "route": grade.route,
                    "coverage": round(grade.coverage, 4),
                    "score": round(grade.score, 4),
                    "reason": grade.reason,
                }
            ],
        }

    async def _rewrite(self, state: RAGState) -> dict[str, Any]:
        grade = state.get("grade")
        existing = state.get("queries") or [state["question"]]
        rewritten = await self.analyzer.rewrite(
            state["question"],
            existing,
            grade.score if grade else 0.0,
        )
        merged = [*existing, *rewritten][:5]
        attempts = state.get("retrieval_attempts", 1) + 1
        return {
            "queries": merged,
            "retrieval_attempts": attempts,
            "trace": [
                {
                    "node": "rewrite",
                    "attempt": attempts,
                    "queries": rewritten or merged,
                }
            ],
        }

    async def _generate(self, state: RAGState) -> dict[str, Any]:
        chunks = state.get("chunks") or []
        question = state["question"]
        provider: str | None = None
        model: str | None = None

        answer: str | None = None
        if self.llm.available and chunks:
            result = await self.llm.complete(
                answer_messages(question, chunks),
                temperature=0.0,
                max_tokens=900,
                node="generate",
            )
            if result is not None and result.text.strip():
                answer = result.text.strip()
                provider, model = result.provider, result.model

        if not answer:
            answer = build_extractive_answer(question, chunks)

        citations = apply_retrieval_scores(
            select_citations(chunks, answer),
            state.get("retrieval").scored_chunks if state.get("retrieval") else [],
        )

        return {
            "answer": answer,
            "citations": citations,
            "provider": provider or self._default_provider(),
            "model": model or ("deterministic-extractive" if not self.llm.available else None),
            "trace": [
                {
                    "node": "generate",
                    "mode": "llm" if provider else "extractive",
                    "provider": provider,
                    "citations": len(citations),
                }
            ],
        }

    async def _verify(self, state: RAGState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        chunks = state.get("chunks") or []
        verification = verify_answer(answer, chunks)

        if (
            self.settings.llm_grading_enabled
            and self.llm.available
            and chunks
            and verification.unsupported_sentences
        ):
            judge = await self._llm_verification(answer, chunks)
            if judge is not None:
                verification = VerificationResult(
                    grounding_score=min(verification.grounding_score, judge),
                    supported_sentences=verification.supported_sentences,
                    unsupported_sentences=verification.unsupported_sentences,
                    sentence_scores=verification.sentence_scores,
                )

        return {
            "verification": verification,
            "grounding_score": verification.grounding_score,
            "trace": [
                {
                    "node": "verify",
                    "grounding": round(verification.grounding_score, 4),
                    "unsupported_claims": len(verification.unsupported_sentences),
                }
            ],
        }

    async def _revise(self, state: RAGState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        chunks = state.get("chunks") or []
        revisions = state.get("revision_count", 0) + 1
        revised = answer

        if self.llm.available and chunks:
            result = await self.llm.complete(
                revision_messages(state["question"], chunks, answer),
                temperature=0.0,
                max_tokens=900,
                node="revise",
            )
            if result is not None and result.text.strip():
                revised = result.text.strip()

        if revised == answer:
            verification = state.get("verification")
            if verification is not None:
                revised = prune_unsupported(answer, verification)

        return {
            "answer": revised,
            "revision_count": revisions,
            "trace": [
                {
                    "node": "revise",
                    "revision": revisions,
                    "mode": "llm" if revised != answer and self.llm.available else "deterministic",
                }
            ],
        }

    async def _abstain(self, state: RAGState) -> dict[str, Any]:
        grade = state.get("grade")
        return {
            "answer": ABSTENTION_ANSWER,
            "citations": [],
            "abstained": True,
            "grounding_score": 0.0,
            "trace": [
                {
                    "node": "abstain",
                    "reason": grade.reason if grade else "insufficient evidence",
                }
            ],
        }

    async def _finalize(self, state: RAGState) -> dict[str, Any]:
        return {
            "trace": [
                {
                    "node": "finalize",
                    "abstained": bool(state.get("abstained")),
                    "grounding": round(float(state.get("grounding_score") or 0.0), 4),
                    "latency_ms": int(
                        (time.perf_counter() - state.get("started_at", time.perf_counter()))
                        * 1_000
                    ),
                }
            ]
        }

    async def _llm_verification(
        self, answer: str, chunks: list[ChunkRecord]
    ) -> float | None:
        result = await self.llm.complete(
            verification_messages(answer, chunks),
            temperature=0.0,
            node="verify",
        )
        if result is None:
            return None
        payload = parse_json_object(result.text)
        if not payload:
            return None
        try:
            return max(0.0, min(1.0, float(payload.get("grounding_score", 0.0))))
        except (TypeError, ValueError):
            return None


def _route_after_grade(state: RAGState) -> str:
    grade = state.get("grade")
    if grade is None:
        return "abstain"
    if grade.route == "generate":
        return "generate"
    if grade.route == "rewrite":
        return "rewrite"
    return "abstain"


def _route_after_verify(state: RAGState) -> str:
    verification = state.get("verification")
    if state.get("abstained") or verification is None:
        return "finalize"
    if not verification.unsupported_sentences:
        return "finalize"
    if state.get("revision_count", 0) >= int(state.get("max_revision_attempts", 1)):
        return "finalize"
    return "revise"
