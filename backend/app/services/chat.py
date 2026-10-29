"""Chat and retrieval-debug services."""

from __future__ import annotations

from app.config import Settings
from app.core.errors import AppError
from app.domain import GroundedAnswer, RetrievalResult
from app.rag.retrieval import HybridRetriever, RetrievalRequest
from app.rag.workflow import RAGWorkflow


class ChatService:
    def __init__(
        self,
        workflow: RAGWorkflow,
        retriever: HybridRetriever,
        settings: Settings,
    ) -> None:
        self.workflow = workflow
        self.retriever = retriever
        self.settings = settings

    async def answer(
        self,
        tenant_id: str,
        question: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        target_document_id: str | None = None,
    ) -> GroundedAnswer:
        question = question.strip()
        if not question:
            raise AppError(422, "VALIDATION_ERROR", "Query must not be empty.")
        if len(question) > 4_000:
            raise AppError(
                422, "VALIDATION_ERROR", "Query must be shorter than 4000 characters."
            )

        resolved_top_k = top_k or self.settings.retrieval_top_k
        if not 1 <= resolved_top_k <= 12:
            raise AppError(422, "VALIDATION_ERROR", "topK must be between 1 and 12.")

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.settings.similarity_threshold
        )
        if not 0.0 <= threshold <= 1.0:
            raise AppError(
                422, "VALIDATION_ERROR", "similarityThreshold must be between 0 and 1."
            )

        return await self.workflow.run(
            tenant_id,
            question,
            top_k=resolved_top_k,
            similarity_threshold=threshold,
            target_document_id=target_document_id,
        )

    async def retrieve(
        self,
        tenant_id: str,
        question: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        target_document_id: str | None = None,
    ) -> RetrievalResult:
        if not question.strip():
            raise AppError(422, "VALIDATION_ERROR", "Query must not be empty.")

        return await self.retriever.retrieve(
            tenant_id,
            RetrievalRequest(
                query=question.strip(),
                queries=[],
                top_k=top_k or self.settings.retrieval_top_k,
                similarity_threshold=(
                    similarity_threshold
                    if similarity_threshold is not None
                    else self.settings.similarity_threshold
                ),
                target_document_id=target_document_id,
                candidates=self.settings.retrieval_candidates,
            ),
        )
