"""Versioned REST API."""

from __future__ import annotations

import asyncio
import base64
import binascii
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import Response
from slowapi import Limiter

from app.api.ai_overrides import (
    AiOverrides,
    apply_overrides,
    parse_ai_overrides,
)
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvalRunRequest,
    RetrievalRequestModel,
    RetrievalResponse,
    UploadRequest,
)
from app.config import Settings
from app.container import Container, build_request_runtime
from app.core.errors import AppError
from app.core.security import AuthContext, require_auth
from app.core.serialization import (
    benchmark_to_dict,
    citation_to_dict,
    document_to_dict,
    summary_to_dict,
)
from app.core.utils import content_tokens
from app.domain import BenchmarkTestCase


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise AppError(503, "NOT_READY", "The service is still starting up.")
    return container


def _validate_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as error:
        raise AppError(422, "INVALID_DOCUMENT_ID", "Document id must be a UUID.") from error


def resolve_runtime(
    container: Container, x_ai_config: str | None
) -> tuple[Container, AiOverrides | None]:
    """Container to serve the request with (per-request AI overrides optional)."""
    overrides = parse_ai_overrides(x_ai_config)
    if overrides is None:
        return container, None
    runtime = build_request_runtime(container, apply_overrides(container.settings, overrides))
    return runtime, overrides


def _decode_base64(value: str) -> bytes:
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise AppError(422, "INVALID_BASE64", "fileBase64 is not valid base64.") from error


def _test_case_to_dict(test: BenchmarkTestCase) -> dict:
    return {
        "id": test.id,
        "documentTitle": test.document_title,
        "question": test.question,
        "expectedPassage": test.expected_passage,
        "expectedPage": test.expected_page,
        "isNegativeTest": test.is_negative_test,
        "description": test.description,
    }


def build_router(settings: Settings, limiter: Limiter) -> APIRouter:
    router = APIRouter()

    # Documents ----------------------------------------------------------
    @router.get("/documents")
    async def list_documents(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        q: str | None = Query(default=None, max_length=200),
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        summaries, total = await container.documents.list_documents(
            auth.tenant_id, limit=limit, offset=offset, query=q
        )
        return {
            "documents": [summary_to_dict(item) for item in summaries],
            "total": total,
        }

    @router.post("/documents/upload", status_code=201)
    async def upload_document(
        request: Request,
        payload: UploadRequest,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        data: bytes | None = None

        if payload.file_base64:
            data = _decode_base64(payload.file_base64)
            if len(data) > settings.max_upload_bytes:
                raise AppError(
                    413,
                    "PAYLOAD_TOO_LARGE",
                    f"File exceeds the {settings.max_upload_bytes} byte limit.",
                )

        document = await container.ingestion.ingest(
            auth.tenant_id,
            title=payload.title,
            filename=payload.filename or f"{payload.title}.txt",
            text=None if data is not None else payload.text_content,
            data=data,
            mime_type=payload.mime_type,
            file_size=payload.file_size,
        )
        return {"document": document_to_dict(document)}

    @router.post("/documents/upload-file", status_code=201)
    async def upload_file(
        request: Request,
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise AppError(
                413,
                "PAYLOAD_TOO_LARGE",
                f"File exceeds the {settings.max_upload_bytes} byte limit.",
            )

        fallback_title = Path(file.filename or "document.txt").stem
        document = await container.ingestion.ingest(
            auth.tenant_id,
            title=(title or fallback_title).strip() or fallback_title,
            filename=file.filename or "document.txt",
            data=data,
            mime_type=file.content_type,
        )
        return {"document": document_to_dict(document)}

    @router.get("/documents/{document_id}")
    async def get_document(
        request: Request,
        document_id: str,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        document = await container.documents.get_document(
            auth.tenant_id, _validate_uuid(document_id)
        )
        return {"document": document_to_dict(document)}

    @router.delete("/documents/{document_id}", status_code=204)
    async def delete_document(
        request: Request,
        document_id: str,
        auth: AuthContext = Depends(require_auth),
    ) -> Response:
        container = get_container(request)
        await container.documents.delete_document(
            auth.tenant_id, _validate_uuid(document_id)
        )
        return Response(status_code=204)

    @router.post("/documents/reset-defaults")
    async def reset_defaults(
        request: Request,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        seeded = await container.documents.reset_defaults(auth.tenant_id)
        summaries, total = await container.documents.list_documents(
            auth.tenant_id, limit=200, offset=0
        )
        return {
            "seeded": seeded,
            "documents": [summary_to_dict(item) for item in summaries],
            "total": total,
        }

    # Chat ---------------------------------------------------------------
    @router.post("/chat", response_model=ChatResponse)
    @limiter.limit(settings.rate_limit_chat)
    async def chat(
        request: Request,
        response: Response,
        payload: ChatRequest,
        auth: AuthContext = Depends(require_auth),
        x_ai_config: str | None = Header(default=None),
    ) -> dict:
        container, overrides = resolve_runtime(get_container(request), x_ai_config)
        question = payload.query

        # Light coreference handling for short follow-up questions.
        if payload.history and len(content_tokens(question)) <= 4:
            previous = [turn.content for turn in payload.history if turn.role == "user"]
            if previous:
                question = f"{previous[-1]} — {question}"

        answer = await container.chat.answer(
            auth.tenant_id,
            question,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
            target_document_id=payload.target_document_id,
        )
        return {
            "answer": answer.answer,
            "citations": [citation_to_dict(item) for item in answer.citations],
            "abstained": answer.abstained,
            "groundingScore": round(answer.grounding_score * 100, 2),
            "latencyMs": answer.latency_ms,
            "retrievalLatencyMs": answer.retrieval_latency_ms,
            "queries": answer.queries,
            "trace": answer.trace,
            "provider": answer.provider,
            "model": answer.model,
            "revisionCount": answer.revision_count,
            "retrievalAttempts": answer.retrieval_attempts,
            "retrievedChunks": [
                citation_to_dict(item) for item in answer.retrieved_chunks
            ],
        }

    @router.post("/retrieval", response_model=RetrievalResponse)
    @limiter.limit(settings.rate_limit_default)
    async def retrieval(
        request: Request,
        response: Response,
        payload: RetrievalRequestModel,
        auth: AuthContext = Depends(require_auth),
        x_ai_config: str | None = Header(default=None),
    ) -> dict:
        container, _overrides = resolve_runtime(get_container(request), x_ai_config)
        result = await container.chat.retrieve(
            auth.tenant_id,
            payload.query,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
            target_document_id=payload.target_document_id,
        )
        return {
            "query": result.query,
            "queries": result.queries,
            "citations": [citation_to_dict(item) for item in result.citations],
            "totalCandidates": result.total_candidates,
            "latencyMs": result.latency_ms,
        }

    # Evaluation ---------------------------------------------------------
    @router.get("/eval/dataset")
    async def eval_dataset(
        request: Request,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        return {
            "testCases": [
                _test_case_to_dict(item) for item in container.evaluation.dataset()
            ]
        }

    @router.get("/eval/latest")
    async def eval_latest(
        request: Request,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        report = await container.evaluation.latest(auth.tenant_id)
        return {"report": benchmark_to_dict(report) if report else None}

    @router.get("/eval/history")
    async def eval_history(
        request: Request,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        reports = await container.evaluation.history(auth.tenant_id, limit=10)
        return {"reports": [benchmark_to_dict(item) for item in reports]}

    @router.post("/eval/run")
    @limiter.limit(settings.rate_limit_default)
    async def eval_run(
        request: Request,
        response: Response,
        payload: EvalRunRequest | None = None,
        auth: AuthContext = Depends(require_auth),
        x_ai_config: str | None = Header(default=None),
    ) -> dict:
        container, _overrides = resolve_runtime(get_container(request), x_ai_config)
        tests: list[BenchmarkTestCase] | None = None

        custom = payload.custom_tests if payload is not None else None
        if custom:
            tests = [
                BenchmarkTestCase(
                    id=item.id or f"tc-custom-{index + 1}",
                    document_title=item.document_title,
                    question=item.question,
                    expected_passage=item.expected_passage or "NONE",
                    expected_page=item.expected_page,
                    is_negative_test=item.is_negative_test,
                    description=item.description or "Custom evaluation test",
                )
                for index, item in enumerate(custom)
            ]

        report = await container.evaluation.run(auth.tenant_id, tests)
        return {"report": benchmark_to_dict(report)}

    # Export -------------------------------------------------------------
    @router.get("/export/files")
    async def export_files(
        request: Request,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        files = await asyncio.to_thread(container.exports.list_files)
        return {"files": [item.to_dict() for item in files]}

    @router.get("/export/download")
    async def export_download(
        request: Request,
        auth: AuthContext = Depends(require_auth),
    ) -> Response:
        container = get_container(request)
        payload = await asyncio.to_thread(container.exports.zip_bytes)
        return Response(
            content=payload,
            media_type="application/zip",
            headers={
                "content-disposition": 'attachment; filename="documind-ai-backend.zip"',
                "content-length": str(len(payload)),
            },
        )

    # Meta ---------------------------------------------------------------
    @router.get("/info")
    async def info(
        request: Request,
        auth: AuthContext = Depends(require_auth),
    ) -> dict:
        container = get_container(request)
        return {
            "version": settings.app_version,
            "environment": settings.environment,
            "backend": "postgres" if container.database else "memory",
            "chatProviders": container.llm.provider_chain or ["extractive"],
            "embeddingProviders": container.embeddings.provider_chain
            or ["deterministic"],
            "embeddingDimensions": settings.embedding_dimensions,
            "graphExpansion": settings.graph_expansion_enabled,
            "maxRetrievalAttempts": settings.max_retrieval_attempts,
            "rateLimitChat": settings.rate_limit_chat,
        }

    return router
