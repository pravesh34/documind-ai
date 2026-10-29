"""Camel-case serialisation helpers shared by repositories and HTTP routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.domain import (
    BenchmarkResult,
    BenchmarkRunReport,
    ChunkRecord,
    Citation,
    DocumentPage,
    DocumentRecord,
    DocumentSummary,
)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def page_to_dict(page: DocumentPage) -> dict[str, Any]:
    return {
        "pageNumber": page.page_number,
        "text": page.text,
        "charCount": page.char_count,
    }


def chunk_to_dict(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "chunkId": chunk.chunk_id,
        "documentId": chunk.document_id,
        "documentTitle": chunk.document_title,
        "pageNumber": chunk.page_number,
        "chunkIndex": chunk.chunk_index,
        "text": chunk.text,
        "tokenCount": chunk.token_count,
        "startChar": chunk.start_char,
        "endChar": chunk.end_char,
        "entities": list(chunk.entities),
    }


def summary_to_dict(summary: DocumentSummary) -> dict[str, Any]:
    return {
        "id": summary.id,
        "title": summary.title,
        "filename": summary.filename,
        "fileSize": summary.file_size,
        "pageCount": summary.page_count,
        "chunkCount": summary.chunk_count,
        "status": summary.status,
        "embeddingStatus": summary.embedding_status,
        "uploadedAt": _iso(summary.uploaded_at),
        "summary": summary.summary,
        "errorMessage": summary.error_message,
    }


def document_to_dict(document: DocumentRecord, *, include_chunks: bool = True) -> dict[str, Any]:
    payload = {
        "id": document.id,
        "title": document.title,
        "filename": document.filename,
        "fileSize": document.file_size,
        "mimeType": document.mime_type,
        "checksum": document.checksum,
        "pageCount": document.page_count,
        "chunkCount": len(document.chunks),
        "status": document.status,
        "embeddingStatus": document.embedding_status,
        "uploadedAt": _iso(document.uploaded_at),
        "summary": document.summary,
        "errorMessage": document.error_message,
        "pages": [page_to_dict(page) for page in document.pages],
    }
    if include_chunks:
        payload["chunks"] = [chunk_to_dict(chunk) for chunk in document.chunks]
    return payload


def citation_to_dict(citation: Citation) -> dict[str, Any]:
    return {
        "documentId": citation.document_id,
        "documentTitle": citation.document_title,
        "pageNumber": citation.page_number,
        "chunkId": citation.chunk_id,
        "excerpt": citation.excerpt,
        "similarityScore": round(citation.similarity_score, 4),
        "lexicalScore": round(citation.lexical_score, 4),
        "combinedScore": round(citation.combined_score, 4),
        "source": citation.source,
    }


def citation_from_dict(payload: dict[str, Any]) -> Citation:
    return Citation(
        document_id=str(payload.get("documentId", "")),
        document_title=str(payload.get("documentTitle", "")),
        page_number=int(payload.get("pageNumber", 1)),
        chunk_id=str(payload.get("chunkId", "")),
        excerpt=str(payload.get("excerpt", "")),
        similarity_score=float(payload.get("similarityScore", 0.0)),
        lexical_score=float(payload.get("lexicalScore", 0.0)),
        combined_score=float(payload.get("combinedScore", 0.0)),
        source=str(payload.get("source", "dense")),  # type: ignore[arg-type]
    )


def benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "testCaseId": result.test_case_id,
        "question": result.question,
        "documentTitle": result.document_title,
        "isNegativeTest": result.is_negative_test,
        "retrievedChunks": [citation_to_dict(citation) for citation in result.retrieved_chunks],
        "generatedAnswer": result.generated_answer,
        "recallHit": result.recall_hit,
        "faithfulnessScore": round(result.faithfulness_score, 2),
        "hallucinationDetected": result.hallucination_detected,
        "abstentionCorrect": result.abstention_correct,
        "latencyMs": result.latency_ms,
        "passed": result.passed,
    }


def benchmark_result_from_dict(payload: dict[str, Any]) -> BenchmarkResult:
    return BenchmarkResult(
        id=str(payload.get("id", "")),
        test_case_id=str(payload.get("testCaseId", "")),
        question=str(payload.get("question", "")),
        document_title=str(payload.get("documentTitle", "")),
        is_negative_test=bool(payload.get("isNegativeTest", False)),
        retrieved_chunks=[
            citation_from_dict(item)
            for item in payload.get("retrievedChunks", [])
            if isinstance(item, dict)
        ],
        generated_answer=str(payload.get("generatedAnswer", "")),
        recall_hit=bool(payload.get("recallHit", False)),
        faithfulness_score=float(payload.get("faithfulnessScore", 0.0)),
        hallucination_detected=bool(payload.get("hallucinationDetected", False)),
        abstention_correct=bool(payload.get("abstentionCorrect", False)),
        latency_ms=int(payload.get("latencyMs", 0)),
        passed=bool(payload.get("passed", False)),
    )


def benchmark_to_dict(report: BenchmarkRunReport) -> dict[str, Any]:
    return {
        "runId": report.run_id,
        "timestamp": _iso(report.timestamp),
        "totalTests": report.total_tests,
        "passedCount": report.passed_count,
        "overallRecallAtK": round(report.overall_recall_at_k, 2),
        "averageFaithfulness": round(report.average_faithfulness, 2),
        "hallucinationRate": round(report.hallucination_rate, 2),
        "abstentionAccuracy": round(report.abstention_accuracy, 2),
        "averageLatencyMs": report.average_latency_ms,
        "results": [benchmark_result_to_dict(item) for item in report.results],
        "tenantId": report.tenant_id,
    }


def benchmark_from_dict(payload: dict[str, Any]) -> BenchmarkRunReport:
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif isinstance(timestamp, datetime):
        parsed = timestamp
    else:
        parsed = datetime.now(UTC)

    return BenchmarkRunReport(
        run_id=str(payload.get("runId", "")),
        timestamp=parsed,
        total_tests=int(payload.get("totalTests", 0)),
        passed_count=int(payload.get("passedCount", 0)),
        overall_recall_at_k=float(payload.get("overallRecallAtK", 0.0)),
        average_faithfulness=float(payload.get("averageFaithfulness", 0.0)),
        hallucination_rate=float(payload.get("hallucinationRate", 0.0)),
        abstention_accuracy=float(payload.get("abstentionAccuracy", 0.0)),
        average_latency_ms=int(payload.get("averageLatencyMs", 0)),
        results=[
            benchmark_result_from_dict(item)
            for item in payload.get("results", [])
            if isinstance(item, dict)
        ],
        tenant_id=str(payload.get("tenantId", "default")),
    )
