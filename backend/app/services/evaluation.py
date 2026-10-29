"""Benchmark execution: recall@k, faithfulness and abstention accuracy."""

from __future__ import annotations

import asyncio
import uuid

from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import content_tokens
from app.domain import (
    BenchmarkResult,
    BenchmarkRunReport,
    BenchmarkTestCase,
    Citation,
    utc_now,
)
from app.rag.workflow import RAGWorkflow
from app.repositories.base import DocumentRepository
from app.services.benchmark_data import default_dataset

logger = get_logger("services.evaluation")


class EvaluationService:
    def __init__(
        self,
        repository: DocumentRepository,
        workflow: RAGWorkflow,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.workflow = workflow
        self.settings = settings

    def dataset(self) -> list[BenchmarkTestCase]:
        return default_dataset()

    async def latest(self, tenant_id: str) -> BenchmarkRunReport | None:
        return await self.repository.latest_benchmark(tenant_id)

    async def history(self, tenant_id: str, limit: int = 10) -> list[BenchmarkRunReport]:
        return await self.repository.list_benchmarks(tenant_id, limit)

    async def run(
        self,
        tenant_id: str,
        custom_tests: list[BenchmarkTestCase] | None = None,
    ) -> BenchmarkRunReport:
        tests = (custom_tests or default_dataset())[: self.settings.eval_max_tests]
        title_to_id = await self._document_index(tenant_id)

        semaphore = asyncio.Semaphore(max(1, self.settings.eval_concurrency))
        started = utc_now()

        async def evaluate(test: BenchmarkTestCase) -> BenchmarkResult:
            async with semaphore:
                return await self._evaluate_one(
                    tenant_id, test, title_to_id.get(test.document_title.lower())
                )

        results = await asyncio.gather(*(evaluate(test) for test in tests))
        report = self._aggregate(tenant_id, results, started)
        await self.repository.save_benchmark(report)
        logger.info(
            "benchmark run completed",
            run_id=report.run_id,
            passed=report.passed_count,
            total=report.total_tests,
        )
        return report

    async def _document_index(self, tenant_id: str) -> dict[str, str]:
        summaries, _ = await self.repository.list_documents(tenant_id, limit=200, offset=0)
        return {summary.title.lower(): summary.id for summary in summaries}

    async def _evaluate_one(
        self,
        tenant_id: str,
        test: BenchmarkTestCase,
        document_id: str | None,
    ) -> BenchmarkResult:
        answer = await self.workflow.run(
            tenant_id,
            test.question,
            top_k=self.settings.retrieval_top_k,
            similarity_threshold=self.settings.similarity_threshold,
            target_document_id=document_id,
        )

        retrieved = answer.retrieved_chunks
        faithfulness = 0.0 if answer.abstained else answer.grounding_score * 100

        if test.is_negative_test:
            recall_hit = False
            abstention_correct = answer.abstained
            hallucination = (not answer.abstained) and answer.grounding_score < (
                self.settings.grounding_threshold
            )
            passed = abstention_correct and not hallucination
        else:
            recall_hit = _recall_hit(test, retrieved)
            abstention_correct = not answer.abstained
            hallucination = (not answer.abstained) and answer.grounding_score < (
                self.settings.grounding_threshold
            )
            passed = (
                recall_hit
                and abstention_correct
                and faithfulness >= self.settings.grounding_threshold * 100
                and not hallucination
            )

        return BenchmarkResult(
            id=f"res-{uuid.uuid4().hex[:12]}",
            test_case_id=test.id,
            question=test.question,
            document_title=test.document_title,
            is_negative_test=test.is_negative_test,
            retrieved_chunks=retrieved[: max(1, self.settings.retrieval_top_k)],
            generated_answer=answer.answer,
            recall_hit=recall_hit,
            faithfulness_score=round(faithfulness, 2),
            hallucination_detected=hallucination,
            abstention_correct=abstention_correct,
            latency_ms=answer.latency_ms,
            passed=passed,
        )

    def _aggregate(
        self,
        tenant_id: str,
        results: list[BenchmarkResult],
        timestamp,
    ) -> BenchmarkRunReport:
        total = len(results)
        positives = [item for item in results if not item.is_negative_test]
        negatives = [item for item in results if item.is_negative_test]

        recall = (
            sum(1 for item in positives if item.recall_hit) / len(positives) * 100
            if positives
            else 0.0
        )
        faithfulness = (
            sum(item.faithfulness_score for item in results) / total if total else 0.0
        )
        hallucination = (
            sum(1 for item in results if item.hallucination_detected) / total * 100
            if total
            else 0.0
        )
        abstention = (
            sum(1 for item in negatives if item.abstention_correct) / len(negatives) * 100
            if negatives
            else 100.0
        )
        latency = int(sum(item.latency_ms for item in results) / total) if total else 0

        return BenchmarkRunReport(
            run_id=f"run-{uuid.uuid4().hex[:16]}",
            timestamp=timestamp,
            total_tests=total,
            passed_count=sum(1 for item in results if item.passed),
            overall_recall_at_k=round(recall, 2),
            average_faithfulness=round(faithfulness, 2),
            hallucination_rate=round(hallucination, 2),
            abstention_accuracy=round(abstention, 2),
            average_latency_ms=latency,
            results=results,
            tenant_id=tenant_id,
        )


def _recall_hit(test: BenchmarkTestCase, retrieved: list[Citation]) -> bool:
    if not retrieved:
        return False

    expected_terms = set(content_tokens(test.expected_passage, min_length=2))
    if not expected_terms:
        return False

    available: set[str] = set()
    page_match = False
    for citation in retrieved:
        available.update(content_tokens(citation.excerpt, min_length=2))
        if test.expected_page and citation.page_number == test.expected_page:
            page_match = True

    coverage = len(expected_terms & available) / len(expected_terms)
    return coverage >= 0.6 and (page_match or test.expected_page == 0)
