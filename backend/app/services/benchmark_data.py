"""Benchmark dataset for retrieval, faithfulness and abstention regression."""

from __future__ import annotations

from app.domain import BenchmarkTestCase

BENCHMARK_TESTS: tuple[BenchmarkTestCase, ...] = (
    BenchmarkTestCase(
        id="tc-001",
        document_title="Apex Holdings Q3 2026 Financial Report",
        question=(
            "What was the consolidated net revenue and adjusted EBITDA margin for Q3 2026?"
        ),
        expected_passage=(
            "net revenue of $482.6 million ... adjusted EBITDA margin ... 24.8%"
        ),
        expected_page=1,
        is_negative_test=False,
        description="Composite financial metric extraction across a single page",
    ),
    BenchmarkTestCase(
        id="tc-002",
        document_title="Apex Holdings Q3 2026 Financial Report",
        question=(
            "What were cash and cash equivalents and short-term investments at quarter end?"
        ),
        expected_passage="totalled $1.24 billion at quarter end",
        expected_page=1,
        is_negative_test=False,
        description="Exact liquidity figure retrieval",
    ),
    BenchmarkTestCase(
        id="tc-003",
        document_title="SOC 2 Security & Encryption Policy",
        question=(
            "What encryption algorithm and key management certification is mandated "
            "for data at rest?"
        ),
        expected_passage="AES-256-GCM ... FIPS 140-3 validated hardware security modules",
        expected_page=2,
        is_negative_test=False,
        description="Security control clause extraction",
    ),
    BenchmarkTestCase(
        id="tc-004",
        document_title="SOC 2 Security & Encryption Policy",
        question="What is the guaranteed triage timeline for a Severity 1 security incident?",
        expected_passage="triage timeline ... Severity 1 ... 15 minutes",
        expected_page=3,
        is_negative_test=False,
        description="SLA numeric threshold retrieval",
    ),
    BenchmarkTestCase(
        id="tc-005",
        document_title="DocuMind Hybrid Graph RAG Architecture",
        question="What formula and parameters are used for Reciprocal Rank Fusion?",
        expected_passage="RRF(d) = 1 / (k + rank_i(d)) ... k = 60",
        expected_page=2,
        is_negative_test=False,
        description="Formula and parameter grounding",
    ),
    BenchmarkTestCase(
        id="tc-006",
        document_title="DocuMind Hybrid Graph RAG Architecture",
        question="Which HNSW parameters are configured and what recall do they achieve?",
        expected_passage="m = 32, ef_construction = 200, ef_search = 64 ... 0.94 recall@10",
        expected_page=2,
        is_negative_test=False,
        description="Technical configuration extraction",
    ),
    BenchmarkTestCase(
        id="tc-007",
        document_title="Apex Holdings Q3 2026 Financial Report",
        question=(
            "What were the revenues generated from the commercial lunar space flight "
            "program in Q3 2026?"
        ),
        expected_passage="NONE",
        expected_page=0,
        is_negative_test=True,
        description="Out-of-domain negative test requiring clean abstention",
    ),
    BenchmarkTestCase(
        id="tc-008",
        document_title="SOC 2 Security & Encryption Policy",
        question="Which quantum encryption patent was granted to Apex Holdings in 2026?",
        expected_passage="NONE",
        expected_page=0,
        is_negative_test=True,
        description="Fabricated-premise negative test requiring clean abstention",
    ),
)


def default_dataset() -> list[BenchmarkTestCase]:
    return list(BENCHMARK_TESTS)
