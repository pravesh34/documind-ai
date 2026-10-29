"""Seed enterprise corpus used for demos, local development and benchmarks.

Each document is multi-page (``\\f`` separates pages) and contains the exact
facts referenced by the benchmark dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedDocument:
    title: str
    filename: str
    mime_type: str
    text: str


APEX_REPORT = """APEX HOLDINGS GROUP — CONSOLIDATED Q3 2026 FINANCIAL REPORT
Prepared by Corporate Finance · Fiscal quarter ended 30 September 2026

EXECUTIVE SUMMARY

Apex Holdings Group delivered consolidated net revenue of $482.6 million in Q3 2026, representing 18.4% year-over-year growth compared with $407.5 million in Q3 2025. Growth was driven by the Cloud Infrastructure segment and by expansion of multi-year enterprise contracts.

The adjusted EBITDA margin for the quarter was 24.8%, up 210 basis points from 22.7% in the prior-year period. Adjusted EBITDA was $119.7 million. Operating leverage improved as headcount growth decelerated to 4.1% while revenue grew 18.4%.

GAAP net income was $71.3 million, or $0.84 per diluted share. Cash and cash equivalents and short-term investments totalled $1.24 billion at quarter end. The company remains net-debt free and repurchased $40.0 million of shares during the quarter.

\fSEGMENT PERFORMANCE

Cloud Infrastructure contributed $279.9 million of net revenue (58% of total), up 27.1% year over year. The segment benefited from a 31% increase in average contract value for enterprise agreements and a net revenue retention rate of 118%.

Payments and Transaction Services contributed $130.3 million (27% of total), up 9.8% year over year. Transaction volume grew 14.2% while take rate compressed by 4 basis points due to mix shift toward lower-cost settlement rails.

Advisory and Data Products contributed $72.4 million (15% of total), up 6.2% year over year. Operating expenses were $214.9 million, of which research and development represented $86.2 million (17.9% of revenue).

Total headcount at quarter end was 3,412 employees, an increase of 4.1% year over year. Annualised revenue per employee improved to $565,900.

\fGUIDANCE, CAPITAL ALLOCATION AND RISK FACTORS

For Q4 2026, management guides consolidated net revenue to $505 million–$520 million, implying 16%–19% year-over-year growth. Full-year adjusted EBITDA margin is expected to be between 24.5% and 25.0%.

Capital allocation priorities remain: (1) organic research and development, (2) strategic acquisitions in data infrastructure, and (3) share repurchases under the existing $500 million authorisation, of which $310 million remains available.

Principal risks include enterprise IT spending cyclicality, foreign exchange volatility affecting approximately 22% of revenue, regulatory changes in cross-border payments, and concentration among the top 25 customers, which represented 31% of net revenue.

The audit committee completed its quarterly review on 14 October 2026. The interim financial statements are unaudited. A detailed reconciliation of adjusted EBITDA to GAAP net income is provided in Appendix A.
"""

SOC2_POLICY = """SOC 2 TYPE II SECURITY, ENCRYPTION AND INCIDENT RESPONSE POLICY
Apex Holdings Group · Information Security Programme · Version 7.2
Effective 1 July 2026 · Owner: Chief Information Security Officer

1. PURPOSE AND SCOPE

This policy defines the mandatory security controls for all production systems, customer data stores, and supporting infrastructure operated by Apex Holdings Group. It applies to every employee, contractor, and subprocessor with access to production environments or customer data.

The control set is mapped to the AICPAA Trust Services Criteria for Security, Availability, and Confidentiality. The most recent SOC 2 Type II examination covered the period 1 July 2025 through 30 June 2026 and resulted in an unqualified opinion.

\f2. DATA CLASSIFICATION AND ENCRYPTION AT REST

Customer data is classified into four tiers: Public, Internal, Confidential, and Restricted. Restricted data includes authentication secrets, payment instrument tokens, and customer content.

All data at rest must be encrypted using AES-256-GCM. Encryption keys are generated and stored in AWS Key Management Service backed by FIPS 140-3 validated hardware security modules. Customer-managed keys are supported through the KMS Bring Your Own Key programme.

Data encryption keys are rotated automatically every 90 days. Root keys are rotated annually. Key access is governed by least-privilege IAM policies and every cryptographic operation is recorded in CloudTrail with a 400-day retention period.

Database backups, object storage buckets, and message queues are encrypted by default; unencrypted storage is blocked by an organisation-wide service control policy.

\f3. INCIDENT RESPONSE AND SERVICE LEVELS

Security incidents are triaged using a four-level severity model. A Severity 1 incident is defined as confirmed or suspected unauthorised access to Restricted data, or a production outage affecting more than 25% of customers.

The guaranteed triage timeline for a Severity 1 security incident is 15 minutes from detection by the on-call security engineer. Containment must begin within 1 hour, and an initial customer notification must be issued within 24 hours. A written root-cause analysis is delivered within 5 business days.

Severity 2 incidents require triage within 2 hours and resolution within 24 hours. Severity 3 and 4 incidents are handled during business hours with resolution targets of 5 and 15 business days respectively.

The security operations centre operates 24x7x365 and maintains a documented escalation path to the CISO and General Counsel. Tabletop exercises are conducted quarterly, and the incident response plan is reviewed annually.

\f4. ACCESS CONTROL, VENDOR MANAGEMENT AND AUDIT

Access to production systems requires hardware-backed multi-factor authentication, least-privilege role assignment, and quarterly access reviews. Privileged access is time-bound and requires approval through the just-in-time access workflow.

Vendor risk assessments are performed before onboarding and annually thereafter. Subprocessors handling Restricted data must provide a current SOC 2 Type II report or ISO 27001 certification.

Audit logs are immutable, centralised, and retained for 400 days. Log integrity is protected using hash chaining and write-once storage. Security awareness training is mandatory for all staff at onboarding and annually thereafter.
"""

RAG_SPEC = """DOCUMIND HYBRID GRAPH RAG — ARCHITECTURE SPECIFICATION
Version 3.0 · Platform Engineering · LangChain + LangGraph

1. SYSTEM OVERVIEW

DocuMind ingests enterprise documents, segments them into page-aware chunks, and answers questions with verifiable page citations. The pipeline is orchestrated as a LangGraph state machine so every stage is observable, retryable, and independently testable.

Ingestion extracts text with pypdf, splits it using LangChain's RecursiveCharacterTextSplitter with 1,200-character chunks and 200-character overlap, extracts salient entities per chunk, and stores 768-dimensional embeddings in PostgreSQL with the pgvector extension.

The retrieval layer is hybrid: dense cosine similarity over HNSW indexes is fused with BM25 lexical scoring. Lexical matching protects exact identifiers, acronyms, and numeric clauses that dense embeddings often blur.

\f2. HYBRID RETRIEVAL AND RECIPROCAL RANK FUSION

Dense and lexical candidates are fused with Reciprocal Rank Fusion. For each query the fused score is computed as:

    RRF(d) = Σ over retrievers i of 1 / (k + rank_i(d))

where rank_i(d) is the 1-indexed rank of document chunk d in retriever i, and k is a smoothing constant. DocuMind uses k = 60, the value recommended in the original RRF paper, because it dampens the influence of top ranks while preserving ordering.

The production fusion weights are 0.45 for dense cosine similarity, 0.25 for normalised BM25, 0.20 for the normalised RRF score, and 0.10 for graph expansion score. Candidates whose similarity is below 0.28 and whose lexical score is below 0.08 are discarded.

The HNSW index is configured with m = 32, ef_construction = 200, and ef_search = 64. These parameters were selected by sweeping recall@10 against p95 latency on a 2.4 million chunk corpus; they deliver 0.94 recall@10 at approximately 11 milliseconds per query.

\f3. GRAPH EXPANSION

Every chunk stores up to 12 salient entities: acronyms, proper nouns, quantities, and technical terms. The entity graph connects chunks that share an entity, weighted by inverse document frequency so that rare entities such as "SOC 2" or "pgvector" create stronger edges than generic ones.

After the initial hybrid ranking, the top six seed chunks are selected. Their entities are used to fetch neighbouring chunks through the chunk_entities join table, capped at 16 entities and 8 neighbour chunks. Graph neighbours receive a bonus of up to 0.10 in the fused score.

Graph expansion improves multi-hop questions, such as linking a policy requirement defined on one page to the control evidence described on another. It is bounded so a single query cannot fan out across the entire corpus.

\f4. LANGGRAPH WORKFLOW, GROUNDING AND ABSTENTION

The workflow executes the following nodes: analyze, retrieve, grade, rewrite, generate, verify, revise, abstain, and finalize. Conditional edges implement two bounded loops: a retrieval rewrite loop of at most two attempts, and a citation-revision loop of at most one attempt.

Query analysis normalises the question, extracts keywords and entities, and produces up to four query variants for multi-query retrieval. Relevance grading measures query term coverage across the retrieved passages before generation is allowed.

Generation is strictly grounded: the model receives numbered passages and must cite every factual sentence. A deterministic verifier computes claim-level lexical support between the answer and its cited passages; answers below the 0.55 grounding threshold are revised or pruned.

If the evidence is insufficient the workflow abstains with a fixed, auditable message instead of speculating. Abstention correctness on out-of-domain questions is tracked as a first-class benchmark metric alongside recall@4 and faithfulness.
"""

DEFAULT_DOCUMENTS: tuple[SeedDocument, ...] = (
    SeedDocument(
        title="Apex Holdings Q3 2026 Financial Report",
        filename="apex-holdings-q3-2026.pdf",
        mime_type="application/pdf",
        text=APEX_REPORT,
    ),
    SeedDocument(
        title="SOC 2 Security & Encryption Policy",
        filename="soc2-security-policy.pdf",
        mime_type="application/pdf",
        text=SOC2_POLICY,
    ),
    SeedDocument(
        title="DocuMind Hybrid Graph RAG Architecture",
        filename="documind-rag-architecture.md",
        mime_type="text/markdown",
        text=RAG_SPEC,
    ),
)
