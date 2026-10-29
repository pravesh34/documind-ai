"""Prompt templates for the LangGraph RAG workflow."""

from __future__ import annotations

from app.domain import ChunkRecord

SYSTEM_PROMPT = """You are DocuMind, a retrieval-augmented analyst.

NON-NEGOTIABLE RULES
1. Answer ONLY from the numbered context passages provided.
2. Cite every factual sentence with the passage number in square brackets, e.g. [1].
3. Never invent numbers, names, dates, policies, or citations.
4. If the passages do not contain the answer, reply exactly:
   "I cannot answer this from the ingested corpus because the retrieved passages do not contain sufficient evidence."
5. Prefer a direct answer first, then 2-4 supporting bullets. Be concise and factual.
6. When passages disagree, say so explicitly and cite both.
"""

ANSWER_PROMPT = """Question:
{question}

Retrieved context passages:
{context}

Write the grounded answer now."""

REWRITE_PROMPT = """The previous retrieval attempt did not surface enough evidence.

Original question: {question}
Previous search queries: {queries}
Best relevance score: {score:.2f}

Write {count} alternative search queries that use different vocabulary, synonyms,
or decompose the question into smaller factual lookups. Return one query per line
with no numbering and no commentary."""

QUERY_EXPANSION_PROMPT = """Rewrite the user question into {count} search queries for a
hybrid dense + keyword document search engine.

Rules:
- Keep entity names, acronyms and numbers verbatim.
- Cover synonyms and alternative phrasings.
- One query per line, no numbering, no commentary.

Question: {question}"""

VERIFY_PROMPT = """Check whether every factual claim in the answer is supported by
the context passages. Return JSON only:

{{"grounding_score": 0.0-1.0, "unsupported_claims": ["..."], "verdict": "supported"|"partially_supported"|"unsupported"}}

Context:
{context}

Answer:
{answer}"""

REVISE_PROMPT = """Rewrite the answer so that every factual claim is directly
supported by the context. Remove unsupported claims entirely. Keep the same
citation style ([1], [2]) and answer the question as far as the evidence allows.

Question:
{question}

Context:
{context}

Previous answer:
{answer}

Revised answer:"""

ENTITY_PROMPT = """Extract up to {limit} salient entities from the passage.
Entities are product names, organizations, policies, standards, metrics, dates,
amounts, or domain-specific technical terms. Return JSON only:

{{"entities": ["..."]}}

Passage:
{passage}"""


def format_context(chunks: list[ChunkRecord], max_chars: int = 12_000) -> str:
    """Render chunks as numbered passages, within a character budget."""
    blocks: list[str] = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"[{index}] {chunk.document_title} — page {chunk.page_number} "
            f"(chunk {chunk.chunk_index})"
        )
        body = chunk.text.strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def answer_messages(question: str, chunks: list[ChunkRecord]):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=ANSWER_PROMPT.format(
                question=question.strip(),
                context=format_context(chunks),
            )
        ),
    ]


def revision_messages(question: str, chunks: list[ChunkRecord], answer: str):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=REVISE_PROMPT.format(
                question=question.strip(),
                context=format_context(chunks),
                answer=answer.strip(),
            )
        ),
    ]


def rewrite_messages(question: str, queries: list[str], score: float, count: int = 3):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(
            content=(
                "You are a search query optimizer. Output only the queries, one per line."
            )
        ),
        HumanMessage(
            content=REWRITE_PROMPT.format(
                question=question.strip(),
                queries="; ".join(queries) or "(none)",
                score=score,
                count=count,
            )
        ),
    ]


def expansion_messages(question: str, count: int = 3):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(
            content="You are a search query optimizer. Output only the queries, one per line."
        ),
        HumanMessage(
            content=QUERY_EXPANSION_PROMPT.format(question=question.strip(), count=count)
        ),
    ]


def verification_messages(answer: str, chunks: list[ChunkRecord]):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content="You are a strict claim-level grounding auditor."),
        HumanMessage(
            content=VERIFY_PROMPT.format(
                context=format_context(chunks),
                answer=answer.strip(),
            )
        ),
    ]


def entity_messages(passage: str, limit: int):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content="You extract structured entities from documents."),
        HumanMessage(content=ENTITY_PROMPT.format(passage=passage, limit=limit)),
    ]
