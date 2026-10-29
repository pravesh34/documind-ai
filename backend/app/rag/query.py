"""Query analysis, multi-query expansion and search-query rewriting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.ai.llm import LLMService
from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import content_tokens, unique_ordered
from app.rag.entities import extract_entities
from app.rag.prompts import expansion_messages, rewrite_messages

logger = get_logger("rag.query")

Intent = Literal["fact", "definition", "comparison", "policy", "metric", "summary"]

FILLER_PREFIXES = re.compile(
    r"^(?:hey|hi|hello|please|kindly|could you|can you|would you|tell me|"
    r"i want to know|i'd like to know|do you know|explain to me)\b[\s,:-]*",
    re.IGNORECASE,
)

INTENT_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    ("definition", re.compile(r"\b(what is|what are|define|meaning of)\b", re.I)),
    ("comparison", re.compile(r"\b(compare|difference|versus|vs\.?|better than)\b", re.I)),
    ("policy", re.compile(r"\b(policy|mandate|require|compliance|must|shall)\b", re.I)),
    ("metric", re.compile(r"\b(revenue|margin|ebitda|cost|percentage|rate|sla)\b", re.I)),
    ("summary", re.compile(r"\b(summarize|summary|overview|describe)\b", re.I)),
]


@dataclass(slots=True)
class QueryAnalysis:
    original: str
    normalized: str
    keywords: list[str]
    entities: list[str]
    intent: Intent
    sub_queries: list[str] = field(default_factory=list)


class QueryAnalyzer:
    def __init__(self, settings: Settings, llm: LLMService | None = None) -> None:
        self.settings = settings
        self.llm = llm

    async def analyze(self, question: str) -> QueryAnalysis:
        normalized = normalise_question(question)
        keywords = unique_ordered(content_tokens(normalized, min_length=3))[:10]
        entities = extract_entities(normalized, limit=8)

        sub_queries: list[str] = [normalized]
        if self.settings.multi_query_enabled and keywords:
            keyword_query = " ".join(keywords[:6])
            if keyword_query and keyword_query != normalized.lower():
                sub_queries.append(keyword_query)
            if entities:
                entity_query = " ".join(entities[:4])
                if entity_query not in sub_queries:
                    sub_queries.append(entity_query)

        if self.settings.multi_query_enabled and self.llm and self.llm.available:
            generated = await self._llm_expansion(normalized, count=3)
            sub_queries.extend(generated)

        return QueryAnalysis(
            original=question,
            normalized=normalized,
            keywords=keywords,
            entities=entities,
            intent=classify_intent(normalized),
            sub_queries=unique_ordered(sub_queries)[:4],
        )

    async def rewrite(self, question: str, queries: list[str], score: float) -> list[str]:
        if self.llm and self.llm.available:
            generated = await self._llm_rewrite(question, queries, score)
            if generated:
                return unique_ordered(generated)[:4]

        return deterministic_rewrites(question, queries)

    async def _llm_expansion(self, question: str, count: int) -> list[str]:
        assert self.llm is not None
        result = await self.llm.complete(
            expansion_messages(question, count),
            temperature=0.2,
            node="query_expansion",
        )
        if result is None:
            return []
        return _parse_query_lines(result.text)

    async def _llm_rewrite(self, question: str, queries: list[str], score: float) -> list[str]:
        assert self.llm is not None
        result = await self.llm.complete(
            rewrite_messages(question, queries, score, count=3),
            temperature=0.2,
            node="query_rewrite",
        )
        if result is None:
            return []
        return _parse_query_lines(result.text)


def normalise_question(question: str) -> str:
    value = re.sub(r"\s+", " ", question).strip()
    # Strip stacked conversational prefixes ("Please tell me ...", "Hey, could
    # you ...") rather than only the first match.
    previous = None
    while previous != value:
        previous = value
        value = FILLER_PREFIXES.sub("", value).strip(" \t\n'\"`")
    if value.endswith("?"):
        value = value[:-1].strip()
    return value or question.strip()


def classify_intent(question: str) -> Intent:
    for intent, pattern in INTENT_PATTERNS:
        if pattern.search(question):
            return intent
    return "fact"


def deterministic_rewrites(question: str, queries: list[str]) -> list[str]:
    keywords = unique_ordered(content_tokens(question, min_length=3))
    entities = extract_entities(question, limit=6)

    candidates: list[str] = []
    if keywords:
        candidates.append(" ".join(keywords[:8]))
    if entities:
        candidates.append(" ".join(entities + keywords[:3]))
    if len(keywords) > 3:
        candidates.append(" ".join(keywords[:3]))
    candidates.append(f"{question.strip()} details evidence")

    return [
        candidate
        for candidate in unique_ordered(candidates)
        if candidate and candidate not in queries
    ][:3]


def _parse_query_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip(" \"'`")
        if 3 <= len(candidate) <= 240 and " " in candidate:
            lines.append(candidate)
    return unique_ordered(lines)
