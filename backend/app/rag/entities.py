"""Entity extraction used to build the chunk/entity graph.

The default extractor is deterministic and dependency-free so ingestion stays
fast and works offline. An optional LLM extractor can be enabled with
``ENTITY_EXTRACTION_MODE=llm``.
"""

from __future__ import annotations

import re
from collections import Counter

from app.ai.llm import LLMService
from app.config import Settings
from app.core.logging import get_logger
from app.core.utils import STOPWORDS, unique_ordered
from app.rag.prompts import entity_messages

logger = get_logger("rag.entities")

ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9&.'-]+(?:\s+(?:of|the|and|for|de|van|von)?\s*[A-Z][a-zA-Z0-9&.'-]+){0,3}\b"
)
QUANTITY_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?"
    r"(?:%|percent|USD|EUR|GBP|ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
TECHNICAL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){1,3}\b", re.IGNORECASE)
QUOTED_RE = re.compile(r"[\"'“”]([^\"'“”]{3,60})[\"'“”]")

GENERIC_TERMS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "additionally",
    "conclusion",
    "summary",
    "introduction",
    "overview",
    "table",
    "figure",
    "section",
    "appendix",
    "page",
    "document",
    "documents",
    "report",
    "reports",
    "company",
    "companies",
    "inc",
    "ltd",
    "llc",
    "corp",
    "united",
    "states",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

# Entity labels that appear so often in prose that they add no graph signal.
LOW_SIGNAL_TERMS = {
    "q1",
    "q2",
    "q3",
    "q4",
    "fy",
    "yes",
    "no",
    "not",
    "all",
    "any",
    "each",
    "per",
    "etc",
}


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\n\r.,;:()[]{}<>\"'`")


def _is_meaningful(value: str) -> bool:
    candidate = _normalise(value)
    if len(candidate) < 3 or len(candidate) > 64:
        return False
    lowered = candidate.lower()
    if lowered in STOPWORDS or lowered in GENERIC_TERMS or lowered in LOW_SIGNAL_TERMS:
        return False
    if lowered.isdigit():
        return False
    return True


def extract_entities(text: str, limit: int = 12) -> list[str]:
    """Heuristic entity extraction: acronyms, proper nouns, quantities, terms."""
    if not text.strip():
        return []

    scores: Counter[str] = Counter()
    display: dict[str, str] = {}

    def add(value: str, weight: int) -> None:
        candidate = _normalise(value)
        if not _is_meaningful(candidate):
            return
        key = candidate.lower()
        scores[key] += weight
        display.setdefault(key, candidate)

    for match in ACRONYM_RE.findall(text):
        if any(character.isalpha() for character in match):
            add(match, 4)

    for match in PROPER_NOUN_RE.findall(text):
        words = match.split()
        if len(words) == 1 and match.islower():
            continue
        add(match, 3 if len(words) > 1 else 2)

    for match in QUANTITY_RE.findall(text):
        add(match, 2)

    for match in TECHNICAL_RE.findall(text):
        if "-" in match and len(match) >= 5:
            add(match, 2)

    for match in QUOTED_RE.findall(text):
        add(match, 2)

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    return unique_ordered(display[key] for key, _ in ranked)[:limit]


class EntityExtractor:
    def __init__(self, settings: Settings, llm: LLMService | None = None) -> None:
        self.settings = settings
        self.llm = llm
        self.mode = settings.entity_extraction_mode
        self.limit = settings.entity_extraction_limit

    async def extract(self, text: str) -> list[str]:
        heuristic = extract_entities(text, self.limit)
        if self.mode != "llm" or self.llm is None or not self.llm.available:
            return heuristic

        try:
            payload = await self.llm.complete_json(
                entity_messages(text[:6_000], self.limit),
                node="entity_extraction",
            )
        except Exception as error:  # noqa: BLE001 - heuristic is the fallback
            logger.warning("llm entity extraction failed", error=str(error))
            return heuristic

        raw = payload.get("entities") if payload else None
        if not isinstance(raw, list):
            return heuristic

        cleaned = [
            _normalise(str(value))
            for value in raw
            if isinstance(value, (str, int, float))
        ]
        merged = unique_ordered([*cleaned, *heuristic])[: self.limit]
        return merged or heuristic
