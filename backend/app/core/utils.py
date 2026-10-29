"""Small, dependency-light helpers used across the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.IGNORECASE)

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "as",
    "into",
    "about",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "have",
    "has",
    "had",
    "it",
    "its",
    "their",
    "there",
    "they",
    "we",
    "our",
    "you",
    "your",
    "i",
    "me",
    "my",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def content_tokens(text: str, *, min_length: int = 3) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) >= min_length and token not in STOPWORDS
    ]


def token_set(text: str, *, min_length: int = 3) -> set[str]:
    return set(content_tokens(text, min_length=min_length))


def lexical_overlap(query: str, text: str) -> float:
    query_terms = token_set(query)
    if not query_terms:
        return 0.0
    document_terms = token_set(text)
    if not document_terms:
        return 0.0
    return len(query_terms & document_terms) / len(query_terms)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(length):
        a = left[index]
        b = right[index]
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
    return dot / denominator if denominator > 0 else 0.0


def fit_dimensions(values: Sequence[float], dimensions: int) -> list[float]:
    """Truncate/pad (Matryoshka-safe) and L2-normalise a vector."""
    output = [0.0] * dimensions
    for index, value in enumerate(values[:dimensions]):
        if math.isfinite(value):
            output[index] = float(value)
    norm = math.sqrt(sum(value * value for value in output))
    if norm == 0:
        return output
    return [value / norm for value in output]


def truncate(value: str, max_length: int) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1].rstrip() + "…"


def token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        candidate = value.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response."""
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def message_text(response: Any) -> str:
    """Normalise LangChain AIMessage content (string or content blocks)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def provider_metadata(response: Any) -> tuple[str | None, str | None]:
    metadata = getattr(response, "response_metadata", None) or {}
    provider = metadata.get("model_provider") or metadata.get("provider")
    model = metadata.get("model_name") or metadata.get("model")
    return (
        str(provider) if provider else None,
        str(model) if model else None,
    )


def tcp_endpoint_open(url: str, *, timeout: float = 0.25) -> bool:
    """Return True when a local TCP endpoint (e.g. Ollama) accepts connections.

    Used only to decide whether to *register* an optional local provider during
    auto-configuration, so a machine without Ollama does not pay connection
    failures on every request.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
