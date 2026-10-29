"""Source-code export: browse and download the running backend repository."""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

CATEGORY_BY_PREFIX: list[tuple[str, str]] = [
    ("backend/app/rag/", "rag"),
    ("backend/app/ai/", "rag"),
    ("backend/app/services/", "rag"),
    ("backend/app/workers/", "celery"),
    ("backend/app/db/", "database"),
    ("backend/app/repositories/", "database"),
    ("backend/app/api/", "fastapi"),
    ("backend/app/core/", "fastapi"),
    ("backend/app/main.py", "fastapi"),
    ("backend/app/container.py", "fastapi"),
    ("backend/app/config.py", "fastapi"),
    ("backend/app/domain.py", "fastapi"),
    ("backend/tests/", "tests"),
    ("backend/Dockerfile", "docker"),
    ("backend/.dockerignore", "docker"),
    ("docker-compose.yml", "docker"),
    ("docker-compose.prod.yml", "docker"),
    ("docs/", "docs"),
    ("README.md", "docs"),
]

DESCRIPTIONS: dict[str, str] = {
    "backend/app/main.py": "FastAPI application factory, lifespan and middleware wiring.",
    "backend/app/container.py": "Composition root with dependency wiring and readiness state.",
    "backend/app/config.py": "Pydantic Settings with free-provider validation.",
    "backend/app/domain.py": "Framework-free domain dataclasses.",
    "backend/app/api/routes.py": "Versioned REST endpoints for documents, chat, eval and export.",
    "backend/app/api/schemas.py": "Pydantic v2 request/response models (camelCase aliases).",
    "backend/app/rag/workflow.py": "LangGraph state machine with grading, revision and abstention.",
    "backend/app/rag/retrieval.py": "Hybrid dense + BM25 + graph retrieval with RRF fusion.",
    "backend/app/rag/graph.py": "Entity co-occurrence graph and IDF-weighted expansion.",
    "backend/app/rag/grading.py": "Retrieval relevance grading and optional LLM judge.",
    "backend/app/rag/citations.py": "Citation selection and claim-level grounding verification.",
    "backend/app/rag/chunking.py": "Page-aware LangChain chunking.",
    "backend/app/rag/entities.py": "Deterministic entity extraction.",
    "backend/app/rag/query.py": "Query analysis, expansion and rewriting.",
    "backend/app/ai/llm.py": "LangChain chat chain across free providers with fallbacks.",
    "backend/app/ai/embeddings.py": "Free embedding chain with deterministic fallback and cache.",
    "backend/app/repositories/postgres.py": "Pgvector repository with HNSW + full-text prefetch.",
    "backend/app/repositories/memory.py": "In-memory repository for dev and tests.",
    "backend/app/db/schema.py": "Idempotent PostgreSQL + pgvector DDL.",
    "backend/app/workers/celery_app.py": "Celery application and queues.",
    "backend/app/workers/tasks.py": "Background re-embedding task.",
    "backend/tests/test_pipeline.py": "Service-level ingestion, retrieval and chat tests.",
    "backend/tests/test_api.py": "HTTP contract tests for every /api/v1 endpoint.",
    "backend/tests/test_units.py": "Unit tests for chunking, scoring, serialisation and settings.",
    "backend/Dockerfile": "Multi-stage build: Vite frontend + Python 3.13 runtime.",
    "docker-compose.yml": "Local/production-like stack: API, worker, PostgreSQL, Redis.",
    "docker-compose.prod.yml": "Production overrides: limits, no published DB ports.",
    ".env.example": "Documented environment template with every free-provider switch.",
    "README.md": "Architecture, free-provider matrix, API reference and operations runbook.",
}

ALLOWED_SUFFIXES = {
    ".py",
    ".toml",
    ".md",
    ".yml",
    ".yaml",
    ".txt",
    ".ini",
    ".cfg",
    ".json",
}
ALLOWED_NAMES = {"Dockerfile", ".dockerignore", ".env.example", "Makefile"}


@dataclass(frozen=True, slots=True)
class RepoFile:
    path: str
    category: str
    description: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "category": self.category,
            "description": self.description,
            "content": self.content,
        }


class ExportService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend_root = Path(__file__).resolve().parents[2]
        self.project_root = self.backend_root.parent

    def list_files(self) -> list[RepoFile]:
        files: list[RepoFile] = []
        for relative, path in self._iter_files():
            if not self.settings.export_include_tests and relative.startswith(
                "backend/tests/"
            ):
                continue
            if path.stat().st_size > 400_000:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            files.append(
                RepoFile(
                    path=relative,
                    category=self._category(relative),
                    description=DESCRIPTIONS.get(
                        relative, f"{relative.rsplit('/', 1)[-1]} source file"
                    ),
                    content=content,
                )
            )
        files.sort(key=lambda item: (item.category, item.path))
        return files

    def zip_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for relative, path in self._iter_files():
                if not self.settings.export_include_tests and relative.startswith(
                    "backend/tests/"
                ):
                    continue
                if path.stat().st_size > 400_000:
                    continue
                archive.write(path, arcname=f"documind-ai/{relative}")
        return buffer.getvalue()

    def _iter_files(self):
        roots = [
            self.backend_root / "app",
            self.backend_root / "tests",
            self.backend_root,
            self.project_root / "docs",
        ]
        seen: set[Path] = set()
        skipped_dirs = {
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".git",
        }

        for root in roots:
            if not root.exists():
                continue
            if root.is_dir():
                for current, directories, filenames in os.walk(root):
                    directories[:] = [name for name in directories if name not in skipped_dirs]
                    current_path = Path(current)
                    if root == self.backend_root and current_path != self.backend_root:
                        # Nested backend trees are covered by the explicit roots.
                        continue
                    for filename in filenames:
                        path = current_path / filename
                        yield from self._maybe_file(path, seen)
            else:
                yield from self._maybe_file(root, seen)

        for name in (
            "docker-compose.yml",
            "docker-compose.prod.yml",
            "README.md",
            ".env.example",
        ):
            path = self.project_root / name
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path.relative_to(self.project_root).as_posix(), path

    def _maybe_file(self, path: Path, seen: set[Path]):
        if not path.is_file() or path in seen:
            return
        if path.suffix.lower() not in ALLOWED_SUFFIXES and path.name not in ALLOWED_NAMES:
            return
        seen.add(path)
        yield path.relative_to(self.project_root).as_posix(), path

    def _category(self, relative: str) -> str:
        for prefix, category in CATEGORY_BY_PREFIX:
            if relative.startswith(prefix):
                return category
        return "docs"
