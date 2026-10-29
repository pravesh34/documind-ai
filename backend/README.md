# DocuMind AI — Python Backend

Production Graph RAG service built with **FastAPI**, **LangChain** and **LangGraph**.

- Hybrid retrieval: pgvector HNSW (dense) + PostgreSQL full-text/BM25 (sparse) + entity-graph expansion, fused with Reciprocal Rank Fusion.
- LangGraph agentic workflow: query analysis → multi-query retrieval → relevance grading → bounded rewrite loop → grounded generation → claim verification → revision/abstention.
- Free/local AI only: Ollama, Gemini free tier, Groq free tier, OpenRouter `:free`, Hugging Face Inference, plus deterministic offline fallbacks.
- Zero-credential startup: in-memory repository + hashed embeddings + extractive answers.

## Quick start (no credentials)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs>. The three enterprise seed documents are
ingested automatically and the whole stack runs with no API keys.

## Production

```bash
docker compose up --build
```

## Layout

```
app/
  api/          routes.py, schemas.py
  ai/           embeddings.py, llm.py
  core/         errors, logging, rate_limit, security, middleware, serialization, utils
  db/           schema.py, session.py, init_db.py
  rag/          workflow.py, retrieval.py, graph.py, grading.py, citations.py,
                chunking.py, entities.py, extraction.py, query.py, prompts.py, bm25.py
  repositories/ base.py, memory.py, postgres.py
  services/     ingestion, documents, chat, evaluation, export, seed/benchmark data
  workers/      celery_app.py, tasks.py
tests/          test_pipeline.py, test_api.py, test_units.py
```

## Testing and linting

```bash
pytest                     # service, HTTP contract and unit tests (in-memory repo)
ruff check app tests
mypy app                   # optional, strict-ish
```

The test suite never needs network access or credentials: it forces
`LLM_PROVIDER=none` and `EMBEDDING_PROVIDER=deterministic`.

## Enabling free providers

```bash
# Local, unlimited
ollama pull llama3.2:3b && ollama pull nomic-embed-text

# Hosted free tiers (any subset)
export GEMINI_API_KEY=...      # chat + embeddings
export GROQ_API_KEY=...        # chat
export OPENROUTER_API_KEY=...  # chat (:free models)
export HF_API_TOKEN=...        # embeddings
```

`LLM_PROVIDER=auto` / `EMBEDDING_PROVIDER=auto` builds an ordered fallback chain
and degrades to extractive answers / hashed embeddings if every provider fails.

See the root `README.md` for the full architecture, environment matrix and
operations runbook.
