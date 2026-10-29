# DocuMind AI

Production-grade **Graph RAG** document assistant: an advanced **LangGraph** agentic
retrieval loop over a **FastAPI** service, with a React + Tailwind UI for document
library management, grounded chat with page citations, a benchmark lab and a live
source-export view.

Everything runs on **free or local AI providers** â€” and with **zero credentials** the
system still works end to end (hashed embeddings + extractive answers).

```
React 19 + Vite + Tailwind  â”€â”€â–º  FastAPI /api/v1
                                     â”‚
                     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                     â”‚        LangGraph workflow      â”‚
                     â”‚ analyze â†’ retrieve â†’ grade     â”‚
                     â”‚   â†º rewrite      â†“ generate    â”‚
                     â”‚            verify â†º revise     â”‚
                     â”‚              â†“ abstain/finalizeâ”‚
                     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                     â”‚
              Hybrid retrieval: pgvector HNSW + BM25 + entity graph (RRF)
                                     â”‚
            PostgreSQL 17 + pgvector Â· Redis Â· Celery Â· free LLM APIs
```

---

## 1. What makes this "advanced" RAG

| Capability | Implementation |
|---|---|
| **Agentic orchestration** | `backend/app/rag/workflow.py` â€” a compiled LangGraph `StateGraph` with conditional edges and two bounded loops (retrieval rewrite â‰¤ 2, citation revision â‰¤ 1). |
| **Query understanding** | Query normalisation, intent classification, keyword/entity extraction, deterministic multi-query expansion plus optional LLM expansion (`rag/query.py`). |
| **Hybrid retrieval** | Dense cosine search (pgvector HNSW) âˆª PostgreSQL full-text search, fused with Reciprocal Rank Fusion `k=60` (`rag/retrieval.py`, `repositories/postgres.py`). |
| **Graph RAG** | Per-chunk salient entities, IDF-weighted entity co-occurrence edges, one-hop expansion with capped fan-out (`rag/graph.py`, `rag/entities.py`). |
| **Relevance grading** | Query-term coverage + unknown-entity ratio, with an optional LLM judge (`rag/grading.py`). Unknown entities force a rewrite or abstention instead of a hallucination. |
| **Grounded generation** | Numbered passages, mandatory `[n]` citations, strict system prompt (`rag/prompts.py`). |
| **Claim-level verification** | Deterministic sentence-level lexical support scoring + one revision/prune pass (`rag/citations.py`, `rag/workflow.py`). |
| **Abstention** | Fixed auditable message when evidence is insufficient; tracked as a first-class benchmark metric. |
| **Evaluation** | 8-case dataset incl. 2 negative tests; recall@k, faithfulness, hallucination rate, abstention accuracy, p50 latency. |
| **Observability** | Per-request node trace returned by `/chat`, request IDs, structured logs, Prometheus `/metrics`, `/healthz`, `/readyz`. |
| **Multi-tenancy** | Every relational and vector query is scoped by `tenant_id`; API keys map to tenants. |

---

## 2. Quick start (no Docker, no keys)

Requirements: Python 3.12+, Node 20+ (only for the UI).

```bash
# 1. Backend (in-memory repo, deterministic embeddings, extractive answers)
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

The three seed documents (Apex Q3 2026 report, SOC 2 policy, RAG architecture spec)
are ingested at startup. Open <http://localhost:8000/docs>.

```bash
# 2. Frontend (separate terminal, project root)
npm install
npm run dev        # http://localhost:5173, proxies /api â†’ :8000
```

> The UI's **Architecture & Export** tab streams the real backend source tree from the
> running service â€” it is not a mock.

## 3. Docker Compose (full stack)

```bash
cp .env.example .env          # optional; sane defaults are built in
docker compose up --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d   # prod shape
```

Services: `api` (uvicorn, serves the built UI), `worker` (Celery re-embedding),
`postgres` (pgvector/pg17), `redis`.

The default compose file runs with `AUTH_MODE=none` so the UI works immediately.
For the production shape, `docker-compose.prod.yml` enables `AUTH_MODE=api-key`;
pass `API_KEYS=key:tenant` and enter the key once in the header's **API Key** button
(or, equivalently, from the browser console):

```js
localStorage.setItem('documind.apiKey', 'dev-local-key');
```

## 4. Free cloud deployment (Render API + Vercel frontend)

`render.yaml` (Render Blueprint) and `vercel.json` in the repo root configure
both platforms. Steps:

1. **Render â€” API:** <https://dashboard.render.com/blueprints> â†’ *New Blueprint
   Instance* â†’ connect this repo â†’ **Apply**. It provisions:
   - `documind-api` (Python 3.13, free tier): `pip install .`,
     `uvicorn app.main:app --port $PORT --proxy-headers`, health check `/readyz`.
   - `documind-db` (free PostgreSQL, auto-injects `DATABASE_URL` â€” pgvector is
     created by the idempotent schema bootstrap on first boot).
   - Prompts for `CORS_ORIGINS`; set it to your Vercel URL
     (e.g. `https://documind-ai.vercel.app`) once known. `API_KEYS` is
     auto-generated â€” note it from the environment tab.
   - Free-tier notes: the web service sleeps after ~15 min idle (cold start
     â‰ˆ 40 s; the seed corpus rebuilds itself in-memory each wake), and the free
     Postgres expires after 30 days â€” swap in a Neon/Supabase URL or delete the
     DB service + `DATABASE_URL` var to keep running in-memory.
2. **Vercel â€” frontend:** <https://vercel.com/new> â†’ import the repo.
   Framework preset: *Vite* (from `vercel.json`). Before the first deploy add,
   under Environment Variables:
   ```
   VITE_API_BASE_URL = https://documind-api.onrender.com   (your Render URL)
   ```
3. **Connect the two:** copy your Render service URL
   (`https://documind-api-xxxx.onrender.com`) into the `CORS_ORIGINS` env var on
   Render, and your Vercel domain (including any wildcard preview domain if you
   want previews to work) into `VITE_API_BASE_URL` on Vercel, then redeploy both.
4. Open the Vercel URL â†’ paste the generated `API_KEYS` value into the header
   **API Key** button once (stored in the browser's localStorage).
5. Verify: `https://documind-api-xxxx.onrender.com/readyz` returns 200, then ask
   anything in the UI â€” answers come back grounded with page citations.

Optional hardening for demos: ping `/healthz` every 5-10 min (e.g. with cron-job.org or uptime-robot) to keep the free instance warm and avoid cold starts.

## 5. Free provider matrix

Set only what you have. `LLM_PROVIDER=auto` / `EMBEDDING_PROVIDER=auto` register every
available provider in priority order and fall through on failure.

**No keys handy? Bring your own from the UI.** The navbar **AI Providers** button lets
each user paste Gemini/Groq/OpenRouter/Hugging Face/Ollama settings. Keys are stored in
that browser's localStorage only and sent per request in an `x-ai-config` JSON header —
the server never persists them. Overrides apply to chat, retrieval and benchmark runs;
requests without the header use the environment-configured providers above.

| Provider | Free tier | Chat | Embeddings | Env |
|---|---|---|---|---|
| **Deterministic (built in)** | unlimited, offline | extractive answers | hashed BoW, 768-d | defaults |
| **Ollama** | unlimited, local | âœ… `llama3.2:3b` | âœ… `nomic-embed-text` | `OLLAMA_BASE_URL` |
| **Google Gemini** | free tier | âœ… `gemini-2.5-flash` | âœ… `gemini-embedding-001` | `GEMINI_API_KEY` |
| **Groq** | free tier | âœ… `openai/gpt-oss-120b` | â€” | `GROQ_API_KEY` |
| **OpenRouter** | `:free` models | âœ… | â€” | `OPENROUTER_API_KEY` |
| **Hugging Face** | free inference credits | â€” | âœ… `BAAI/bge-small-en-v1.5` | `HF_API_TOKEN` |

Notes:

- Auto-config probes `OLLAMA_BASE_URL` with a 250 ms TCP connect, so a machine without
  Ollama never pays connection timeouts.
- Remote embeddings are applied on upload in the background (Celery when configured,
  otherwise an in-process task). Deterministic vectors are always available immediately,
  so retrieval never blocks on a provider.
- Every LLM call is retried with exponential jitter and falls back across providers;
  final fallback is the deterministic extractive answerer.

## 6. Configuration

All settings live in `backend/app/config.py` (Pydantic Settings) and are documented in
`.env.example`. Production validation enforces:

- `DATABASE_URL` is set
- `API_KEYS` is non-empty when `AUTH_MODE=api-key` (the production default)
- `CHUNK_OVERLAP < CHUNK_SIZE`, `64 â‰¤ EMBEDDING_DIMENSIONS â‰¤ 2000`

Key knobs: `RETRIEVAL_TOP_K`, `SIMILARITY_THRESHOLD`, `MAX_RETRIEVAL_ATTEMPTS`,
`GROUNDING_THRESHOLD`, `GRAPH_EXPANSION_ENABLED`, `RATE_LIMIT_CHAT`.

## 7. API surface

Base path `/api/v1`. All responses are camelCase.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/documents?limit&offset&q` | Paginated library |
| `POST` | `/documents/upload` | JSON upload (`fileBase64` or `textContent`) |
| `POST` | `/documents/upload-file` | Multipart PDF/text upload |
| `GET`/`DELETE` | `/documents/{id}` | Detail (pages + chunks) / delete |
| `POST` | `/documents/reset-defaults` | Re-seed the demo corpus |
| `POST` | `/chat` | Graph RAG answer + citations + trace |
| `POST` | `/retrieval` | Retrieval-only debug view |
| `GET` | `/eval/dataset`, `/eval/latest`, `/eval/history` | Benchmark data |
| `POST` | `/eval/run` | Run the suite (`customTests` to override) |
| `GET` | `/export/files`, `/export/download` | Source browser / zip |
| `GET` | `/info` | Runtime provider + config summary |
| â€” | `/healthz`, `/readyz`, `/metrics` | Ops probes (no auth) |

Auth: `x-api-key: <key>` or `Authorization: Bearer <key>` when `AUTH_MODE=api-key`.

## 8. Data model (PostgreSQL + pgvector)

- `tenants` â€” tenant registry
- `documents` â€” metadata, extracted `pages` (jsonb), status, embedding status
- `chunks` â€” page-aware text, `vector(768)` embedding, generated `tsv`, char offsets
- `chunk_entities` â€” normalized entity index for graph expansion
- `benchmark_reports` â€” JSONB report history

Schema bootstrap is idempotent DDL under `pg_advisory_xact_lock`, so N replicas can boot
concurrently without racing (`backend/app/db/init_db.py`).

Indexes: HNSW (`vector_cosine_ops`, m=32 / ef_construction=200), GIN on `tsv`,
`(tenant_id, created_at DESC)`, `(tenant_id, lower(entity))`.

## 8. Testing

```bash
cd backend
pytest                    # 40+ tests: pipeline, API contract, units
ruff check app tests
```

Coverage focus: chunk offset integrity, RRF/BM25 ranking, entity graph edges, grounding
verification, abstention on negative tests, tenant isolation, API error envelopes.

## 9. Operations runbook

| Symptom | Check |
|---|---|
| `/readyz` 503 | `curl /readyz` â†’ `checks.repository`; verify `DATABASE_URL` and pgvector extension. |
| Chat returns extractive text | `/api/v1/info` â†’ `chatProviders` is `["extractive"]`; no provider key/Ollama reachable. |
| `embeddingStatus: unavailable` | Remote embedding failed; deterministic vectors remain active. Inspect worker logs. |
| Slow first query | Model cold start; `latencyMs` and the per-node `trace` isolate generation vs retrieval. |
| 429 responses | Raise `RATE_LIMIT_DEFAULT` / `RATE_LIMIT_CHAT` or configure Redis-backed limiter. |

Scaling: API and worker are stateless â€” scale horizontally. pgvector HNSW handles
millions of chunks per node; beyond that shard by `tenant_id`. Redis backs rate limiting
and the Celery broker.

## 10. Layout

```
backend/
  app/
    api/          routes.py (all endpoints), schemas.py
    ai/           embeddings.py (fallback chain + LRU), llm.py (LangChain chains)
    core/         errors, logging, security, middleware, serialization, utils
    db/           schema.py (DDL), session.py, init_db.py
    rag/          workflow.py, retrieval.py, graph.py, grading.py,
                  citations.py, chunking.py, entities.py, query.py, prompts.py, bm25.py
    repositories/ base.py, memory.py, postgres.py
    services/     ingestion, documents, chat, evaluation, export, seed/benchmark data
    workers/      celery_app.py, tasks.py
  tests/          test_pipeline.py, test_api.py, test_units.py
src/              React UI (unchanged design, now on /api/v1)
docker-compose.yml, docker-compose.prod.yml, backend/Dockerfile
.github/workflows/ci.yml
```

## 11. License

MIT.
