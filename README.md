# DocuMind AI

Production-grade **Graph RAG** document assistant: an advanced **LangGraph** agentic
retrieval loop over a **FastAPI** service, with a React + Tailwind UI for document
library management, grounded chat with page citations, a benchmark lab and a live
source-export view.

Everything runs on **free or local AI providers** - and with **zero credentials** the
system still works end to end (hashed embeddings + extractive answers).

```text
React 19 + Vite + Tailwind  -->  FastAPI /api/v1
                                      |
                      +---------------+----------------+
                      |        LangGraph workflow      |
                      | analyze -> retrieve -> grade    |
                      |   | rewrite       | generate     |
                      |            verify | revise       |
                      |              | abstain/finalize  |
                      +---------------+----------------+
                                      |
              Hybrid retrieval: pgvector HNSW + BM25 + entity graph (RRF)
                                      |
            PostgreSQL 17 + pgvector . Redis . Celery . free LLM APIs
```

---

## 1. What makes this "advanced" RAG

| Capability | Implementation |
|---|---|
| **Agentic orchestration** | `backend/app/rag/workflow.py` - a compiled LangGraph `StateGraph` with conditional edges and two bounded loops (retrieval rewrite <= 2, citation revision <= 1). |
| **Query understanding** | Query normalisation, intent classification, keyword/entity extraction, deterministic multi-query expansion plus optional LLM expansion (`rag/query.py`). |
| **Hybrid retrieval** | Dense cosine search (pgvector HNSW) UNION PostgreSQL full-text search, fused with Reciprocal Rank Fusion `k=60` (`rag/retrieval.py`, `repositories/postgres.py`). |
| **Graph RAG** | Per-chunk salient entities, IDF-weighted entity co-occurrence edges, one-hop expansion with capped fan-out (`rag/graph.py`, `rag/entities.py`). |
| **Relevance grading** | Query-term coverage + unknown-entity ratio, with an optional LLM judge (`rag/grading.py`). Unknown entities force a rewrite or abstention instead of a hallucination. |
| **Grounded generation** | Numbered passages, mandatory `[n]` citations, strict system prompt (`rag/prompts.py`). |
| **Claim-level verification** | Deterministic sentence-level lexical support scoring + one revision/prune pass (`rag/citations.py`, `rag/workflow.py`). |
| **Abstention** | Fixed auditable message when evidence is insufficient; tracked as a first-class benchmark metric. |
| **Evaluation** | 8-case dataset incl. 2 negative tests; recall@k, faithfulness, hallucination rate, abstention accuracy, p50 latency. |
| **Observability** | Per-request node trace returned by `/chat`, request IDs, structured logs, Prometheus `/metrics`, `/healthz`, `/readyz`. |
| **Multi-tenancy** | Every relational and vector query is scoped by `tenant_id`; API keys map to tenants. |
| **Bring-your-own keys** | Navbar **AI Providers** modal: per-user Gemini/Groq/OpenRouter/HF/Ollama keys sent per request, never stored server-side. |

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
npm run dev        # http://localhost:5173, proxies /api -> :8000
```

> The UI's **Architecture & Export** tab streams the real backend source tree from the
> running service - it is not a mock.

Optional: copy `.env.example` to `.env` and add any free provider keys. The backend
reads `.env` from the `backend/` directory or the project root automatically.

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
both platforms.

1. **Render - API:** <https://dashboard.render.com/blueprints> -> *New Blueprint
   Instance* -> connect this repo -> **Apply**. It provisions:
   - `documind-api` (Python, free tier): `pip install .`,
     `uvicorn app.main:app --port $PORT --proxy-headers`, health check `/readyz`.
   - `documind-db` (free PostgreSQL, auto-injects `DATABASE_URL`; pgvector is
     created by the idempotent schema bootstrap on first boot).
   - Prompts for `CORS_ORIGINS`; set it to your Vercel URL
     (e.g. `https://documind-ai.vercel.app`) once known. `API_KEYS` is
     auto-generated - copy it from the service's Environment tab.
   - Free-tier notes: the web service sleeps after ~15 min idle (cold start
     ~40 s; seeded state is rebuilt on wake when running in-memory), and the free
     Postgres expires after 30 days - see the fallback below.
2. **Vercel - frontend:** <https://vercel.com/new> -> import the repo.
   Framework preset: *Vite* (from `vercel.json`). Before the first deploy add,
   under Environment Variables:
   ```
   VITE_API_BASE_URL = https://documind-api.onrender.com
   ```
   A bare origin is fine: `/api/v1` is appended automatically unless the value
   already ends in an API version prefix.
3. **Connect the two:** put your Vercel origin into `CORS_ORIGINS` on Render
   (comma-separate additional origins, e.g. preview deployments), then redeploy
   if needed. CORS preflight from an unlisted origin fails with HTTP 400.
4. Open the Vercel URL -> paste the generated `API_KEYS` value into the header
   **API Key** button once (stored in the browser's localStorage). That is the
   app's access key, not a provider key.
5. Verify: `https://documind-api-xxxx.onrender.com/readyz` returns 200, then ask
   anything in the UI - answers come back grounded with page citations.

**Free Postgres expiry (30 days).** Either paste a longer-lived free Postgres
(Neon/Supabase) into `DATABASE_URL`, or run without a database: delete the
`DATABASE_URL` variable and set `ALLOW_MEMORY_BACKEND=true` in the Render
environment. Without the flag, `ENVIRONMENT=production` requires `DATABASE_URL`
by design - the process would otherwise start in-memory by surprise.

**Keeping the free instance warm (optional).** Free services sleep after ~15 min
without traffic. An external ping (cron-job.org, UptimeRobot, ...) to
`https://<service>.onrender.com/healthz` every 5-10 minutes avoids cold starts;
there is no in-app self-ping.

## 5. Free provider matrix

Set only what you have. `LLM_PROVIDER=auto` / `EMBEDDING_PROVIDER=auto` register every
available provider in priority order and fall through on failure.

**No keys handy? Bring your own from the UI.** The navbar **AI Providers** button lets
each user paste Gemini/Groq/OpenRouter/Hugging Face/Ollama settings. Keys are stored in
that browser's localStorage only and sent per request in an `x-ai-config` JSON header -
the server never persists them. Overrides apply to chat, retrieval and benchmark runs;
requests without the header use the environment-configured providers above.

| Provider | Free tier | Chat | Embeddings | Env |
|---|---|---|---|---|
| **Deterministic (built in)** | unlimited, offline | extractive answers | hashed BoW, 768-d | defaults |
| **Ollama** | unlimited, local | yes - `llama3.2:3b` | yes - `nomic-embed-text` | `OLLAMA_BASE_URL` |
| **Google Gemini** | free tier | yes - `gemini-2.5-flash` | yes - `gemini-embedding-001` | `GEMINI_API_KEY` |
| **Groq** | free tier | yes - `openai/gpt-oss-120b` | - | `GROQ_API_KEY` |
| **OpenRouter** | `:free` models | yes | - | `OPENROUTER_API_KEY` |
| **Hugging Face** | free inference credits | - | yes - `BAAI/bge-small-en-v1.5` | `HF_API_TOKEN` |

Notes:

- Auto-config probes `OLLAMA_BASE_URL` with a 250 ms TCP connect, so a machine without
  Ollama never pays connection timeouts.
- Remote embeddings are applied on upload in the background (Celery when configured,
  otherwise an in-process task). Deterministic vectors are always available immediately,
  so retrieval never blocks on a provider.
- Every LLM call is retried with exponential jitter and falls back across providers;
  final fallback is the deterministic extractive answerer.
- Groq model names change over time; if you see `model_not_found`, pick a current
  model from <https://console.groq.com/docs/models> or set it in the AI Providers
  modal's *Custom models* section.

## 6. Configuration

All settings live in `backend/app/config.py` (Pydantic Settings) and are documented in
`.env.example`. Production validation enforces:

- `DATABASE_URL` is set unless `ALLOW_MEMORY_BACKEND=true`
- `API_KEYS` is non-empty when `AUTH_MODE=api-key` (the production default)
- `CHUNK_OVERLAP < CHUNK_SIZE`, `64 <= EMBEDDING_DIMENSIONS <= 2000`

Key knobs: `RETRIEVAL_TOP_K`, `SIMILARITY_THRESHOLD`, `MAX_RETRIEVAL_ATTEMPTS`,
`GROUNDING_THRESHOLD`, `GRAPH_EXPANSION_ENABLED`, `RATE_LIMIT_CHAT`.

Provider keys can be supplied either through the environment (server-wide defaults)
or per request from the browser (see the AI Providers modal above).

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
| - | `/healthz`, `/readyz`, `/metrics` | Ops probes (no auth) |

Auth: `x-api-key: <key>` or `Authorization: Bearer <key>` when `AUTH_MODE=api-key`.
Provider overrides: optional `x-ai-config: {json}` header on `/chat`, `/retrieval`
and `/eval/run`.

## 8. Data model (PostgreSQL + pgvector)

- `tenants` - tenant registry
- `documents` - metadata, extracted `pages` (jsonb), status, embedding status
- `chunks` - page-aware text, `vector(768)` embedding, generated `tsv`, char offsets
- `chunk_entities` - normalized entity index for graph expansion
- `benchmark_reports` - JSONB report history

Schema bootstrap is idempotent DDL under `pg_advisory_xact_lock`, so N replicas can boot
concurrently without racing (`backend/app/db/init_db.py`).

Indexes: HNSW (`vector_cosine_ops`, m=32 / ef_construction=200), GIN on `tsv`,
`(tenant_id, created_at DESC)`, `(tenant_id, lower(entity))`.

Without `DATABASE_URL` the same semantics run on an in-memory repository (dev, tests,
or `ALLOW_MEMORY_BACKEND=true` deployments).

## 9. Testing

```bash
cd backend
pytest                    # 69 tests: pipeline, API contract, units
ruff check app tests
```

Coverage focus: chunk offset integrity, RRF/BM25 ranking, entity graph edges, grounding
verification, abstention on negative tests, tenant isolation, API error envelopes,
per-request AI overrides. CI runs the same suite plus frontend typecheck/build and a
Docker image build on every push.

## 10. Operations runbook

| Symptom | Check |
|---|---|
| `404 Not Found` on every call | `VITE_API_BASE_URL` must point at the API host; `/api/v1` is appended automatically. A relative base on Vercel 404s because Vercel does not proxy `/api`. |
| Browser calls blocked / preflight 400 | The UI origin is missing from `CORS_ORIGINS` on the API. |
| `401 Unauthorized` | The navbar **API Key** value does not match `API_KEYS` (use the Render-generated value, format `key` or `key:tenant`). |
| Upload returns `413` | File exceeds `MAX_UPLOAD_BYTES` (25 MB default). |
| Upload returns `422 EMPTY_DOCUMENT` | The PDF has no extractable text layer (e.g. scanned image); OCR it first or upload text. |
| Upload returns `502` on free hosting | Instance ran out of memory; try a smaller PDF or a paid instance size. |
| `/readyz` 503 | `curl /readyz` -> `checks.repository`; verify `DATABASE_URL` and the pgvector extension. |
| Chat returns extractive text | `/api/v1/info` -> `chatProviders` is `["extractive"]`; no provider key/Ollama reachable (or the provider returned `model_not_found`). |
| `embeddingStatus: unavailable` | Remote embedding failed; deterministic vectors remain active. Inspect worker logs. |
| Slow first query | Model cold start (or free-tier instance wake); `latencyMs` and the per-node `trace` isolate generation vs retrieval. |
| 429 responses | Raise `RATE_LIMIT_DEFAULT` / `RATE_LIMIT_CHAT` or configure Redis-backed limiting. |

Scaling: API and worker are stateless - scale horizontally. pgvector HNSW handles
millions of chunks per node; beyond that shard by `tenant_id`. Redis backs rate limiting
and the Celery broker.

## 11. Layout

```text
backend/
  app/
    api/          routes.py (all endpoints), schemas.py, ai_overrides.py
    ai/           embeddings.py (fallback chain + LRU), llm.py (LangChain chains)
    core/         errors, logging, security, middleware, serialization, utils
    db/           schema.py (DDL), session.py, init_db.py
    rag/          workflow.py, retrieval.py, graph.py, grading.py,
                  citations.py, chunking.py, entities.py, query.py, prompts.py, bm25.py
    repositories/ base.py, memory.py, postgres.py
    services/     ingestion, documents, chat, evaluation, export, seed/benchmark data
    workers/      celery_app.py, tasks.py
  tests/          test_pipeline.py, test_api.py, test_units.py
src/              React UI (documents, chat, benchmark, export, AI providers)
docker-compose.yml, docker-compose.prod.yml, backend/Dockerfile
render.yaml, vercel.json, .github/workflows/ci.yml
```

## 12. License

MIT.
