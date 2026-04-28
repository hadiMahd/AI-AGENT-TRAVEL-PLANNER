# Checkpoint — Smart Travel Planner (Week 4)

**Last updated:** 2026-04-29
**LLM Provider:** Azure OpenAI (Kimi-K2.6-1 + DeepSeek-V3.2-1 + text-embedding-3-small)
**Status:** Infrastructure & RAG complete. Agent, Auth, Frontend, Tests, and Webhook not yet built.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│  React/Vite  │────▶│              FastAPI Backend             │
│  (boilerplate│     │                                          │
│   only)      │     │  Routers: health, rag, ml, auth*, user*  │
│              │     │  Services: ml_inference, send_plan*       │
└─────────────┘     │  RAG: load_doc → chunker → embedder →    │
                    │        retriever (pgvector cosine)          │
                    │  DB: PostgreSQL + pgvector (asyncpg)        │
                    │  Secrets: HashiCorp Vault (dev mode)       │
                    │  LLMs: Azure OpenAI via langchain-openai   │
                    │  Embeddings: text-embedding-3-small (1536) │
                    └──────────────────────────────────────────┘
```

---

## INSTRUCTIONS.md — 9 Mandatory Requirements: Status

| # | Requirement | Status | What's Done | What's Missing |
|---|-------------|--------|-------------|----------------|
| 1 | **ML Tool Integration** | ✅ Done | `services/ml_inference.py` — async wrapper with Pydantic input, `joblib` model loaded at lifespan, exposed via `Depends(get_model)`, `POST /ml/predict` endpoint, `models/schemas.py` with `TravelStyleFeatures` + `TravelStylePredictionResponse` | Needs to be wired as a LangGraph tool (not just a standalone route) |
| 2 | **RAG Tool** | ✅ Done | Full `rag/` pipeline (5 files), 19 JSON destination docs, `POST /rag/ingest`, `POST /rag/query`, pgvector cosine similarity, HNSW+GIN indexes, chunk_size=500 overlap=50, k=5, Alembic migrations | Needs to be wired as a LangGraph tool |
| 3 | **Agent With 3 Tools** | ❌ Not Started | DB tables `agent_runs` + `tool_logs` exist. Two LLM clients (strong/cheap) exist. | **LangGraph graph code, 3 tools (RAG retriever, ML predictor, live conditions fetcher), tool allowlist, synthesis logic, LangSmith tracing** |
| 4 | **Two Models, One Agent** | ✅ Infra Done | `llm.py` creates `strong_llm` (Kimi-K2.6-1) + `cheap_llm` (DeepSeek-V3.2-1). Both exposed via `Depends()`. Health check with 429 handling. | Needs to be USED by the agent graph (cheap for arg extraction/query rewrite, strong for synthesis) |
| 5 | **Persistence** | ✅ Done | SQLAlchemy 2.x async, `db/session.py`, 5 ORM models (`User`, `AgentRun`, `ToolLog`, `Document`, `DeliveryLog`), 3 Alembic migrations, `postgresql+asyncpg` | `AgentRun` + `ToolLog` rows need to be WRITTEN by the agent (no code writes them yet) |
| 6 | **Auth** | ❌ Not Started | `routers/auth.py` + `routers/user.py` + `routers/admin.py` exist as **empty files**. `User` ORM model exists. `JWT_SECRET_KEY` in `.env` + Vault. | **Sign-up, login, password hashing, JWT generation/validation, `get_current_user` dependency, scope runs to user** |
| 7 | **React Frontend** | ❌ Not Started | `frontend/` has Vite+React boilerplate (default counter app). `package.json` exists. `node_modules` installed. | **Chat UI, agent reasoning display (tools fired, inputs/outputs, plan), streaming responses, CORS config** |
| 8 | **Webhook Delivery** | ❌ Not Started | `services/send_plan.py` exists as **stub** (Resend import only, no logic). `DeliveryLog` ORM model exists. `RESEND_API_KEY` in `.env`. | **Complete send_plan service, timeout + retry with backoff, structured logging on failure, fire after agent response** |
| 9 | **Docker** | ✅ Done | `docker-compose.yml` with db (pgvector), vault, pgadmin, backend. Named volume for Postgres. `vault-init.sh` seeds secrets. Backend Dockerfile. | Frontend service not in docker-compose yet |

---

## File Inventory

### Backend (`backend/`) — 25 Python files

```
backend/
├── main.py                  # FastAPI app, lifespan, composite /health
├── config.py                 # pydantic-settings + HashiCorpVaultSource
├── dependencies.py           # Depends() for model, strong_llm, cheap_llm, embedder
├── llm.py                    # Azure OpenAI client factories + health check
├── models/
│   ├── alchemy.py            # SQLAlchemy ORM (5 tables: User, AgentRun, ToolLog, Document, DeliveryLog)
│   └── schemas.py            # Pydantic schemas (TravelStyleFeatures, TravelStylePredictionResponse, APIMessage)
├── db/
│   └── session.py            # Async engine, sessionmaker, get_db(), init_db(), dispose_db(), check_db_health()
├── routers/
│   ├── health.py             # GET /health/db, GET /health/llm
│   ├── rag.py                # POST /rag/ingest, POST /rag/query
│   ├── ml_model.py           # POST /ml/predict
│   ├── auth.py               # EMPTY
│   ├── user.py               # EMPTY
│   ├── admin.py              # EMPTY
│   └── llm.py                # EMPTY
├── services/
│   ├── ml_inference.py       # Async ML prediction (joblib model + Pydantic input)
│   └── send_plan.py          # STUB — Resend import only, no logic
├── rag/
│   ├── load_doc.py           # JSON document loader (anyio async)
│   ├── chuncker.py           # RecursiveCharacterTextSplitter wrapper
│   ├── embedder.py           # OpenAIEmbeddings batch+query helpers, EMBEDDING_DIM=1536, BATCH_SIZE=16
│   ├── ingestion.py           # Full pipeline: load → chunk → embed → wipe+insert
│   └── retriever.py          # Cosine similarity SQL with CAST(:embedding AS vector(1536))
└── alembic/
    ├── env.py                # Alembic config with pgvector render support
    └── versions/
        ├── 001_initial_schema.py   # Baseline (stamped)
        ├── 002_vector_dim.py       # 1536→768 (applied, historical)
        └── 003_vector_1536.py     # 768→1536 (for Azure OpenAI, needs `alembic upgrade head`)
```

### Project Root

```
project/
├── .env                     # Runtime config (Azure OpenAI keys, DB, Vault, LangSmith, Resend)
├── .env.example             # Template
├── .gitignore
├── INSTRUCTIONS.md          # All requirements and constraints
├── db-schema.sql            # Full DDL with vector(1536), HNSW+GIN indexes
├── docker-compose.yml       # db, vault, pgadmin, backend (no frontend yet)
├── vault-init.sh            # Seeds JW-SECRET-KEY + AZURE-OPENAI-KEY
├── documents/               # 19 JSON destination docs (6 categories)
├── frontend/                # Vite+React boilerplate (default counter app)
└── backend/                 # (see above)
```

---

## Key Design Decisions (defend in evaluation)

| Decision | Why |
|----------|-----|
| Lifespan singletons + `Depends()` | INSTRUCTIONS.md mandates it. Testable via `app.dependency_overrides`. No globals. |
| `ChatOpenAI` (v1 API) over `AzureChatOpenAI` | Simpler — no `api_version`/`azure_deployment` params. LangChain docs recommend for v1 endpoints. |
| `CAST(:embedding AS vector(1536))` over `::vector(1536)` | SQLAlchemy's `text()` parser chokes on `:param::type` — the `::` confuses named parameter detection. Standard SQL `CAST()` works. |
| HashiCorp Vault dev mode | Simple for dev, secrets re-seeded on start. Production would use Raft storage. |
| Full wipe-and-replace ingestion | Corpus is small (~19 files, ~25 chunks). Incremental upsert adds complexity for no benefit. |
| Alembic migrations (not fresh DB) | Trackable, reversible, defensible in evaluation. |
| `vector(1536)` (text-embedding-3-small default) over `dimensions=768` | Better retrieval quality with full 1536 dims. Worth the migration. |
| 429 rate-limit = "healthy but throttled" | API IS reachable, just throttled. Per INSTRUCTIONS.md: graceful fallback. |
| `BATCH_SIZE=16` for embeddings | Azure OpenAI has tighter TPM/RPM limits than local Ollama. 16 is conservative. |

---

## Working Endpoints (tested)

| Endpoint | Method | Status |
|----------|--------|--------|
| `/health` | GET | ✅ Composite health (API + ML + LLM) |
| `/health/db` | GET | ✅ Database connectivity |
| `/health/llm` | GET | ✅ Azure OpenAI reachability (TTL-cached 60s, 429=healthy) |
| `/ml/predict` | POST | ✅ Travel style prediction with confidence |
| `/rag/ingest` | POST | ✅ Full RAG ingestion pipeline |
| `/rag/query` | POST | ✅ Cosine similarity retrieval |

---

## Provider History (why we ended up at Azure OpenAI)

1. **Azure OpenAI** — first attempt, API key was invalid, connection failed
2. **Gemini** — worked briefly, then hit 429 RESOURCE_EXHAUSTED (free tier quota = 0)
3. **Ollama** — installed on Windows host, but WSL2 can't reach `localhost:11434` (network isolation)
4. **Azure OpenAI** — got a valid API key, everything works: Kimi-K2.6-1, DeepSeek-V3.2-1, text-embedding-3-small (1536-dim)

---

## Implementation Priority (what to build next)

### Phase 1: Auth (Requirement 6) — foundation for everything else
1. `routers/auth.py` — POST /auth/signup, POST /auth/login
2. `services/auth.py` — password hashing (bcrypt/argon2), JWT creation/validation
3. `dependencies.py` — add `get_current_user` dependency
4. Scope all future routes to authenticated user

### Phase 2: Agent with LangGraph (Requirement 3) — core of the project
1. Create `backend/agent/` directory with graph definition
2. Define 3 tools: `rag_retriever`, `ml_predictor`, `live_conditions_fetcher`
3. Pydantic input validation on every tool
4. Tool allowlist — reject hallucinated tools
5. Cheap model for arg extraction + query rewrite, strong model for synthesis
6. LangSmith tracing integration
7. Write `AgentRun` + `ToolLog` rows to DB on each invocation

### Phase 3: Live Conditions Fetcher (part of Requirement 3)
1. Weather API integration (httpx.AsyncClient, TTL cache ≤10 min)
2. Flights API integration
3. FX/currency API integration
4. Timeout + retry with backoff per INSTRUCTIONS.md

### Phase 4: Webhook Delivery (Requirement 8)
1. Complete `services/send_plan.py` — Resend email delivery
2. Timeout + retry with exponential backoff
3. Structured logging on failure
4. Write `DeliveryLog` rows
5. Fire after agent response, failure must not break user-facing response

### Phase 5: React Frontend (Requirement 7)
1. Chat UI for queries
2. Display agent reasoning: tools fired, inputs/outputs, final plan
3. Stream responses if possible
4. CORS already configured (`ALLOWED_ORIGINS=http://localhost:5173`)

### Phase 6: Tests + CI
1. Tool isolation tests (fake LLM)
2. Pydantic valid/invalid input tests
3. E2E agent test (mock APIs)
4. GitHub Actions CI

### Phase 7: Polish
1. Add frontend service to docker-compose
2. README with architecture diagram, chunking rationale, cost breakdown, LangSmith trace screenshot
3. 3-minute demo video

---

## Environment Configuration

| Key | Value (or description) |
|-----|------------------------|
| `AZURE_OPENAI_KEY` | Azure OpenAI API key (also in Vault) |
| `AZURE_OPENAI_ENDPOINT` | `https://hadymahdy44-0734-week4-resource.services.ai.azure.com/openai/v1` |
| `AZURE_STRONG_MODEL` | `Kimi-K2.6-1` |
| `AZURE_CHEAP_MODEL` | `DeepSeek-V3.2-1` |
| `AZURE_EMBEDDING_MODEL` | `text-embedding-3-small-1` (outputs 1536-dim) |
| `DATABASE_URL` | `postgresql+asyncpg://user:password@localhost:5432/vectordb` |
| `VAULT_ADDR` | `http://localhost:8200` |
| `VAULT_TOKEN` | `myroot` |
| `JWT_SECRET_KEY` | In `.env` + Vault |
| `RESEND_API_KEY` | For email webhook delivery |
| `LANGSMITH_*` | Tracing configured and working |

---

## Known Issues & Gotchas

1. **Alembic migration `003_vector_1536` not yet applied** — run `alembic upgrade head` after Docker services are up
2. **RAG retrieval returns empty list when no documents ingested** — fixed the SQL syntax error (`CAST` instead of `::`), but the route still raises 500 on other errors. Consider catching the empty case gracefully.
3. **`routers/auth.py`, `routers/user.py`, `routers/admin.py`, `routers/llm.py` are empty** — need implementation
4. **`services/send_plan.py` is a stub** — only has a Resend import, no logic
5. **Frontend is default Vite boilerplate** — no chat UI or agent display
6. **No tests** — zero test files exist
7. **No LangGraph/agent code** — the core orchestration is missing
8. **19 RAG documents** — INSTRUCTIONS.md says 20-30. May need a few more or confirmation that 19 is acceptable.
9. **No ML model artifact** — `artifacts/ml_model/random_forest_travel_model.pkl` path exists in code but file hasn't been verified

---

## Commands to Resume Development

```bash
# Start infrastructure
docker compose up -d

# Run Alembic migration (768→1536 for Azure OpenAI embeddings)
cd backend && uv run alembic upgrade head

# Ingest documents into pgvector
curl -X POST http://localhost:8000/rag/ingest

# Test retrieval
curl -X POST http://localhost:8000/rag/query -H "Content-Type: application/json" -d '{"query": "budget-friendly beach in Asia", "k": 5}'

# Test LLM health
curl http://localhost:8000/health/llm

# Test ML prediction
curl -X POST http://localhost:8000/ml/predict -H "Content-Type: application/json" -d '{"active_movement": 0.8, "relaxation": 0.2, "cultural_interest": 0.1, "cost_sensitivity": 0.1, "luxury_preference": 0.9, "family_friendliness": 0.0, "nature_orientation": 0.9, "social_group": 0.1}'

# Run backend directly (without Docker)
cd backend && uv run uvicorn main:app --reload
```
