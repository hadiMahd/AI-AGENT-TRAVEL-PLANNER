# Smart Travel Planner — AI Agent System

An AI-powered travel planning agent that plans entire trips from a single chat message:
destination research, weather, flights, currency exchange, and personalized travel style
prediction — all synthesized into a coherent, real-time plan.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                      │
│  Chat UI │ Tool Log Panel │ Auth │ Email Prompt │ SSE Streaming    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /agent/chat (SSE)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FASTAPI LIFESPAN                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ ML Model │  │Strong LLM │  │Cheap LLM │  │    Embedder      │  │
│  │(joblib)  │  │(Kimi 2.6) │  │ (o4-mini)│  │(text-embed-3-sm) │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       └──────────────┴─────────────┴──────────────────┘            │
│                          │ app.state (Depends)                      │
└──────────────────────────┼──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH AGENT (StateGraph)                     │
│                                                                     │
│   ┌───────────────┐                                                 │
│   │ classify_intent│── casual ──►┌─────────────┐──► END            │
│   │  (cheap LLM)   │             │ casual_reply │                   │
│   └───────┬───────┘             └─────────────┘                   │
│           │ travel                                                  │
│           ▼                                                         │
│   ┌───────────────┐     needs_input                                │
│   │ check_context  │──────────────► END (ask user)                  │
│   │(cheap LLM+RAG) │                                                │
│   └───────┬───────┘                                                │
│           │ has both dest + origin                                   │
│           ▼                                                         │
│   ┌───────────────────┐                                             │
│   │ route_and_run_tools│── 5 tools in parallel ──┐                  │
│   │   (cheap LLM)      │                         │                  │
│   └───────────────────┘                         │                  │
│           │                                      ▼                  │
│           │              ┌──────────────────────────────────────┐  │
│           │              │  rag_retriever  │  weather_fetcher   │  │
│           │              │  ml_predictor   │  flight_searcher   │  │
│           │              │  fx_checker                         │  │
│           │              └──────────────────────────────────────┘  │
│           ▼                                                         │
│   ┌───────────────┐                                                 │
│   │   synthesize   │── streaming SSE tokens ──► END                 │
│   │ (strong LLM)   │                                                │
│   └───────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  ┌──────────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐ │
│  │  PostgreSQL   │  │ Azure     │  │ External │  │   External    │ │
│  │  + pgvector   │  │ OpenAI    │  │  APIs    │  │   Delivery    │ │
│  │  (HNSW idx)   │  │ (v1 API)  │  │ wttr.in  │  │   Resend      │ │
│  │               │  │           │  │ frankfurter│ │   (Email)     │ │
│  └──────────────┘  └───────────┘  └──────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Flow per query:** `classify_intent` → `check_context` (RAG destination detection + origin check) → `route_and_run_tools` (parallel tool execution) → `synthesize` (streaming strong LLM)

---

## Dataset Labeling Rules

The knowledge base consists of **19 destination documents** stored in `documents/` as JSON files.
Each document has hand-assigned metadata and a natural-language content body.

### Metadata Fields

| Field | Values | Purpose |
|-------|--------|---------|
| `country` | "Indonesia", "Japan", "Albania", etc. | Country name — used for currency lookup, destination detection |
| `continent` | "Asia", "Europe", "North America", "Africa", "South America" | Geographic region |
| `style` | "budget", "relaxation", "luxury", "adventure", "culture", "family" | Travel style classification — one of 6 labels |
| `latitude` | -8.3405, 35.6762 | For weather API lookup |
| `longitude` | 115.092, 139.650 | For weather API lookup |
| `source_file` | Injected at load time (e.g., `"albania_budget.json"`) | Traceability |
| `chunk_index` | Injected at chunk time (almost always 0) | Chunk tracking |

### Destination Distribution by Style

| Style | Destinations |
|-------|-------------|
| **Budget** | Albania, Thailand, Vietnam |
| **Relaxation** | Bali (Indonesia), Japan, Maldives |
| **Luxury** | Dubai (UAE), Seychelles, Switzerland |
| **Adventure** | Costa Rica, Iceland, Nepal, Peru |
| **Culture** | Egypt, India, Italy |
| **Family** | Canada, Morocco, Portugal |

### Content Format

Each document's `text` field is a 2–4 sentence natural language paragraph with style-relevant
details, price signals, and activity descriptions. Example:

> "Albania is a budget-friendly destination in Europe, with relatively low prices for
> accommodation and meals. Coastal towns and the Albanian Riviera attract visitors looking
> for Mediterranean-style beaches at a fraction of Western-Europe costs."

---

## Chunking & Retrieval Rationale

### Chunking Strategy

| Parameter | Value | Why |
|-----------|-------|-----|
| **Chunk size** | 500 characters | Each destination doc is ~300–400 characters, so nearly all fit in a single chunk — preserving semantic integrity per destination. |
| **Chunk overlap** | 50 characters | Provides continuity for the rare doc exceeding 500 characters. Minimal since splitting is uncommon. |
| **Splitter** | LangChain `RecursiveCharacterTextSplitter` | Splits on natural boundaries: `\n\n` → `\n` → `. ` → ` ` → ``. Respects paragraph, sentence, and word boundaries. |
| **Result** | ~20–25 chunks from 19 documents | Nearly 1:1 document-to-chunk mapping. |

### Retrieval Strategy

| Parameter | Value | Why |
|-----------|-------|-----|
| **k (default)** | 5 | Corpus is ~20–25 chunks. k=5 returns ~25% — enough coverage to find relevant destinations without drowning in noise. |
| **k (agent)** | 3 | Smaller k inside the agent to keep prompt context tight. |
| **Score threshold** | 0.6 | Results below 0.6 cosine similarity are filtered out. Prevents low-confidence noise from polluting the synthesis prompt. |
| **Index** | HNSW (pgvector) | Best for small-to-medium datasets (<1M vectors). Trades minor accuracy for significant speed improvement over exact search. |
| **Distance** | Cosine (`<=>` operator) | Converted to similarity: `1 - (embedding <=> query)`. 1.0 = identical, 0.0 = orthogonal. |
| **Embeddings** | Azure OpenAI `text-embedding-3-small-1` | 1536-dim output, matching the `vector(1536)` column. Native dimensionality for maximum retrieval quality. |
| **Embed batch size** | 16 per API call | Conservative to stay within Azure OpenAI TPM/RPM rate limits. |

### Ingestion Pipeline (manual trigger via `POST /rag/ingest`)

```
LOAD (19 JSON files) → CHUNK (RecursiveCharacterTextSplitter) → EMBED (batch=16) → INSERT (full wipe-and-replace)
```

Idempotent — full wipe-and-replace ensures no duplicates or stale data. Not automatic on startup.

---

## Model Comparison Table

| | **Strong Model** | **Cheap Model** | **Embedding Model** |
|---|---|---|---|
| **Name** | Kimi-K2.6-1 | o4-mini | text-embedding-3-small-1 |
| **Provider** | Azure OpenAI v1 API | Azure OpenAI v1 API | Azure OpenAI v1 API |
| **Role** | Final plan synthesis — coherent, cross-tool reasoning | Mechanical tasks: intent classification, context checking, tool routing, casual replies | RAG pipeline: document ingestion + query embedding |
| **Why chosen** | High-quality output justifies cost for synthesis where quality matters most | Fast, cheap, structured JSON for deterministic routing/classification tasks | 1536-dim output matches pgvector column; good retrieval quality |
| **Nodes using it** | `synthesize` (streaming, in routers/agent.py) | `classify_intent`, `casual_reply`, `check_context`, `route_and_run_tools` | `check_context` (destination detection), `rag_retriever` tool, ingestion |
| **Timeouts** | 90s (streaming, no hard cutoff) | 60s (classify/casual), 90s (check/route) | N/A |
| **Request timeout** | 120s | 120s | N/A |
| **API client** | `ChatOpenAI(base_url=..., model=...)` | Same | `OpenAIEmbeddings(base_url=..., model=...)` |

**Design philosophy:** The cheap model handles the majority of work at minimal cost.
Only the final synthesis — where quality directly impacts user experience — uses the expensive model.

---

## Per-Query Cost Breakdown

### Scenario 1: Casual Query ("hi", "thanks")

| Step | Model | Calls |
|------|-------|-------|
| classify_intent | Cheap | 1 × `ainvoke` |
| casual_reply | Cheap | 1 × `ainvoke` |
| **Total** | | **2 cheap, 0 strong** |

### Scenario 2: Full Travel Plan (origin known, e.g., "plan a trip to Japan from Lebanon")

| Step | Model | Calls |
|------|-------|-------|
| classify_intent | Cheap | 1 × `ainvoke` (60s timeout) |
| check_context | Cheap | 1 × `ainvoke` (90s timeout) + 1 embedding call |
| route_and_run_tools | Cheap | 1 × `ainvoke` (90s timeout) + 1 embedding call (RAG) |
| synthesize | **Strong** | 1 × `astream` (streaming) |
| **Total** | | **3 cheap + 1 strong** |

### Scenario 3: First Turn Needs Origin, Then Answered (2 turns)

| Turn | Calls |
|------|-------|
| Turn 1: "plan a trip to Maldives" → asks "where from?" | 1 cheap (classify) + 1 cheap (check_context) |
| Turn 2: "Lebanon" → full plan | 1 cheap (classify) + 1 cheap (check_context) + 1 cheap (route) + 1 strong (synthesis) |
| **Total across 2 turns** | **5 cheap + 1 strong** |

### Tool Execution & Cost Notes

- **All 5 tools run in parallel** via `asyncio.gather` — no sequential bottleneck
- Tool API calls (wttr.in, frankfurter.app, DuckDuckGo) are **free**
- Embedding calls (text-embedding-3-small): ~2 per full plan — negligible cost
- Health check TTL-cached for 60s — max 1 token/min for monitoring
- Token counts (prompt + completion) tracked per step and persisted in `agent_runs` table
- `cost_usd` field exists in schema (Decimal(10,6)) — ready for cost calculation

---

## LangSmith Trace Screenshot

> **TODO:** Insert LangSmith trace screenshot showing a full multi-tool agent run.
> The trace should show all nodes (`classify_intent`, `check_context`, `route_and_run_tools`,
> `synthesize`) with token counts, tool execution latencies, and the final SSE streaming output.

---

## Optional Extensions Completed

| Extension | Status | Details |
|-----------|--------|---------|
| **Email delivery (Resend)** | Complete | Converts Markdown to styled HTML, sends via Resend API. Triggered from frontend after `ask_email` SSE event. `POST /agent/send-email`. |
| **HashiCorp Vault integration** | Complete | Secrets (JWT_SECRET_KEY, AZURE_OPENAI_KEY) fetched from Vault KV v2 at startup. Falls back to `.env` if Vault unreachable. Custom `HashiCorpVaultSource` settings source. |
| **SSE streaming** | Complete | 7 event types: `thinking`, `tool_start`, `tool_result`, `needs_input`, `token`, `final`, `ask_email`. Strong LLM streams token-by-token for responsive UI. |
| **Tool allowlist with hallucination guard** | Complete | Only 5 registered tools can execute. Any tool name from the LLM not in `ALLOWED_TOOLS` is rejected before execution. |
| **Clarification loop guard** | Complete | After 3 failed attempts to get origin from user, proceeds with planning anyway — prevents infinite loops. |
| **API response caching** | Complete | Per-service TTL: weather 10min, flights 30min, FX 60min. In-memory `@lru_cache` for config + TTL dict for tool responses. |
| **Delivery logs audit trail** | Complete | `delivery_logs` table tracks every email: recipient, status, error, provider message ID, latency. |
| **pgAdmin database UI** | Complete | Included in `docker-compose.yml` on port 8080. |
| **3-tier destination priority** | Complete | LLM-determined > prior state > embedding search. Prevents origin-like replies ("lebanon") from hijacking destination ("Maldives"). |
| **RAG score threshold (0.6)** | Complete | Low-confidence vector matches are filtered out before reaching the synthesis prompt. |
| **Config from Vault + env fallback** | Complete | Hybrid config: init kwargs > OS env > HashiCorp Vault > .env > file secrets. Typed `pydantic-settings`. |
| **Async all the way** | Complete | `async def` throughout. No `time.sleep` or blocking I/O. `httpx.AsyncClient`, async SQLAlchemy, async LLM SDKs. |
| **Dependency injection** | Complete | All singletons via `Depends()`. Overridable in tests. Zero module-level globals. |
| **LangSmith tracing** | Configured | End-to-end tracing enabled for all LangChain/LangGraph operations. |

---

## Project Structure

```
project/
├── backend/
│   ├── agent/           # LangGraph nodes + prompts + state
│   │   ├── graph.py     # 4-node agent graph
│   │   ├── prompts.py   # 5 prompt templates
│   │   └── state.py     # AgentState TypedDict
│   ├── alembic/         # DB migrations (3 versions)
│   ├── db/              # Async session factory
│   ├── llm/             # LLM + embedder client factories
│   ├── models/          # SQLAlchemy ORM + Pydantic schemas
│   ├── rag/             # Chunker, embedder, retriever, ingestion
│   ├── routers/         # FastAPI route handlers (6 routers)
│   ├── services/        # Weather, flights, FX, ML inference, auth, email
│   ├── tools/           # 5 agent tools with allowlist
│   ├── config.py        # Single pydantic-settings class
│   ├── dependencies.py  # Depends() providers
│   └── main.py          # FastAPI app + lifespan
├── frontend/            # React + Vite (Chat UI, auth, tool logs)
├── documents/           # 19 destination JSON files
├── artifacts/           # ML model (joblib)
├── docker-compose.yml   # Postgres, pgAdmin, backend, frontend
├── db-schema.sql        # Full database schema
└── .env.example         # Config template
```

---

## Quick Start

```bash
# 1. Start services
docker compose up -d

# 2. Run migrations
cd backend && alembic upgrade head

# 3. Ingest documents
curl -X POST http://localhost:8000/rag/ingest

# 4. Open frontend
open http://localhost:5173
```

---

*Built with: FastAPI, LangGraph, LangChain, Azure OpenAI, PostgreSQL + pgvector, SQLAlchemy 2.x async, React + Vite, Docker*
