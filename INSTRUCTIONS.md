# CLAUDE.md - Smart Travel Planner (Week 4)

## 🎯 Project Status & Scope
- **ML Component:** Pre-trained, saved in `artifacts/` via `joblib`. No training, CV, or dataset collection required.
- **Focus:** Integration of saved model as an agent tool, RAG pipeline, LangGraph orchestration, async engineering, persistence, auth, frontend, webhook delivery, and Dockerization.
- **Goal:** Ship a production-grade agent system where AI functionality and engineering rigor are equally prioritized.

---

## 📦 Core Requirements (9 Mandatory)
1. **ML Tool Integration**
   - Load `joblib` model once at startup via FastAPI lifespan.
   - Expose via `Depends()`. Zero globals. Zero per-request loading.
   - Wrap in an async agent tool with Pydantic input validation. Return structured prediction + confidence.
2. **RAG Tool**
   - Retrieve over 20–30 destination docs (10–15 destinations).
   - Store embeddings in Postgres via `pgvector` (same DB as app).
   - Justify chunk size, overlap, and retrieval strategy in README.
   - Test manually on hand-written queries before agent integration.
3. **Agent With Three Tools**
   - Framework: LangGraph or LangChain.
   - Tools: RAG retriever, ML classifier predictor, live conditions fetcher (weather/flights/FX).
   - Pydantic validation on every tool input. Invalid → structured error → LLM retry. Never crash.
   - Explicit tool allowlist. Reject hallucinated tools.
   - Genuine synthesis across tools. Resolve conflicts (e.g., RAG vs live API). No concatenation.
   - Trace end-to-end with LangSmith (free tier).
4. **Two Models, One Agent**
   - Cheap model (`gpt-4o-mini`/Haiku-clascd /home/hadym/AIE-BOOTCAMP/Week4/project/backend
uv run uvicorn main:app --reloads) for mechanical tasks: arg extraction, RAG query rewrite.
   - Strong model for final synthesis.
   - Log token usage per step. Report real cost per query in README.
5. **Persistence — Postgres + pgvector + SQLAlchemy**
   - Single DB for users, agent runs, tool logs, embeddings.
   - SQLAlchemy 2.x async. Track: who asked, what they asked, agent answer, tools fired, timestamps.
   - Migrations via Alembic.
6. **Auth**
   - Sign-up/login with password hashing (JWT or sessions).
   - Scope all runs, history, webhook targets to authenticated user.
7. **React Frontend**
   - Vite + React. Chat UI for queries.
   - Display agent reasoning: tools fired, inputs/outputs, final plan.
   - Stream responses if possible.
8. **Webhook Delivery**
   - Fire plan to real channel (Discord/Slack/Sheets/Email).
   - Timeout + retry with backoff. Structured logging on failure.
   - Webhook failure must not break user-facing response.
9. **Docker**
   - `docker-compose.yml` for backend, frontend, Postgres.
   - Named volume for Postgres. `docker compose up` works standalone.

---

## 🛠️ Engineering Standards (Non-Negotiable)
- **Async All the Way:** `async def` routes/tools, SQLAlchemy async sessions, `httpx.AsyncClient`, async LLM SDKs. Zero `time.sleep` or `requests`.
- **Dependency Injection:** `Depends()` for LLM clients, DB sessions, ML model, vector store, agent executor, current user. Override in tests.
- **Singletons via Lifespan:** Init DB engine, ML model, vector store, LLM client once. Dispose on shutdown. Expose via `Depends()`.
- **Caching:** `@lru_cache` for config/model paths. TTL cache for volatile tool responses (e.g., weather ≤10 min). Document where/why.
- **Configuration:** Single `pydantic-settings` class. Typed/validated env vars at startup. No `os.getenv` or magic strings.
- **Type Hints & Pydantic Boundaries:** 100% typed. Validate HTTP bodies, tool inputs, LLM outputs, webhooks at the edge only.
- **Errors & Retries:** Timeouts + backoff on all external calls. Exhausted retries → structured log + graceful fallback. Tool errors returned to LLM, not raised.
- **Code Hygiene:** Modular layout. JSON logging. No `print()`. Linters/formatters + pre-commit. `.env.example` present.
- **Tests:** Tool isolation (fake LLM), Pydantic valid/invalid tests, E2E agent test (mock APIs). Run via GitHub Actions CI.

---

## 📐 Architecture & Integration Rules
- Separate concerns: routes, services, models, tools, agent code. Split prompts by purpose.
- Keep files small. If a file's job can't be described in one sentence, split it.
- Meaningful names everywhere. Naming is documentation.
- Use `uv` for environment management and lockfiles.
- Networks enable container communication by service name. Volumes persist data across restarts.
- Never hardcode environment-specific URLs in `docker-compose.yml`. Use `${VAR:-fallback}`.
- CORS middleware correctly configured for frontend-backend communication.
- Real logging setup: levels, formatters, persistent output. stdout-only logs vanish on restart.

---

## 🧠 RAG & Agent Orchestration
- **Ingestion Frequency:** Clarify what runs once (startup/ingest) vs per request. Do not re-ingest on every query.
- **Vector Store:** Postgres + `pgvector`. Know schema, metadata, and index type (HNSW/IVFFlat). Filesystem DB ≠ server DB.
- **Tool Execution:** Explicitly state if tools run parallel, sequential, or routed. Document the graph flow.
- **Retrieval Pipeline:** query → embed → similarity search → top-k → prompt assembly → generate. State `k` and justify it.
- **Synthesis:** Strong LLM must compare tool outputs, explain tension if conflicting, and output a coherent plan. JSON structure recommended.

---

## 📝 Deliverables
- GitHub repo with full codebase.
- README: self-drawn architecture diagram, chunking/retrieval rationale, per-query cost breakdown, LangSmith multi-tool trace screenshot, optional extensions list.
- 3-minute demo video: React UI query → tool execution → webhook delivery.

---

## 🎯 Evaluation & Defense Checklist
- System works end-to-end. AI parts show real synthesis, traceable runs, proper validation.
- Engineering is real: async, DI, lifespan singletons, typed boundaries, no globals, no blocking I/O.
- You must defend every choice: why cache here, why `Depends` not global, why chunk size X, why those tools, why this retry strategy.
- "The tutorial said so" fails. "I tried X, it failed because Y, so I did Z" passes.
- Be ready for two demo modes: free-form (follow data flow) and instructor-led (jump to specific files/functions).
- Every line, flag, library, or folder structure is a potential question. Know your codebase cold.