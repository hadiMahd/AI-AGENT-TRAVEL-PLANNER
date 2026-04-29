"""
FastAPI application entry point.

Lifespan startup initializes all singletons:
- DB engine (asyncpg + pgvector)
- ML model (joblib scikit-learn artifact)
- Strong LLM client (Azure OpenAI — e.g. Kimi-K2.6-1 — for agent synthesis)
- Cheap LLM client (Azure OpenAI — e.g. DeepSeek-V3.2-1 — for mechanical tasks)
- Embedder client (Azure OpenAI text-embedding-3-small — for RAG)

Lifespan shutdown disposes the DB engine.
Azure OpenAI clients are garbage collected — no explicit close needed.

Per INSTRUCTIONS.md: "Singletons via Lifespan — init once, expose via Depends()"
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import anyio
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from db.session import dispose_db, init_db
from llm import create_cheap_llm, create_embedder, create_strong_llm
from routers.auth import router as auth_router
from routers.health import router as health_router
from routers.ml_model import router as ml_model_router
from routers.rag import router as rag_router
from routers.user import router as user_router
from services.ml_inference import ModelNotAvailableError, get_model_path


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── Startup: initialize all singletons ──────────────────

    # 1. Database engine
    await init_db()

    # 2. ML model (scikit-learn artifact loaded via joblib)
    try:
        model_path = get_model_path()
        app.state.ml_model = await anyio.to_thread.run_sync(joblib.load, model_path)
        app.state.ml_model_error = None
    except Exception as exc:  # noqa: BLE001 - surface error at request time
        app.state.ml_model = None
        app.state.ml_model_error = ModelNotAvailableError(str(exc))

    # 3. Azure OpenAI LLM clients — created once, reused for all requests
    #    If the endpoint is unreachable, the client is still created
    #    but will fail at request time with a clear error message.
    try:
        app.state.strong_llm = create_strong_llm()
        app.state.cheap_llm = create_cheap_llm()
        app.state.embedder = create_embedder()
        app.state.llm_error = None
    except Exception as exc:  # noqa: BLE001
        app.state.strong_llm = None
        app.state.cheap_llm = None
        app.state.embedder = None
        app.state.llm_error = str(exc)

    yield

    # ── Shutdown: release resources ─────────────────────────
    await dispose_db()
    # Azure OpenAI clients don't need explicit disposal —
    # they're garbage collected when app.state goes out of scope


app = FastAPI(
    title="AI Travel Planner API",
    description="FastAPI backend for travel-style prediction and trip planning tools.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(ml_model_router)
app.include_router(rag_router)
app.include_router(auth_router)
app.include_router(user_router)

# CORS middleware — allows the Vite frontend (localhost:5173) to call the API.
# Why not allow all origins ("*")?
# - Security: restricting to our frontend URL prevents CSRF from other sites
# - INSTRUCTIONS.md: "CORS middleware correctly configured for frontend-backend communication"
# - The allowed origin comes from Settings (ALLOWED_ORIGINS in .env)
settings = get_settings()
origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
if "http://localhost:5173" in origins:
    origins.append("http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Composite health check — API, ML model, LLM clients."""
    model_loaded = getattr(app.state, "ml_model", None) is not None
    model_error = getattr(app.state, "ml_model_error", None)
    ml_status = "loaded" if model_loaded else ("error" if model_error else "not_loaded")

    llm_ok = getattr(app.state, "strong_llm", None) is not None
    llm_error = getattr(app.state, "llm_error", None)
    llm_status = "initialized" if llm_ok else ("error" if llm_error else "not_initialized")

    all_ok = model_loaded and llm_ok
    overall = "ok" if all_ok else "degraded"

    return {"status": overall, "api": "running", "ml_model": ml_status, "llm": llm_status}


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
