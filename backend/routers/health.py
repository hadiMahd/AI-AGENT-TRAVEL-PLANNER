"""
Health check endpoints for all system components.

Endpoints:
- GET /health/db  — database connectivity (asyncpg + pgvector)
- GET /health/llm — Azure OpenAI reachability (TTL-cached, checks strong model)
- GET /health     — composite check (API + ML model + LLM) — defined in main.py
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from db.session import check_db_health
from dependencies import get_strong_llm
from llm import check_llm_health
from langchain_openai import ChatOpenAI

router = APIRouter(tags=["Health"])


# ── Response schemas ─────────────────────────────────────────


class DBHealthResponse(BaseModel):
    status: str
    database: str
    detail: str | None = None


class LLMHealthResponse(BaseModel):
    status: str
    llm: str
    detail: str | None = None


# ── Endpoints ─────────────────────────────────────────────────


@router.get(
    "/health/db",
    response_model=DBHealthResponse,
    summary="Check database connectivity",
)
async def db_health() -> DBHealthResponse:
    """
    Check if the Postgres + pgvector database is reachable.

    Executes SELECT 1 via the async engine. No session needed.
    """
    healthy = await check_db_health()
    if healthy:
        return DBHealthResponse(status="ok", database="reachable")
    return DBHealthResponse(
        status="error",
        database="unreachable",
        detail="Could not connect to the database",
    )


@router.get(
    "/health/llm",
    response_model=LLMHealthResponse,
    summary="Check Azure OpenAI LLM reachability",
)
async def llm_health(
    llm: ChatOpenAI = Depends(get_strong_llm),
) -> LLMHealthResponse:
    """
    Check if the Azure OpenAI endpoint is reachable and the API key is valid.

    Uses a TTL-cached result (60s) to avoid spending tokens on every
    health check request. Only the strong model is tested — if the
    API key works for one model, it works for all.

    How it works:
    1. If we've checked within the last 60s, return the cached result
    2. Otherwise, send a minimal prompt ("ok") with a 10s timeout
    3. If the API responds → healthy. If timeout/error → unhealthy.
    4. 429 rate-limit responses are treated as "healthy but throttled"
       — the API IS reachable, just rate-limited.
    """
    healthy = await check_llm_health(llm)
    if healthy:
        return LLMHealthResponse(status="ok", llm="reachable")
    return LLMHealthResponse(
        status="error",
        llm="unreachable",
        detail="Azure OpenAI did not respond within 10 seconds or returned an error",
    )
