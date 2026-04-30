"""
LLM client management — creates, caches, and health-checks Azure OpenAI clients.

Three clients initialized once during FastAPI lifespan and stored on app.state:
- strong_llm: for final agent synthesis (e.g. Kimi-K2.6-1)
- cheap_llm:  for mechanical tasks like arg extraction and RAG query rewrite
              (e.g. DeepSeek-V3.2-1)
- embedder:   for RAG embedding pipeline (text-embedding-3-small, 1536-dim output)

Why lifespan singletons instead of module-level globals?
- INSTRUCTIONS.md mandates: "Init LLM client once via lifespan. Dispose on
  shutdown. Expose via Depends()"
- Testable: app.dependency_overrides can swap any client for a fake
- Consistent: follows the same pattern as the ML model (app.state.ml_model)
- Clean shutdown: lifespan exit ensures resources are released

Why langchain-openai ChatOpenAI (v1 API) instead of AzureChatOpenAI?
- Azure's v1 API allows using ChatOpenAI directly with base_url + api_key
- No need for api_version or azure_deployment parameters
- Simpler configuration: just base_url, model, api_key
- LangChain docs recommend this approach for v1 API endpoints
- Same async interface (ainvoke, aembed_documents, aembed_query)
- LangSmith tracing works out of the box (LangChain callbacks)

Cache invalidation for the health check:
- LLM clients themselves don't cache data — they're stateless wrappers
  around HTTP connections. There's nothing to "invalidate" in the client.
- The real caching concern is the HEALTH CHECK result. We TTL-cache it
  for 60 seconds to avoid spending tokens on every /health/llm hit.
  This is a reasonable trade-off: we detect outages within 1 minute
  without burning API quota.
- If settings change (e.g., API key rotated in Vault), the server must
  be restarted. This is standard for production deployments — config
  changes shouldn't happen at runtime.

Two Models, One Agent (per INSTRUCTIONS.md):
- Cheap model for mechanical tasks: argument extraction, RAG query rewrite
- Strong model for final synthesis: combining tool outputs into a plan
- This saves cost — most work is cheap, only the final answer uses the
  expensive model
"""

import asyncio
import logging
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# ── TTL-cached health check state ──────────────────────────────
# Why TTL? Each health check costs ~1 token (ainvoke("ok")).
# At 60s TTL, that's ~1 token/minute max. Without TTL, a monitoring
# service hitting /health/llm every 5s would burn 12 tokens/min.
_LAST_HEALTH_CHECK: dict[str, float | bool | None] = {
    "timestamp": 0.0,
    "healthy": None,
}
_HEALTH_CACHE_TTL = 60  # seconds — reasonable for dev and production


# ── Client factory functions ──────────────────────────────────
# Called once during lifespan startup. Each creates a fully configured
# LangChain client ready for async use.


def create_strong_llm(settings: Settings | None = None) -> ChatOpenAI:
    """
    Create the strong Azure OpenAI model client for agent synthesis.

    Used for the final step of the agent pipeline: combining tool outputs
    into a coherent, well-reasoned travel plan. This is the expensive model
    that justifies its cost with higher-quality output.

    Uses the v1 API — ChatOpenAI with base_url pointing to the Azure endpoint.

    Args:
        settings: Optional Settings override (useful for testing).
                  Defaults to the cached get_settings() singleton.

    Returns:
        A configured ChatOpenAI instance pointing to Azure.
    """
    if settings is None:
        settings = get_settings()

    logger.info(
        "Creating strong LLM client — model=%s, endpoint=%s",
        settings.azure_strong_model,
        settings.azure_openai_endpoint,
    )

    return ChatOpenAI(
        model=settings.azure_strong_model,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        request_timeout=120,
    )


def create_cheap_llm(settings: Settings | None = None) -> ChatOpenAI:
    """
    Create the cheap Azure OpenAI model client for mechanical tasks.

    Used for:
    - Argument extraction from user queries
    - RAG query rewriting (optimizing the search query)
    - Tool input validation / formatting

    These tasks don't need creative output — they need fast, cheap,
    structured responses. DeepSeek-V3.2-1 is ideal.

    Args:
        settings: Optional Settings override (useful for testing).

    Returns:
        A configured ChatOpenAI instance pointing to Azure.
    """
    if settings is None:
        settings = get_settings()

    logger.info(
        "Creating cheap LLM client — model=%s, endpoint=%s",
        settings.azure_cheap_model,
        settings.azure_openai_endpoint,
    )

    return ChatOpenAI(
        model=settings.azure_cheap_model,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        request_timeout=120,
    )


def create_embedder(settings: Settings | None = None) -> OpenAIEmbeddings:
    """
    Create the Azure OpenAI embedding client for the RAG pipeline.

    Uses text-embedding-3-small which outputs 1536-dim vectors that match
    the vector(1536) column in the documents table. Used during ingestion
    (batch embed) and retrieval (query embed).

    Note: text-embedding-3-small supports a `dimensions` parameter to
    shorten embeddings (e.g. dimensions=768 would avoid a DB migration),
    but we use the default 1536-dim output for maximum retrieval quality.

    Args:
        settings: Optional Settings override (useful for testing).

    Returns:
        A configured OpenAIEmbeddings instance pointing to Azure.
    """
    if settings is None:
        settings = get_settings()

    logger.info(
        "Creating embedder client — model=%s, endpoint=%s",
        settings.azure_embedding_model,
        settings.azure_openai_endpoint,
    )

    return OpenAIEmbeddings(
        model=settings.azure_embedding_model,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
    )


# ── Health check ───────────────────────────────────────────────


async def check_llm_health(llm: ChatOpenAI) -> bool:
    """
    Check if the Azure OpenAI endpoint is reachable and the API key is valid.

    Strategy:
    1. If we've checked within the last 60 seconds, return the cached result
       (avoids spending tokens on every /health/llm request)
    2. Otherwise, send a minimal prompt ("ok") with a 10-second timeout
    3. If the API responds, it's healthy. If timeout or error, it's not.

    Why only check the strong model?
    - Both models use the same API key and endpoint
    - If the key works for the strong model, it works for the cheap model too
    - Testing both would double the cost for no additional signal

    Why TTL cache?
    - Health endpoints are often hit every 5-10 seconds by monitoring
    - Without caching: ~6-12 tokens/minute just on health checks
    - With 60s TTL: ~1 token/minute max — acceptable overhead

    429 handling:
    - 429 RateLimitError means the API IS reachable and the key IS valid
    - We're just rate-limited. This is "healthy but throttled", not "down".
    - Per INSTRUCTIONS.md: "Timeouts + backoff on all external calls.
      Exhausted retries → structured log + graceful fallback."

    Args:
        llm: The strong ChatOpenAI client to test.

    Returns:
        True if the Azure OpenAI endpoint is reachable, False otherwise.
    """
    now = time.monotonic()
    cached_healthy = _LAST_HEALTH_CHECK.get("healthy")
    cached_ts = _LAST_HEALTH_CHECK.get("timestamp", 0.0)

    # Return cached result if still within TTL
    if cached_healthy is not None and (now - cached_ts) < _HEALTH_CACHE_TTL:
        logger.debug("Returning cached LLM health result: %s", cached_healthy)
        return cached_healthy  # type: ignore[return-value]

    # Perform a real API call to verify connectivity
    try:
        # Minimal prompt — the model should respond with a single token
        # 10-second timeout — if the API is slow to respond, it's degraded
        await asyncio.wait_for(
            llm.ainvoke("ok"),
            timeout=10.0,
        )
        healthy = True
        logger.info("LLM health check passed")
    except asyncio.TimeoutError:
        healthy = False
        logger.warning("LLM health check timed out (10s)")
    except Exception as exc:
        # 429 RateLimitError means the API IS reachable and the key IS valid
        # — we're just rate-limited. This is "healthy but throttled", not "down".
        exc_str = str(exc)
        if "429" in exc_str or "rate_limit" in exc_str.lower() or "quota" in exc_str.lower():
            healthy = True
            logger.info("LLM health check passed (rate-limited but reachable)")
        else:
            healthy = False
            logger.warning("LLM health check failed: %s", exc)

    # Update the TTL cache
    _LAST_HEALTH_CHECK["timestamp"] = now
    _LAST_HEALTH_CHECK["healthy"] = healthy

    return healthy
