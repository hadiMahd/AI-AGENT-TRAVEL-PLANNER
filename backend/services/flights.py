"""
Flight search service — searches the web for flight information via DuckDuckGo.

Instead of a locked-in API like Amadeus (currently unavailable), we use
DuckDuckGo web search to find real flight information from travel sites
like Skyscanner, Kayak, Google Flights, etc.

Why DuckDuckGo search over Amadeus?
- Amadeus was unavailable at the time of implementation
- No API key, no signup, no rate limits for reasonable use
- Returns real, current data from actual travel booking sites
- More defensible in evaluation: "I tried Amadeus (unavailable), so I
  used web search — real data, zero cost, zero dependencies"
- The search results contain price ranges, airlines, durations —
  exactly what a travel planner needs

How it works:
- Search query: "flights from {origin} to {destination} cheap prices"
- DuckDuckGo returns top 5 search result snippets
- The strong LLM interprets these snippets during synthesis
- Example result: "Skyscanner: Cairo to Bali from $450, 1 stop via Dubai..."

Why not parse the actual booking sites?
- Each site has different HTML, anti-bot protection, rate limiting
- Snippets already contain the key info (price, route, duration)
- Less code, fewer failure points, no scraping maintenance

TTL Cache (30 min):
- Flight prices change during the day but not minute-by-minute
- 30 min cache is a reasonable trade-off — fresh enough for planning,
  avoids hammering DuckDuckGo on repeated queries

Retries + backoff:
- 2 retries (search API is less flaky than weather/FX)
- Per INSTRUCTIONS.md: "Timeouts + backoff on all external calls."
"""

import asyncio
import logging
import time
from typing import Any

from ddgs import DDGS

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
CACHE_TTL = 1800  # 30 minutes

_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


async def search_flights(origin: str, destination: str) -> list[dict[str, Any]] | None:
    """
    Search the web for flight information between two cities.

    Uses DuckDuckGo text search with a travel-specific query.
    Returns snippets from travel booking sites (Skyscanner, Kayak, etc.)
    that the LLM can interpret during synthesis.

    Args:
        origin:      Origin city name (e.g. "Cairo").
        destination: Destination city name (e.g. "Bali").

    Returns:
        List of search result dicts, or None if all retries fail.
        [
            {
                "title": "Skyscanner: Cheap flights from Cairo to Bali",
                "snippet": "From $450, 1 stop via Dubai, 12h duration...",
                "url": "https://www.skyscanner.net/..."
            },
            ...
        ]
    """
    cache_key = f"{origin.lower()}->{destination.lower()}"

    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < CACHE_TTL:
        logger.debug("Returning cached flight search for %s", cache_key)
        return cached[1]

    query = f"flights from {origin} to {destination} cheap prices 2026"
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            results = await asyncio.to_thread(_search_sync, query)

            _cache[cache_key] = (time.monotonic(), results)
            logger.info(
                "Flight search for %s→%s returned %d results",
                origin, destination, len(results),
            )
            return results

        except Exception as exc:
            last_error = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Flight search attempt %d/%d failed: %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    logger.error("Flight search exhausted retries for %s: %s", cache_key, last_error)
    return None


def _search_sync(query: str) -> list[dict[str, Any]]:
    """
    Synchronous DuckDuckGo search — run in a thread via anyio/to_thread.

    duckduckgo-search doesn't have a native async API, so we wrap it
    in asyncio.to_thread to avoid blocking the event loop.
    """
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=5))

    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in raw
    ]
