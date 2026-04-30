"""
Weather service — fetches current weather via wttr.in.

wttr.in is a free, no-key-needed weather API that supports:
- Location by latitude/longitude: wttr.in/{lat},{long}?format=j1
- JSON response with current conditions + 3-day forecast
- No authentication, no rate limits for reasonable use

Why wttr.in over OpenWeatherMap?
- Zero setup — no API key, no signup, no paid tiers
- Supports lat/long directly — matches our document metadata
- Returns comprehensive JSON data — more than we need, we extract what matters
- Reliable — backed by Open-Meteo, community-maintained since 2016

TTL Cache (10 min):
- Weather changes frequently but not second-by-second
- 10 min is the INSTRUCTIONS.md recommendation: "TTL cache for volatile
  tool responses (e.g., weather ≤10 min)"
- Avoids hammering wttr.in on repeated queries for the same destination

Retries + backoff:
- 3 retries with exponential backoff (1s, 2s, 4s)
- Per INSTRUCTIONS.md: "Timeouts + backoff on all external calls.
  Exhausted retries → structured log + graceful fallback."
- On final failure: returns None — the tool wrapper handles this gracefully
"""

import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

WTTR_BASE_URL = "https://wttr.in"
REQUEST_TIMEOUT = 10.0
MAX_RETRIES = 3
CACHE_TTL = 600  # 10 minutes in seconds

_cache: dict[str, tuple[float, dict[str, Any]]] = {}


async def get_weather_by_city(city: str) -> dict[str, Any] | None:
    """Fetch current weather for a city name (e.g. 'Bali', 'Tokyo')."""
    cache_key = city.lower().strip()
    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < CACHE_TTL:
        return cached[1]

    url = f"{WTTR_BASE_URL}/{city}"
    params = {"format": "j1"}
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            result = _extract_weather(data)
            _cache[cache_key] = (time.monotonic(), result)
            logger.info("Weather fetched for city %s: %.1f°C, %s", city, result["temp_c"], result["condition"])
            return result

        except Exception as exc:
            last_error = exc
            wait = 2 ** (attempt - 1)
            logger.warning("Weather fetch attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, city, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    logger.error("Weather fetch exhausted retries for city %s: %s", city, last_error)
    return None


async def get_weather(lat: float, long: float) -> dict[str, Any] | None:
    """
    Fetch current weather for a location by latitude/longitude.

    Uses wttr.in with lat/long format — more precise than city name
    (e.g. "Bali" is a whole island, lat/long pins the exact spot).

    Args:
        lat:   Latitude (e.g. -8.3405 for Bali).
        long:  Longitude (e.g. 115.092 for Bali).

    Returns:
        Dict with weather data, or None if all retries fail.
        {
            "temp_c": 28,
            "feels_like_c": 30,
            "condition": "Partly cloudy",
            "humidity": 75,
            "wind_kph": 12,
            "location": "Bali"
        }
    """
    cache_key = f"{lat:.2f},{long:.2f}"

    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < CACHE_TTL:
        logger.debug("Returning cached weather for %s", cache_key)
        return cached[1]

    url = f"{WTTR_BASE_URL}/{lat},{long}"
    params = {"format": "j1"}

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            result = _extract_weather(data)
            _cache[cache_key] = (time.monotonic(), result)
            logger.info("Weather fetched for %s: %.1f°C, %s", cache_key, result["temp_c"], result["condition"])
            return result

        except Exception as exc:
            last_error = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Weather fetch attempt %d/%d failed: %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    logger.error("Weather fetch exhausted retries for %s: %s", cache_key, last_error)
    return None


def _extract_weather(data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the fields we care about from wttr.in's verbose JSON response.

    wttr.in returns a large JSON with current_condition, weather (forecast),
    nearest_area, etc. We only need the current conditions.
    """
    current = data.get("current_condition", [{}])[0]
    area = data.get("nearest_area", [{}])[0]

    return {
        "temp_c": int(current.get("temp_C", 0)),
        "feels_like_c": int(current.get("FeelsLikeC", 0)),
        "condition": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
        "humidity": int(current.get("humidity", 0)),
        "wind_kph": float(current.get("windspeedKmph", 0)),
        "location": area.get("areaName", [{}])[0].get("value", "Unknown"),
    }
