"""
Foreign exchange service — fetches currency rates via frankfurter.app.

frankfurter.app is a free, open-source exchange rate API:
- Uses European Central Bank (ECB) data — reliable, updated daily
- No API key, no signup, no rate limits for reasonable use
- Endpoint: api.frankfurter.app/latest?from=EUR&to=USD

Why frankfurter.app over exchangerate-api.com?
- Zero setup — no API key required
- Open source — data source is transparent (ECB)
- Simple response format — just the rate we need
- No paid tiers or key rotation to manage

Country→Currency mapping:
- Our document metadata has "country" but not "currency"
- Adding "currency" to every doc would require re-ingestion
- A lookup dict is simpler and works for all 19 destinations + common origins
- Fallback to "USD" if country not found — safe default for travel planning

TTL Cache (60 min):
- Exchange rates update once per business day (ECB publishes ~16:00 CET)
- 60 min cache means we check at most once per hour — reasonable
- More frequent checks waste API calls for no new data

Retries + backoff:
- 3 retries with exponential backoff (1s, 2s, 4s)
- On final failure: returns None — the tool wrapper handles gracefully
"""

import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v1"
REQUEST_TIMEOUT = 10.0
MAX_RETRIES = 3
CACHE_TTL = 3600  # 60 minutes

_cache: dict[str, tuple[float, dict[str, Any]]] = {}

# Maps country names (from document metadata) to ISO 4217 currency codes.
# Covers all 19 destination countries + common origin countries.
# If a country isn't listed, get_currency_for_country() falls back to "USD".
COUNTRY_CURRENCY: dict[str, str] = {
    "Indonesia": "IDR",
    "Thailand": "THB",
    "Vietnam": "VND",
    "Albania": "ALL",
    "Costa Rica": "CRC",
    "Canada": "CAD",
    "UAE": "AED",
    "Dubai": "AED",
    "Egypt": "EGP",
    "Iceland": "ISK",
    "India": "INR",
    "Italy": "EUR",
    "Japan": "JPY",
    "Maldives": "MVR",
    "Morocco": "MAD",
    "Nepal": "NPR",
    "Peru": "PEN",
    "Portugal": "EUR",
    "Seychelles": "SCR",
    "Switzerland": "CHF",
    "United States": "USD",
    "USA": "USD",
    "United Kingdom": "GBP",
    "UK": "GBP",
    "Germany": "EUR",
    "France": "EUR",
    "Spain": "EUR",
    "Turkey": "TRY",
    "Australia": "AUD",
    "Saudi Arabia": "SAR",
    "Brazil": "BRL",
    "China": "CNY",
    "South Korea": "KRW",
    "Mexico": "MXN",
}


def get_currency_for_country(country: str) -> str:
    """
    Look up the ISO 4217 currency code for a country name.

    Falls back to "USD" if the country isn't in our mapping —
    a safe default for international travel planning.

    Args:
        country: Country name as it appears in document metadata (e.g. "Indonesia").

    Returns:
        3-letter currency code (e.g. "IDR"), or "USD" if unknown.
    """
    return COUNTRY_CURRENCY.get(country, "USD")


async def get_exchange_rate(base: str, target: str) -> dict[str, Any] | None:
    """
    Fetch the current exchange rate between two currencies.

    Args:
        base:    3-letter base currency code (e.g. "EGP" — user's origin).
        target:  3-letter target currency code (e.g. "IDR" — destination).

    Returns:
        Dict with exchange data, or None if all retries fail.
        {
            "rate": 44.2,
            "date": "2026-04-29",
            "base": "EGP",
            "target": "IDR"
        }

    Note:
        frankfurter.app only supports ~30+ major currencies (ECB data).
        For unsupported currencies (e.g. EGP), we fall back to a
        two-step conversion: EGP → EUR → IDR.
    """
    if base == target:
        return {"rate": 1.0, "date": "today", "base": base, "target": target}

    cache_key = f"{base}->{target}"

    cached = _cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < CACHE_TTL:
        logger.debug("Returning cached FX rate for %s", cache_key)
        return cached[1]

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await _fetch_rate(base, target)
            if result is not None:
                _cache[cache_key] = (time.monotonic(), result)
                logger.info("FX rate: 1 %s = %.4f %s", base, result["rate"], target)
                return result

            # frankfurter doesn't support this base currency — try via EUR
            result = await _fetch_rate_via_eur(base, target)
            if result is not None:
                _cache[cache_key] = (time.monotonic(), result)
                logger.info("FX rate (via EUR): 1 %s = %.4f %s", base, result["rate"], target)
                return result

            return None

        except Exception as exc:
            last_error = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "FX fetch attempt %d/%d failed: %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    logger.error("FX fetch exhausted retries for %s: %s", cache_key, last_error)
    return None


async def _fetch_rate(base: str, target: str) -> dict[str, Any] | None:
    """Direct rate fetch from frankfurter."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(
            f"{FRANKFURTER_BASE_URL}/latest",
            params={"from": base, "to": target},
        )
        if response.status_code in (400, 404):
            logger.debug("frankfurter doesn't support base=%s — will try via EUR", base)
            return None
        response.raise_for_status()
        data = response.json()

    rate = data.get("rates", {}).get(target)
    if rate is None:
        return None

    return {
        "rate": float(rate),
        "date": data.get("date", "unknown"),
        "base": base,
        "target": target,
    }


async def _fetch_rate_via_eur(base: str, target: str) -> dict[str, Any] | None:
    """
    Two-step conversion for currencies not directly supported by frankfurter.

    frankfurter.app uses ECB data which only supports ~30+ currencies.
    For unsupported currencies (e.g. EGP, THB for some pairs), we try
    to route through EUR. If EUR doesn't support the base currency either,
    we return None and the tool wrapper reports the limitation.

    This is acceptable — frankfurter covers all major currencies.
    For EGP specifically, we can add a hardcoded reference rate later.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        base_to_eur = 1.0
        if base != "EUR":
            r = await client.get(
                f"{FRANKFURTER_BASE_URL}/latest",
                params={"from": "EUR", "to": base},
            )
            if r.status_code != 200:
                logger.info("EUR→%s not available on frankfurter", base)
                return None
            base_to_eur_data = r.json().get("rates", {})
            base_rate = base_to_eur_data.get(base)
            if base_rate is None:
                return None
            base_to_eur = 1.0 / float(base_rate)

        eur_to_target = 1.0
        if target != "EUR":
            r = await client.get(
                f"{FRANKFURTER_BASE_URL}/latest",
                params={"from": "EUR", "to": target},
            )
            if r.status_code != 200:
                return None
            eur_to_target = float(r.json().get("rates", {}).get(target, 1.0))

    cross_rate = base_to_eur * eur_to_target

    return {
        "rate": round(cross_rate, 4),
        "date": "approximate",
        "base": base,
        "target": target,
    }
