"""
Flight searcher tool — wraps DuckDuckGo web search for flight information.

Takes origin and destination city names and searches the web for
flight prices and availability from travel booking sites.

Per INSTRUCTIONS.md: "Pydantic validation on every tool input."
"""

import json
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.flights import search_flights

logger = logging.getLogger(__name__)


class FlightToolInput(BaseModel):
    origin: str = Field(
        ...,
        min_length=1,
        description="Origin city name (e.g. 'Cairo', 'London')",
    )
    destination: str = Field(
        ...,
        min_length=1,
        description="Destination city name (e.g. 'Bali', 'Paris')",
    )


@tool(args_schema=FlightToolInput)
async def flight_searcher(origin: str, destination: str) -> str:
    """
    Search the web for flight information between two cities.

    Use this tool when the user asks about flights or travel costs.
    Returns snippets from travel booking sites (Skyscanner, Kayak, etc.)
    with price ranges and route information.
    """
    try:
        results = await search_flights(origin, destination)
        if results is None:
            return "[flight_searcher] Flight search unavailable — DuckDuckGo may be temporarily blocked."
        if not results:
            return f"[flight_searcher] No flight results found for {origin} → {destination}."
        return json.dumps(results, indent=2)
    except Exception as exc:
        logger.error("flight_searcher failed: %s", exc)
        return f"[flight_searcher ERROR] {exc}"
