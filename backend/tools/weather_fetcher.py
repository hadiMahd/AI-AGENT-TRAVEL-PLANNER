"""
Weather fetcher tool — wraps the wttr.in weather service as a LangGraph tool.

Takes latitude and longitude (from RAG metadata) and returns current
weather conditions at that location.

Per INSTRUCTIONS.md: "Pydantic validation on every tool input."
"""

import json
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.weather import get_weather

logger = logging.getLogger(__name__)


class WeatherToolInput(BaseModel):
    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude of the destination (from RAG metadata)",
    )
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude of the destination (from RAG metadata)",
    )


@tool(args_schema=WeatherToolInput)
async def weather_fetcher(latitude: float, longitude: float) -> str:
    """
    Get current weather conditions at a destination by latitude/longitude.

    Use this tool when the user asks about weather at a destination.
    Requires latitude and longitude which come from RAG document metadata.
    Returns temperature, condition, humidity, and wind speed.
    """
    try:
        result = await get_weather(latitude, longitude)
        if result is None:
            return "[weather_fetcher] Weather data unavailable — wttr.in may be temporarily down."
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("weather_fetcher failed: %s", exc)
        return f"[weather_fetcher ERROR] {exc}"
