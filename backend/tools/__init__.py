"""
Tool registry and allowlist for the LangGraph agent.

Per INSTRUCTIONS.md: "Explicit tool allowlist. Reject hallucinated tools."
If the LLM tries to call a tool not in this list, we reject it.

The allowlist serves two purposes:
1. Security — prevent the LLM from inventing tools we haven't defined
2. Debugging — every tool has a known name, input schema, and output format

Why a centralized registry?
- One place to see all available tools — easy to audit
- The graph imports from here, not from individual tool files
- Adding a new tool = adding it to ALLOWED_TOOLS + creating the file
"""

from tools.fx_checker import fx_checker
from tools.flight_searcher import flight_searcher
from tools.ml_predictor import ml_predictor
from tools.rag_retriever import rag_retriever
from tools.weather_fetcher import weather_fetcher

ALLOWED_TOOLS: set[str] = {
    "rag_retriever",
    "ml_predictor",
    "weather_fetcher",
    "flight_searcher",
    "fx_checker",
}

TOOL_MAP: dict[str, object] = {
    "rag_retriever": rag_retriever,
    "ml_predictor": ml_predictor,
    "weather_fetcher": weather_fetcher,
    "flight_searcher": flight_searcher,
    "fx_checker": fx_checker,
}


def validate_tool(name: str) -> bool:
    """Check if a tool name is in the allowlist."""
    return name in ALLOWED_TOOLS


def get_tool(name: str):
    """Get a tool function by name. Returns None if not in allowlist."""
    return TOOL_MAP.get(name)


__all__ = [
    "ALLOWED_TOOLS",
    "TOOL_MAP",
    "validate_tool",
    "get_tool",
    "rag_retriever",
    "ml_predictor",
    "weather_fetcher",
    "flight_searcher",
    "fx_checker",
]
