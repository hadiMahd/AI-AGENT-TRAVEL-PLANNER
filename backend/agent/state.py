"""
Agent state — TypedDict for the LangGraph StateGraph.

This defines what data flows between graph nodes. Each node reads
and writes to this state, and the graph passes it along.

Why TypedDict over a Pydantic model?
- LangGraph's StateGraph expects a TypedDict for state
- It's simpler — just keys and types, no validation logic
- Validation happens at the tool input level (Pydantic schemas)
- The state is internal to the graph — not exposed to the API
"""

from typing import Any, TypedDict


class AgentState(TypedDict):
    query: str
    intent: str  # "casual" | "info" | "travel"
    history: list[dict]
    working_summary: str
    rewritten_query: str
    origin_country: str | None
    destination_country: str | None
    destination_lat: float | None
    destination_long: float | None
    destination_currency: str | None
    origin_currency: str | None
    needs_user_input: bool
    user_question: str | None
    tool_results: dict[str, str]
    data: dict[str, Any]
    tool_logs: list[dict]
    final_response: str
    prompt_tokens: int
    completion_tokens: int
    clarification_attempts: int
