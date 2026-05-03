"""E2E agent test — full LangGraph pipeline with all external services mocked."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_openai import ChatOpenAI

from agent.graph import build_graph
from agent.state import AgentState


def _make_llm_responses(responses: list[str]):
    """Create a cheap LLM mock that returns successive JSON strings."""
    async def fake_ainvoke(prompt):
        msg = MagicMock()
        msg.content = responses.pop(0) if responses else "{}"
        msg.response_metadata = {}
        return msg

    llm = MagicMock(spec=ChatOpenAI)
    llm.ainvoke = fake_ainvoke
    return llm


@pytest.mark.asyncio
async def test_full_travel_plan_flow(mocker, fake_strong_llm, fake_embedder, fake_ml_model, fake_db):
    """User: 'plan to Bali from Cairo' → full agent run with 5 tools."""
    # Mock all external service boundaries
    mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
    mocker.patch("agent.graph.similarity_search", return_value=[{
        "content": "Bali is a tropical paradise with temples and beaches.",
        "metadata": {"country": "Indonesia", "style": "relaxation", "latitude": -8.34, "longitude": 115.09},
        "score": 0.92,
    }])
    mocker.patch("services.weather.get_weather", return_value={
        "temp_c": 28, "feels_like_c": 30, "condition": "Sunny",
        "humidity": 75, "wind_kph": 12, "location": "Bali",
    })
    mocker.patch("services.flights.search_flights", return_value=[{
        "title": "Cairo to Bali", "snippet": "From $450, 1 stop via Dubai, 12h",
        "url": "https://example.com/flights",
    }])
    mocker.patch("services.fx.get_exchange_rate", return_value={
        "rate": 0.000016, "date": "2026-04-29", "base": "EGP", "target": "IDR",
    })

    # Cheap LLM responds sequentially: classify → check_context → route
    cheap = _make_llm_responses([
        json.dumps({"intent": "travel"}),
        json.dumps({
            "needs_origin": True, "has_origin": True,
            "origin_country": "Egypt", "destination_country": "Indonesia",
            "needs_dest": False, "question": None,
        }),
        json.dumps({"tools": [
            {"name": "rag_retriever", "input": {"query": "Indonesia", "k": 3}},
            {"name": "weather_fetcher", "input": {"city": "Bali"}},
            {"name": "flight_searcher", "input": {"origin": "CAI", "destination": "DPS"}},
            {"name": "fx_checker", "input": {"base_currency": "EGP", "target_currency": "IDR"}},
            {"name": "ml_predictor", "input": {
                "active_movement": 0.5, "relaxation": 0.7, "cultural_interest": 0.5,
                "cost_sensitivity": 0.5, "luxury_preference": 0.5,
                "family_friendliness": 0.5, "nature_orientation": 0.5, "social_group": 0.5,
            }},
        ]}),
    ])

    strong = fake_strong_llm([
        "## ", "Your ", "Trip ", "to ", "Bali", "\n\n",
        "**Weather:** ", "28°C", " sunny", "\n\n",
        "## Budget\n\n",
        "Flights from ", "Cairo ", "cost ~$450", "\n",
    ])

    state: AgentState = {
        "query": "plan a trip to Bali from Cairo",
        "intent": "travel",
        "history": [],
        "working_summary": "",
        "rewritten_query": "",
        "origin_country": None,
        "destination_country": None,
        "destination_lat": None,
        "destination_long": None,
        "destination_currency": None,
        "origin_currency": None,
        "needs_user_input": False,
        "user_question": None,
        "tool_results": {},
        "data": {},
        "tool_logs": [],
        "final_response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "clarification_attempts": 0,
    }

    graph = build_graph(cheap, strong, fake_ml_model, fake_embedder, fake_db)
    compiled = graph.compile()
    result = await compiled.ainvoke(state)

    assert result["destination_country"] == "Indonesia"
    assert result["origin_country"] == "Egypt"
    assert "rag_retriever" in result["tool_results"]
    assert "weather_fetcher" in result["tool_results"]
    assert "flight_searcher" in result["tool_results"]
    assert "fx_checker" in result["tool_results"]
    assert "ml_predictor" in result["tool_results"]
    assert len(result["tool_logs"]) == 5
    for log in result["tool_logs"]:
        assert log["status"] == "success", f"tool {log['tool_name']} failed"


@pytest.mark.asyncio
async def test_casual_query_bypasses_tools(mocker, fake_embedder, fake_db, fake_ml_model):
    """'hi' → casual reply, no tools run, no synthesis."""
    cheap = _make_llm_responses([
        json.dumps({"intent": "casual"}),
    ])
    cheap.ainvoke.side_effect = None  # reset for casual_reply node
    async def casual_ainvoke(prompt):
        msg = MagicMock()
        msg.content = "Hey! Tell me where you'd like to go."
        msg.response_metadata = {}
        return msg
    cheap.ainvoke = casual_ainvoke

    strong = MagicMock(spec=ChatOpenAI)

    state: AgentState = {
        "query": "hi",
        "intent": "travel",
        "history": [],
        "working_summary": "",
        "rewritten_query": "",
        "origin_country": None,
        "destination_country": None,
        "destination_lat": None,
        "destination_long": None,
        "destination_currency": None,
        "origin_currency": None,
        "needs_user_input": False,
        "user_question": None,
        "tool_results": {},
        "data": {},
        "tool_logs": [],
        "final_response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "clarification_attempts": 0,
    }

    graph = build_graph(cheap, strong, fake_ml_model, fake_embedder, fake_db)
    compiled = graph.compile()
    result = await compiled.ainvoke(state)

    assert result.get("intent") == "casual"
    assert result.get("final_response") is not None
    assert len(result.get("tool_logs", [])) == 0


@pytest.mark.asyncio
async def test_missing_input_asks_user(mocker, fake_embedder, fake_db, fake_ml_model):
    """'plan a trip' without destination → user_question returned."""
    mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
    mocker.patch("agent.graph.similarity_search", return_value=[])

    cheap = _make_llm_responses([
        json.dumps({"intent": "travel"}),
        json.dumps({
            "needs_origin": True, "has_origin": False,
            "origin_country": None, "destination_country": None,
            "needs_dest": True, "question": "Which destination are you interested in?",
        }),
    ])

    strong = MagicMock(spec=ChatOpenAI)

    state: AgentState = {
        "query": "plan a trip",
        "intent": "travel",
        "history": [],
        "working_summary": "",
        "rewritten_query": "",
        "origin_country": None,
        "destination_country": None,
        "destination_lat": None,
        "destination_long": None,
        "destination_currency": None,
        "origin_currency": None,
        "needs_user_input": False,
        "user_question": None,
        "tool_results": {},
        "data": {},
        "tool_logs": [],
        "final_response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "clarification_attempts": 0,
    }

    graph = build_graph(cheap, strong, fake_ml_model, fake_embedder, fake_db)
    compiled = graph.compile()
    result = await compiled.ainvoke(state)

    assert result["needs_user_input"] is True
    assert result["user_question"] is not None


@pytest.mark.asyncio
async def test_tool_failure_graceful_degradation(mocker, fake_embedder, fake_ml_model, fake_db):
    """When a tool fails, the agent still produces a result."""
    mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
    mocker.patch("agent.graph.similarity_search", return_value=[])
    mocker.patch("agent.graph.get_weather_by_city", side_effect=RuntimeError("weather down"))

    cheap = _make_llm_responses([
        json.dumps({"intent": "travel"}),
        json.dumps({"needs_origin": False, "has_origin": False, "origin_country": None,
                     "destination_country": "Indonesia", "needs_dest": False, "question": None}),
        json.dumps({"tools": [
            {"name": "weather_fetcher", "input": {"city": "Bali"}},
        ]}),
    ])

    strong = MagicMock(spec=ChatOpenAI)
    async def fake_stream(prompt):
        chunk = MagicMock()
        chunk.content = "Weather unavailable but here's what I know..."
        yield chunk
    strong.astream = fake_stream

    state: AgentState = {
        "query": "weather in Bali",
        "intent": "travel",
        "history": [],
        "working_summary": "",
        "rewritten_query": "",
        "origin_country": None,
        "destination_country": None,
        "destination_lat": None,
        "destination_long": None,
        "destination_currency": None,
        "origin_currency": None,
        "needs_user_input": False,
        "user_question": None,
        "tool_results": {},
        "data": {},
        "tool_logs": [],
        "final_response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "clarification_attempts": 0,
    }

    graph = build_graph(cheap, strong, fake_ml_model, fake_embedder, fake_db)
    compiled = graph.compile()
    result = await compiled.ainvoke(state)

    assert "weather_fetcher" in result["tool_results"]
    assert "ERROR" in result["tool_results"]["weather_fetcher"]
    assert len(result["tool_logs"]) >= 1
    # Tool errors caught by _execute_tool are returned as structured messages,
    # not raised — so the tool log status remains "success".
    # The error is visible in the output_payload.
    assert any(
        "ERROR" in str(log.get("output_payload", ""))
        for log in result["tool_logs"]
    )
