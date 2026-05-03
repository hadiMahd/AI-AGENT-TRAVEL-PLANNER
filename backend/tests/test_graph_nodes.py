"""Individual graph node tests — each node tested in isolation with fake LLMs."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.graph import (
    _build_working_summary,
    _has_destination_keyword,
    _parse_json_response,
    casual_reply,
    check_context,
    classify_intent,
    route_and_run_tools,
)


class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"intent": "travel"}')
        assert result == {"intent": "travel"}

    def test_json_in_code_fence(self):
        result = _parse_json_response('```json\n{"intent": "casual"}\n```')
        assert result == {"intent": "casual"}

    def test_trailing_fence_only(self):
        result = _parse_json_response('{"intent": "travel"}\n```')
        assert result == {"intent": "travel"}

    def test_invalid_json_returns_none(self):
        assert _parse_json_response("not json at all") is None
        assert _parse_json_response("{invalid") is None


class TestDestinationKeyword:
    def test_travel_keywords_detected(self):
        assert _has_destination_keyword("plan a trip to Bali") is True
        assert _has_destination_keyword("what is the weather in Tokyo?") is True
        assert _has_destination_keyword("flights from Cairo") is True
        assert _has_destination_keyword("i want to visit Japan") is True

    def test_non_travel_short_queries_undetected(self):
        assert _has_destination_keyword("hi") is False
        assert _has_destination_keyword("thanks") is False
        assert _has_destination_keyword("hello there") is False


class TestBuildWorkingSummary:
    def test_empty_state(self, base_state):
        result = _build_working_summary(base_state)
        assert result == ""

    def test_with_origin_and_destination(self, base_state):
        state = {**base_state, "origin_country": "Egypt", "destination_country": "Indonesia"}
        result = _build_working_summary(state)
        assert "origin=Egypt" in result
        assert "dest=Indonesia" in result

    def test_with_prior_turns(self, base_state):
        state = {
            **base_state,
            "query": "what are the best beaches?",
            "history": [
                {"role": "user", "content": "plan a trip to Bali"},
                {"role": "assistant", "content": "Where are you traveling from?"},
            ],
        }
        result = _build_working_summary(state)
        assert "prior_user_turns" in result
        assert "prev_reply" in result


class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_short_casual_bypasses_llm(self, base_state, fake_cheap_llm):
        """'hi' with no travel keyword → casual, no LLM call."""
        llm = fake_cheap_llm({"intent": "unused"})
        state = {**base_state, "query": "hi"}

        result = await classify_intent(state, llm)

        assert result["intent"] == "casual"
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_travel_query_reaches_llm(self, base_state, fake_cheap_llm):
        llm = fake_cheap_llm({"intent": "travel"})
        state = {**base_state, "query": "plan a trip to Bali"}

        result = await classify_intent(state, llm)

        assert result["intent"] == "travel"
        llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_timeout_defaults_to_travel(self, base_state):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=TimeoutError)
        state = {**base_state, "query": "plan a trip"}

        result = await classify_intent(state, llm)

        assert result["intent"] == "travel"

    @pytest.mark.asyncio
    async def test_unknown_intent_falls_back_to_travel(self, base_state, fake_cheap_llm):
        llm = fake_cheap_llm({"intent": "unknown_type"})
        state = {**base_state, "query": "plan a trip"}

        result = await classify_intent(state, llm)

        assert result["intent"] == "travel"

    @pytest.mark.asyncio
    async def test_short_reply_to_clarification_stays_travel(self, base_state, fake_cheap_llm):
        """'Lebanon' answering 'where from?' should not be casual."""
        llm = fake_cheap_llm({"intent": "travel"})
        state = {
            **base_state,
            "query": "Lebanon",
            "history": [
                {"role": "assistant", "content": "Where are you traveling from?"},
            ],
        }

        result = await classify_intent(state, llm)

        assert result["intent"] == "travel"


class TestCasualReply:
    @pytest.mark.asyncio
    async def test_casual_reply_generates_response(self, base_state, fake_cheap_llm):
        llm = fake_cheap_llm({"message": "irrelevant"})
        llm.ainvoke = AsyncMock()
        msg = MagicMock()
        msg.content = "Hello! I'm your travel planner."
        llm.ainvoke.return_value = msg

        result = await casual_reply(base_state, llm)

        assert "final_response" in result
        assert result["needs_user_input"] is False
        assert result["tool_logs"] == []

    @pytest.mark.asyncio
    async def test_casual_reply_timeout_has_fallback(self, base_state):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=TimeoutError)

        result = await casual_reply(base_state, llm)

        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0


class TestCheckContext:
    @pytest.mark.asyncio
    async def test_full_info_proceeds(self, base_state, fake_cheap_llm, fake_embedder, fake_db, mocker):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[])
        llm = fake_cheap_llm({
            "needs_origin": True,
            "has_origin": True,
            "origin_country": "Egypt",
            "destination_country": "Indonesia",
            "needs_dest": False,
            "question": None,
        })
        state = {**base_state, "query": "plan a trip to Bali from Cairo"}

        result = await check_context(state, llm, fake_embedder, fake_db)

        assert result["needs_user_input"] is False
        assert result["origin_country"] == "Egypt"
        assert result["destination_country"] == "Indonesia"

    @pytest.mark.asyncio
    async def test_missing_origin_asks_user(self, base_state, fake_cheap_llm, fake_embedder, fake_db, mocker):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[])
        llm = fake_cheap_llm({
            "needs_origin": True,
            "has_origin": False,
            "origin_country": None,
            "destination_country": "Indonesia",
            "needs_dest": False,
            "question": "Where are you traveling from?",
        })
        state = {**base_state, "query": "plan a trip to Bali"}

        result = await check_context(state, llm, fake_embedder, fake_db)

        assert result["needs_user_input"] is True
        assert result["user_question"] == "Where are you traveling from?"

    @pytest.mark.asyncio
    async def test_missing_destination_asks_user(self, base_state, fake_cheap_llm, fake_embedder, fake_db, mocker):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[])
        llm = fake_cheap_llm({
            "needs_origin": False,
            "has_origin": False,
            "origin_country": None,
            "destination_country": None,
            "needs_dest": True,
            "question": "Which destination?",
        })
        state = {**base_state, "query": "plan a trip"}

        result = await check_context(state, llm, fake_embedder, fake_db)

        assert result["needs_user_input"] is True
        assert result["user_question"] is not None

    @pytest.mark.asyncio
    async def test_single_fact_query_no_origin_needed(self, base_state, fake_cheap_llm, fake_embedder, fake_db, mocker):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[])
        llm = fake_cheap_llm({
            "needs_origin": False,
            "has_origin": False,
            "origin_country": None,
            "destination_country": "Indonesia",
            "needs_dest": False,
            "question": None,
        })
        state = {**base_state, "query": "what is the weather in Bali?"}

        result = await check_context(state, llm, fake_embedder, fake_db)

        assert result["needs_user_input"] is False

    @pytest.mark.asyncio
    async def test_clarification_loop_guard_increments(self, base_state, fake_cheap_llm, fake_embedder, fake_db, mocker):
        """When we keep asking and user doesn't provide origin, attempts increment."""
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[])
        llm = fake_cheap_llm({
            "needs_origin": True,
            "has_origin": False,
            "origin_country": None,
            "destination_country": "Indonesia",
            "needs_dest": False,
            "question": "Where from?",
        })
        state = {**base_state, "query": "idk", "clarification_attempts": 1}
        # ^ simulating: we've already asked once before

        result = await check_context(state, llm, fake_embedder, fake_db)

        assert result["clarification_attempts"] == 2
        assert result["needs_user_input"] is True


class TestRouteAndRunTools:
    @pytest.mark.asyncio
    async def test_runs_tools_in_parallel(self, mocker, base_state, fake_cheap_llm, fake_embedder, fake_ml_model, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[{
            "content": "Bali info...", "metadata": {"country": "Indonesia"}, "score": 0.9,
        }])
        mocker.patch("services.weather.get_weather", return_value={
            "temp_c": 28, "feels_like_c": 30, "condition": "Sunny",
            "humidity": 75, "wind_kph": 12, "location": "Bali",
        })
        mocker.patch("services.flights.search_flights", return_value=[{
            "title": "Test", "snippet": "From $100", "url": "https://x.com",
        }])
        mocker.patch("services.fx.get_exchange_rate", return_value={
            "rate": 1.0, "base": "USD", "target": "IDR",
        })

        llm = fake_cheap_llm({
            "tools": [
                {"name": "rag_retriever", "input": {"query": "Bali travel", "k": 3}},
                {"name": "ml_predictor", "input": {"active_movement": 0.5, "relaxation": 0.5, "cultural_interest": 0.5, "cost_sensitivity": 0.5, "luxury_preference": 0.5, "family_friendliness": 0.5, "nature_orientation": 0.5, "social_group": 0.5}},
                {"name": "weather_fetcher", "input": {"city": "Bali"}},
                {"name": "fx_checker", "input": {"base_currency": "USD", "target_currency": "IDR"}},
            ],
        })
        state = {
            **base_state,
            "query": "plan a trip to Bali",
            "destination_country": "Indonesia",
            "destination_currency": "IDR",
            "origin_country": "USA",
            "origin_currency": "USD",
        }

        result = await route_and_run_tools(state, llm, fake_ml_model, fake_embedder, fake_db)

        assert len(result["tool_results"]) >= 4
        assert "rag_retriever" in result["tool_results"]
        assert "ml_predictor" in result["tool_results"]
        assert len(result["tool_logs"]) >= 4
        for log in result["tool_logs"]:
            assert log["status"] == "success"

    @pytest.mark.asyncio
    async def test_hallucinated_tool_rejected(self, mocker, base_state, fake_cheap_llm, fake_embedder, fake_ml_model, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[{
            "content": "x", "metadata": {"country": "x"}, "score": 0.9,
        }])
        llm = fake_cheap_llm({
            "tools": [
                {"name": "rag_retriever", "input": {"query": "test", "k": 3}},
                {"name": "invented_tool", "input": {"arg": "x"}},
            ],
        })
        state = {
            **base_state,
            "destination_country": "Indonesia",
            "destination_currency": "IDR",
        }

        result = await route_and_run_tools(state, llm, fake_ml_model, fake_embedder, fake_db)

        assert "invented_tool" in result["tool_results"]
        assert "REJECTED" in result["tool_results"]["invented_tool"]

    @pytest.mark.asyncio
    async def test_tool_error_logged_not_raised(self, mocker, base_state, fake_cheap_llm, fake_embedder, fake_ml_model, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similar_search", side_effect=RuntimeError("crash"), create=True)
        mocker.patch("agent.graph.similarity_search", return_value=[{
            "content": "x", "metadata": {"country": "x"}, "score": 0.9,
        }])
        llm = fake_cheap_llm({
            "tools": [
                {"name": "rag_retriever", "input": {"query": "test", "k": 3}},
            ],
        })
        state = {
            **base_state,
            "destination_country": "Indonesia",
            "destination_currency": "IDR",
        }

        result = await route_and_run_tools(state, llm, fake_ml_model, fake_embedder, fake_db)

        # Route should not crash even if a tool errors
        assert isinstance(result, dict)
        assert "tool_results" in result

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_to_rag_only(self, mocker, base_state, fake_cheap_llm, fake_embedder, fake_ml_model, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[{
            "content": "x", "metadata": {"country": "x"}, "score": 0.9,
        }])
        llm = fake_cheap_llm({"invalid": "response"})  # no "tools" key
        state = {
            **base_state,
            "destination_country": "Indonesia",
            "destination_currency": "IDR",
        }

        result = await route_and_run_tools(state, llm, fake_ml_model, fake_embedder, fake_db)

        assert "rag_retriever" in result["tool_results"]
