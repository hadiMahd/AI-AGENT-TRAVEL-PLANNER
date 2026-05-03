"""Tool isolation tests — _execute_tool with mocked services, no real external calls."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.graph import _execute_tool


class TestMLPredictor:
    @pytest.mark.asyncio
    async def test_valid_dict_input_predicts(self, fake_ml_model):
        tool_input = {
            "active_movement": 0.8,
            "relaxation": 0.2,
            "cultural_interest": 0.5,
            "cost_sensitivity": 0.3,
            "luxury_preference": 0.7,
            "family_friendliness": 0.1,
            "nature_orientation": 0.9,
            "social_group": 0.4,
        }

        payload, status = await _execute_tool(
            "ml_predictor", tool_input, fake_ml_model, None, None
        )

        assert payload is not None
        assert payload["predicted_style"] == "adventure"
        assert "adventure" in status
        assert "conf" in status

    @pytest.mark.asyncio
    async def test_list_input_rejected_with_error(self, fake_ml_model):
        """LLM mistake: positional array instead of keyed dict."""
        tool_input = [0.8, 0.2, 0.5, 0.3, 0.7, 0.1, 0.9, 0.4]

        payload, status = await _execute_tool(
            "ml_predictor", tool_input, fake_ml_model, None, None
        )

        assert payload is None
        assert "ERROR" in status
        assert "expected dict input" in status

    @pytest.mark.asyncio
    async def test_missing_keys_rejected(self, fake_ml_model):
        tool_input = {"active_movement": 0.5}

        payload, status = await _execute_tool(
            "ml_predictor", tool_input, fake_ml_model, None, None
        )

        assert payload is None
        assert "ERROR" in status

    @pytest.mark.asyncio
    async def test_out_of_range_value_rejected(self, fake_ml_model):
        tool_input = {
            "active_movement": 1.5,
            "relaxation": 0.5,
            "cultural_interest": 0.5,
            "cost_sensitivity": 0.5,
            "luxury_preference": 0.5,
            "family_friendliness": 0.5,
            "nature_orientation": 0.5,
            "social_group": 0.5,
        }

        payload, status = await _execute_tool(
            "ml_predictor", tool_input, fake_ml_model, None, None
        )

        assert payload is None
        assert "ERROR" in status

    @pytest.mark.asyncio
    async def test_none_model_returns_error(self):
        payload, status = await _execute_tool(
            "ml_predictor", {"active_movement": 0.5, "relaxation": 0.5, "cultural_interest": 0.5, "cost_sensitivity": 0.5, "luxury_preference": 0.5, "family_friendliness": 0.5, "nature_orientation": 0.5, "social_group": 0.5}, None, None, None
        )

        assert payload is None
        assert "ERROR" in status
        assert "not loaded" in status.lower() or "attribute" in status.lower()


class TestRAGRetriever:
    @pytest.mark.asyncio
    async def test_retrieval_with_results(self, mocker, fake_embedder, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch(
            "agent.graph.similarity_search",
            return_value=[
                {
                    "content": "Bali is a tropical paradise...",
                    "metadata": {"country": "Indonesia", "style": "relaxation"},
                    "score": 0.92,
                },
                {
                    "content": "Maldives offers pristine beaches...",
                    "metadata": {"country": "Maldives", "style": "relaxation"},
                    "score": 0.85,
                },
            ],
        )

        payload, status = await _execute_tool(
            "rag_retriever", {"query": "beach destination", "k": 3},
            None, fake_embedder, fake_db,
        )

        assert payload is not None
        assert len(payload) == 2
        assert payload[0]["score"] == 0.92
        assert "hits about Indonesia, Maldives" in status

    @pytest.mark.asyncio
    async def test_no_results_found(self, mocker, fake_embedder, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch("agent.graph.similarity_search", return_value=[])

        payload, status = await _execute_tool(
            "rag_retriever", {"query": "unknown", "k": 3},
            None, fake_embedder, fake_db,
        )

        assert payload == []
        assert "no documents found" in status

    @pytest.mark.asyncio
    async def test_all_below_threshold_filtered(self, mocker, fake_embedder, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        mocker.patch(
            "agent.graph.similarity_search",
            return_value=[{"content": "x", "metadata": {}, "score": 0.3}],
        )

        payload, status = await _execute_tool(
            "rag_retriever", {"query": "noise", "k": 3},
            None, fake_embedder, fake_db,
        )

        assert payload == []
        assert "below relevance threshold" in status

    @pytest.mark.asyncio
    async def test_k_capped_at_3(self, mocker, fake_embedder, fake_db):
        mocker.patch("agent.graph.embed_query", return_value=[0.1] * 1536)
        patched = mocker.patch("agent.graph.similarity_search", return_value=[])

        await _execute_tool(
            "rag_retriever", {"query": "test", "k": 10},
            None, fake_embedder, fake_db,
        )

        # k=10 should be capped to min(10, 3) = 3
        assert patched.call_args[0][2] == 3
