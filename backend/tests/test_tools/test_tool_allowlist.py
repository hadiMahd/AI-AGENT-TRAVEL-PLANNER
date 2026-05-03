"""Tool allowlist tests — hallucination guard."""

from tools import validate_tool


class TestAllowlist:
    def test_registered_tools_pass(self):
        assert validate_tool("rag_retriever") is True
        assert validate_tool("ml_predictor") is True
        assert validate_tool("weather_fetcher") is True
        assert validate_tool("flight_searcher") is True
        assert validate_tool("fx_checker") is True

    def test_hallucinated_tools_rejected(self):
        assert validate_tool("search_internet") is False
        assert validate_tool("book_flight") is False
        assert validate_tool("send_sms") is False
        assert validate_tool("calculate_budget") is False
        assert validate_tool("") is False

    def test_case_sensitive(self):
        assert validate_tool("RAG_RETRIEVER") is False
        assert validate_tool("Weather_Fetcher") is False
