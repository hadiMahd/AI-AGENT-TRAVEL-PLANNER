import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def fake_cheap_llm():
    """Factory: call `fake_cheap_llm({"intent": "travel"})` to get a mock.

    Returns a ChatOpenAI mock whose `ainvoke` resolves with the given JSON
    dict as `.content` and has a `response_metadata` with token counts.
    """

    def _make(json_response: dict) -> MagicMock:
        msg = MagicMock()
        msg.content = json.dumps(json_response)
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        llm = MagicMock(spec=ChatOpenAI)
        llm.ainvoke = AsyncMock(return_value=msg)
        return llm

    return _make


@pytest.fixture
def fake_strong_llm():
    """Factory: call `fake_strong_llm(["Hello", " world"])` to get a mock.

    Returns a ChatOpenAI mock whose `astream` yields chunks with `.content`.
    """

    def _make(tokens: list[str]) -> MagicMock:
        async def _stream(_prompt):
            for t in tokens:
                chunk = MagicMock()
                chunk.content = t
                yield chunk

        llm = MagicMock(spec=ChatOpenAI)
        llm.astream = _stream
        return llm

    return _make


@pytest.fixture
def fake_embedder() -> MagicMock:
    """Returns an OpenAIEmbeddings mock that returns dummy 1536-dim vectors."""
    emb = MagicMock(spec=OpenAIEmbeddings)
    emb.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    emb.aembed_documents = AsyncMock(
        return_value=[[0.1] * 1536 for _ in range(5)]
    )
    return emb


@pytest.fixture
def fake_ml_model() -> MagicMock:
    """Returns a scikit-learn model mock that predicts 'adventure'."""
    import numpy as np

    model = MagicMock()
    model.predict = MagicMock(return_value=np.array(["adventure"]))
    model.predict_proba = MagicMock(
        return_value=np.array([[0.1, 0.85, 0.05]])
    )
    model.classes_ = np.array(["budget", "adventure", "luxury"])
    return model


@pytest.fixture
def fake_db() -> AsyncMock:
    """Returns an AsyncMock matching AsyncSession interface."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    session.get_bind = MagicMock()
    return session


@pytest.fixture
def base_state() -> dict:
    """Minimal AgentState dict for node tests."""
    return {
        "query": "plan a trip to Bali",
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


@pytest.fixture
def app_with_overrides(fake_cheap_llm, fake_strong_llm, fake_embedder, fake_ml_model, fake_db):
    """FastAPI app with all core dependencies overridden to fakes.

    Unsuitable for SSE streaming tests — use build_graph + ainvoke directly
    for agent E2E tests.
    """
    from dependencies import (
        get_cheap_llm,
        get_embedder,
        get_model,
        get_strong_llm,
    )
    from db.session import get_db
    from main import app

    cheap = fake_cheap_llm({"intent": "travel"})
    strong = fake_strong_llm(["hello"])

    app.dependency_overrides[get_cheap_llm] = lambda: cheap
    app.dependency_overrides[get_strong_llm] = lambda: strong
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_model] = lambda: fake_ml_model
    app.dependency_overrides[get_db] = lambda: fake_db

    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def test_settings(monkeypatch):
    """Override DATABASE_URL and JWT secret for tests."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/testdb")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-tests-only")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com/openai/v1")
    monkeypatch.setenv("VAULT_ADDR", "")
    monkeypatch.setenv("VAULT_TOKEN", "")
