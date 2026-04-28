"""
FastAPI dependency injection providers.

Each function reads from app.state (populated during lifespan startup)
and returns the singleton. This makes every dependency:
- Overridable in tests via app.dependency_overrides[dep] = lambda: fake
- Consistent with the lifespan singleton pattern (INSTRUCTIONS.md)
- Type-safe — callers get the concrete type, not Any

Current dependencies:
- get_model:      ML model (scikit-learn, loaded via joblib)
- get_strong_llm: Azure OpenAI strong model — agent synthesis
- get_cheap_llm:  Azure OpenAI cheap model — mechanical tasks
- get_embedder:    Azure OpenAI text-embedding-3-small — RAG pipeline
"""

from typing import Any

from fastapi import Depends, Request
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from services.ml_inference import ModelNotAvailableError


# ── ML Model ────────────────────────────────────────────────


def get_model(request: Request) -> Any:
    """
    Provide the ML model loaded at FastAPI lifespan startup.

    Stored on app.state.ml_model by the lifespan context.
    If it failed to load, app.state.ml_model_error holds the exception.
    """
    model = getattr(request.app.state, "ml_model", None)
    if model is None:
        error = getattr(request.app.state, "ml_model_error", None)
        if error:
            raise error
        raise ModelNotAvailableError("ML model is not loaded")
    return model


# ── LLM Clients ──────────────────────────────────────────────


def get_strong_llm(request: Request) -> ChatOpenAI:
    """
    Provide the strong Azure OpenAI model client for agent synthesis.

    Created during lifespan startup and stored on app.state.strong_llm.
    If creation failed, raises a RuntimeError with the error detail.

    Use for: final answer synthesis, complex reasoning, plan generation.
    """
    llm = getattr(request.app.state, "strong_llm", None)
    if llm is None:
        error = getattr(request.app.state, "llm_error", None)
        raise RuntimeError(f"Strong LLM not initialized: {error or 'unknown error'}")
    return llm


def get_cheap_llm(request: Request) -> ChatOpenAI:
    """
    Provide the cheap Azure OpenAI model client for mechanical tasks.

    Created during lifespan startup and stored on app.state.cheap_llm.
    If creation failed, raises a RuntimeError with the error detail.

    Use for: arg extraction, query rewriting, tool input formatting.
    """
    llm = getattr(request.app.state, "cheap_llm", None)
    if llm is None:
        error = getattr(request.app.state, "llm_error", None)
        raise RuntimeError(f"Cheap LLM not initialized: {error or 'unknown error'}")
    return llm


def get_embedder(request: Request) -> OpenAIEmbeddings:
    """
    Provide the Azure OpenAI embedding client for the RAG pipeline.

    Created during lifespan startup and stored on app.state.embedder.
    If creation failed, raises a RuntimeError with the error detail.

    Use for: document ingestion (batch embed) and retrieval (query embed).
    """
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        error = getattr(request.app.state, "llm_error", None)
        raise RuntimeError(f"Embedder not initialized: {error or 'unknown error'}")
    return embedder


# ── Pre-built Depends() objects ──────────────────────────────
# Use these in route signatures for cleaner code:
#   async def my_route(llm: ChatOpenAI = StrongLlmDep):

ModelDep = Depends(get_model)
StrongLlmDep = Depends(get_strong_llm)
CheapLlmDep = Depends(get_cheap_llm)
EmbedderDep = Depends(get_embedder)
