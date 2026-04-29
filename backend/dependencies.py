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
- get_embedder:   Azure OpenAI text-embedding-3-small — RAG pipeline
- get_current_user: JWT-authenticated User ORM object — for protected routes
"""

from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from models.alchemy import User
from services.auth import decode_token
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


# ── Auth: Current User ──────────────────────────────────────────


# OAuth2PasswordBearer tells FastAPI to look for "Authorization: Bearer <token>"
# in the request header. tokenUrl points to our login endpoint so Swagger UI
# can auto-generate the "Authorize" button.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Extract and validate the JWT from the Authorization header, then look up the User.

    Flow:
    1. OAuth2PasswordBearer extracts the token from "Authorization: Bearer <token>"
    2. decode_token() validates signature + expiration — raises JWTError on failure
    3. Extract the "sub" claim (user ID) from the payload
    4. Query the users table for that ID
    5. Return the User ORM object (or 401 if not found)

    This is the gatekeeper for all protected routes — add it as a Depends()
    to any endpoint that requires authentication.

    Why not cache the user lookup?
    - JWT validation is already fast (symmetric key, no DB hit)
    - The DB hit is a single SELECT by primary key — sub-millisecond
    - Caching would add complexity and risk stale data (e.g. user deleted)

    Args:
        token: JWT string extracted from the Authorization header.
        db:    Async SQLAlchemy session (from Depends(get_db)).

    Returns:
        User ORM object — the authenticated user.

    Raises:
        HTTPException 401: if the token is invalid, expired, or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Step 1: Decode and validate the JWT
    try:
        payload = decode_token(token)
    except JWTError:
        raise credentials_exception

    # Step 2: Extract user ID from the "sub" claim
    user_id: str | None = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    # Step 3: Look up the user in the database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user


# ── Pre-built Depends() objects ──────────────────────────────
# Use these in route signatures for cleaner code:
#   async def my_route(llm: ChatOpenAI = StrongLlmDep):

ModelDep = Depends(get_model)
StrongLlmDep = Depends(get_strong_llm)
CheapLlmDep = Depends(get_cheap_llm)
EmbedderDep = Depends(get_embedder)
CurrentUserDep = Depends(get_current_user)
