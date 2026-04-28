"""
RAG API endpoints — ingestion and retrieval.

Endpoints:
  POST /rag/ingest  — load docs from disk → chunk → embed → insert into DB
  POST /rag/query   — embed query → cosine similarity search → return top-k results

Ingestion is NOT automatic on startup — it's a manual trigger.
This follows INSTRUCTIONS.md: "Do not re-ingest on every query."
Run POST /rag/ingest once after fresh DB setup, or whenever documents change.

Query is used for manual testing of the retrieval pipeline.
In production, retrieval will be called internally by the agent tool.

All LLM/embedding clients come from Depends() — they're lifespan singletons
stored on app.state, not module-level globals.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db.session import get_db
from dependencies import EmbedderDep
from rag.embedder import embed_query, embed_texts
from rag.ingestion import ingest_documents
from rag.retriever import similarity_search

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# ──────────────────────────────────────────────
# Pydantic request/response schemas
# ──────────────────────────────────────────────


class IngestionResponse(BaseModel):
    """Response body for POST /rag/ingest."""

    status: str
    documents_loaded: int
    chunks_created: int


class QueryRequest(BaseModel):
    """
    Request body for POST /rag/query.

    Attributes:
        query: Natural language query (e.g., "budget-friendly beach destination")
        k:     Number of results to return.
               Default: 5 (~25% of our ~20-25 chunk corpus).
               Min 1, max 20 to prevent excessive results.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query about travel destinations",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar documents to return",
    )


class QueryResultItem(BaseModel):
    """A single retrieved document with its similarity score."""

    content: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    """Response body for POST /rag/query."""

    query: str
    results: list[QueryResultItem]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Ingest all JSON documents into the vector store",
    description=(
        "Loads all JSON files from the documents/ directory, chunks them, "
        "generates embeddings via Azure OpenAI, and inserts them into the "
        "documents table. Idempotent — wipes and re-inserts on each call."
    ),
)
async def ingest_endpoint(
    db: AsyncSession = Depends(get_db),
    embedder: OpenAIEmbeddings = EmbedderDep,
) -> IngestionResponse:
    """
    Trigger full RAG ingestion pipeline.

    No request body needed — settings (documents_dir, chunk_size, overlap)
    come from the Settings singleton. The embedder comes from Depends()
    (lifespan singleton stored on app.state).

    Returns counts of documents loaded and chunks created on success.
    Returns HTTP 500 on any pipeline failure.
    """
    settings = get_settings()

    try:
        result = await ingest_documents(
            db=db,
            embedder=embedder,
            documents_dir=settings.documents_dir,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
    except FileNotFoundError as exc:
        logger.error("Ingestion failed — directory not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.error("Ingestion failed — no documents: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Ingestion failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return IngestionResponse(
        status="ok",
        documents_loaded=result["documents_loaded"],
        chunks_created=result["chunks_created"],
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG vector store with cosine similarity search",
    description=(
        "Embeds the query via Azure OpenAI, performs cosine similarity "
        "search against the documents table, and returns the top-k most "
        "similar document chunks with their scores."
    ),
)
async def query_endpoint(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedder: OpenAIEmbeddings = EmbedderDep,
) -> QueryResponse:
    """
    Perform a similarity search against the ingested documents.

    The embedder comes from Depends() (lifespan singleton).

    Request body:
        query: "budget-friendly beach destination in Europe"
        k:     5  (optional, default 5)

    Returns the query string and a list of matching documents
    sorted by cosine similarity score (highest first).
    """
    try:
        # Step 1: Embed the query via Azure OpenAI
        query_embedding = await embed_query(embedder, body.query)

        # Step 2: Perform cosine similarity search against documents table
        results = await similarity_search(
            db=db,
            query_embedding=query_embedding,
            k=body.k,
        )
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc

    return QueryResponse(
        query=body.query,
        results=[QueryResultItem(**r) for r in results],
    )
