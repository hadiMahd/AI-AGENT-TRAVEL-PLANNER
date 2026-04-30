"""
RAG retriever tool — wraps the RAG retrieval pipeline as a LangGraph tool.

This tool takes a natural language query, embeds it, and performs cosine
similarity search against the documents table to find relevant travel
destination information.

The tool wraps rag/embedder.py + rag/retriever.py — the raw retrieval
logic lives there, this file just adds Pydantic validation + tool metadata.

Per INSTRUCTIONS.md: "Pydantic validation on every tool input. Invalid →
structured error → LLM retry. Never crash."
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from rag.embedder import embed_query
from rag.retriever import similarity_search

logger = logging.getLogger(__name__)


class RAGToolInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language query about travel destinations",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return",
    )


@tool(args_schema=RAGToolInput)
async def rag_retriever(
    query: str,
    k: int = 3,
    embedder: OpenAIEmbeddings | None = None,
    db: AsyncSession | None = None,
) -> str:
    """
    Search the travel destination knowledge base for relevant information.

    Use this tool when the user asks about destinations, travel styles,
    countries, or any travel-related facts. Returns the top-k most
    similar document chunks with their metadata (country, style, lat/long).

    The results include latitude and longitude in metadata — these can be
    passed to the weather_fetcher tool.
    """
    if embedder is None or db is None:
        return "[rag_retriever ERROR] embedder and db session are required"

    try:
        query_embedding = await embed_query(embedder, query)
        results = await similarity_search(db, query_embedding, k)

        if not results:
            return "[rag_retriever] No documents found — has ingestion been run? Run POST /rag/ingest first."

        formatted = []
        for r in results:
            entry = {
                "content": r["content"][:300],
                "metadata": r["metadata"],
                "score": round(r["score"], 3),
            }
            formatted.append(entry)

        return json.dumps(formatted, indent=2)

    except Exception as exc:
        logger.error("rag_retriever failed: %s", exc)
        return f"[rag_retriever ERROR] {exc}"
