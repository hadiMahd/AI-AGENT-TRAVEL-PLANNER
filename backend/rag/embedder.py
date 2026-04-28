"""
Embedding helpers — thin async wrappers around the Azure OpenAI embedding client.

The actual OpenAIEmbeddings instance is created during lifespan startup
(in llm.py) and stored on app.state.embedder. It's exposed to routes via
Depends(get_embedder) from dependencies.py.

This module only provides convenience functions that batch-embed texts
and single-embed queries. It receives the embedder instance as a parameter
rather than owning it — the lifespan is the single owner.

Why separate functions instead of calling embedder.aembed_documents directly?
- Batch splitting: Azure OpenAI has per-deployment rate limits (TPM/RPM);
  we embed in sub-batches of 16 to stay within limits
- Logging: every embed call is logged for observability
- Single place to change: if we need to add retries or fallbacks later,
  it's all here, not scattered across route handlers
"""

import logging

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# text-embedding-3-small native output dimensionality — matches vector(1536) in DB
# This model supports a `dimensions` parameter to shorten embeddings,
# but we use the default 1536-dim output for maximum retrieval quality.
EMBEDDING_DIM = 1536

# Maximum texts per API call — safe batch size for Azure OpenAI rate limits
# Azure OpenAI has per-deployment TPM/RPM limits; 16 is a conservative batch
# size that stays well within typical limits.
BATCH_SIZE = 16


async def embed_texts(
    embedder: OpenAIEmbeddings,
    texts: list[str],
) -> list[list[float]]:
    """
    Embed a batch of texts using the provided Azure OpenAI embedder.

    Called during ingestion to embed all document chunks before
    inserting them into the documents table.

    Splits texts into sub-batches of 16 to stay within Azure OpenAI's
    rate limits and avoid timeouts on large payloads.

    Args:
        embedder: The lifespan-managed OpenAIEmbeddings instance.
        texts:    List of chunk content strings to embed.

    Returns:
        List of 1536-dim float vectors, one per input text.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.debug("Embedding batch %d-%d of %d", i, i + len(batch), len(texts))
        batch_embeddings = await embedder.aembed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("Embedded %d texts into %d-dim vectors", len(texts), EMBEDDING_DIM)
    return all_embeddings


async def embed_query(
    embedder: OpenAIEmbeddings,
    query: str,
) -> list[float]:
    """
    Embed a single query string using the provided Azure OpenAI embedder.

    Called during retrieval to embed the user's query before
    performing similarity search against the documents table.

    Args:
        embedder: The lifespan-managed OpenAIEmbeddings instance.
        query:    The user's natural language query string.

    Returns:
        A single 1536-dim float vector.
    """
    embedding = await embedder.aembed_query(query)
    logger.debug("Embedded query into %d-dim vector", len(embedding))
    return embedding
