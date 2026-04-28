"""
Retriever — performs cosine similarity search against the documents table.

Retrieval pipeline (per INSTRUCTIONS.md):
  query → embed → similarity search → top-k → return results

Why raw SQL instead of SQLAlchemy ORM?
- pgvector's cosine distance operator `<=>` isn't natively supported
  by SQLAlchemy's ORM query API
- Raw SQL is transparent, debuggable, and easy to defend in evaluation
- The `<=>` operator returns cosine DISTANCE (0 = identical, 2 = opposite)
  We compute cosine SIMILARITY as `1 - distance` for intuitive scoring

k=5 rationale:
- Our corpus is ~20-25 chunks from 19 destinations
- k=5 returns ~25% of the corpus — enough coverage to find relevant
  destinations across different styles without drowning in noise
- For a larger corpus, k could be tuned higher; for our size, 5 is optimal

HNSW index (defined in db-schema.sql):
- Approximate nearest neighbor search — trades a small accuracy loss
  for significant speed improvement over exact search
- Best for small-to-medium datasets like ours (<1M vectors)
- Alternative (IVFFlat) requires training and is better for >1M vectors

Embedding dimensionality:
- Azure OpenAI text-embedding-3-small outputs 1536-dim vectors
- The documents table uses vector(1536) to match
- The SQL cast `:embedding::vector(1536)` ensures pgvector
  knows the dimensionality for the comparison
"""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Default number of results to return from similarity search
# See rationale above for why k=5
DEFAULT_K = 5

# Raw SQL for cosine similarity search using pgvector
# - `<=>` is the cosine distance operator provided by pgvector
# - `1 - (embedding <=> :embedding)` converts distance to similarity score
#   where 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
# - We cast the embedding parameter to vector(1536) so pgvector
#   knows the dimensionality for the comparison
SIMILARITY_SQL = text("""
    SELECT
        content,
        metadata,
        1 - (embedding <=> CAST(:embedding AS vector(1536))) AS score
    FROM documents
    ORDER BY embedding <=> CAST(:embedding AS vector(1536))
    LIMIT :k
""")


async def similarity_search(
    db: AsyncSession,
    query_embedding: list[float],
    k: int = DEFAULT_K,
) -> list[dict[str, Any]]:
    """
    Find the k most similar documents to the given query embedding.

    Args:
        db:             Async SQLAlchemy session (from Depends(get_db))
        query_embedding: 1536-dim float vector from embed_query()
        k:              Number of results to return (default: 5)

    Returns:
        List of dicts, each with:
          - "content":  str   — the matched chunk text
          - "metadata": dict  — source metadata (country, style, etc.)
          - "score":    float — cosine similarity (1.0 = perfect match)

    Note:
        Returns an empty list if the documents table is empty
        (e.g., if ingestion hasn't been run yet).
    """
    result = await db.execute(
        SIMILARITY_SQL,
        {"embedding": str(query_embedding), "k": k},
    )

    rows = result.fetchall()

    if not rows:
        logger.warning("No documents found — has ingestion been run?")
        return []

    # Convert Row objects to plain dicts for JSON serialization
    results = [
        {
            "content": row.content,
            "metadata": row.metadata,
            "score": float(row.score),
        }
        for row in rows
    ]

    logger.info(
        "Similarity search returned %d results (top score: %.4f)",
        len(results),
        results[0]["score"] if results else 0.0,
    )
    return results
