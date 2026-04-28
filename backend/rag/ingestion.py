"""
Document ingestion — orchestrates the full RAG ingestion pipeline.

Pipeline: load → chunk → embed → insert

1. LOAD:      Read all JSON files from the documents/ directory
2. CHUNK:     Split each document into chunks (most stay as 1 chunk
              since they're ~350 chars and chunk_size=500)
3. EMBED:     Generate 1536-dim embeddings via Azure OpenAI
              (text-embedding-3-small)
4. INSERT:    Upsert all chunks + embeddings into the documents table

Idempotency:
  Each call to ingest_documents() performs a full wipe-and-replace.
  This ensures the vector store always reflects the latest files on disk.
  No duplicate rows, no stale data.

  Why not incremental upsert?
  - Our document set is small (~19 files, ~25 chunks) — full replace is fast
  - Incremental upsert requires tracking which files changed (content hashing,
    file modification timestamps) — added complexity for no real benefit
  - Full replace is simpler to reason about and defend

Ingestion is NOT automatic on startup — it's triggered manually via
POST /rag/ingest. This follows INSTRUCTIONS.md guidance:
  "Clarify what runs once (startup/ingest) vs per request.
   Do not re-ingest on every query."
"""

import logging
from typing import TypedDict

from langchain_openai import OpenAIEmbeddings
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from models.alchemy import Document
from rag.chuncker import chunk_documents
from rag.embedder import embed_texts
from rag.load_doc import load_documents

logger = logging.getLogger(__name__)


class IngestionResult(TypedDict):
    """Structured result returned after ingestion completes."""

    documents_loaded: int
    chunks_created: int


async def ingest_documents(
    db: AsyncSession,
    embedder: OpenAIEmbeddings,
    documents_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IngestionResult:
    """
    Run the full ingestion pipeline: load → chunk → embed → insert.

    Args:
        db:             Async SQLAlchemy session (from Depends(get_db))
        embedder:       The lifespan-managed Azure OpenAI embedding client
                        (from Depends(get_embedder))
        documents_dir:  Path to the directory containing JSON doc files
        chunk_size:     Maximum characters per chunk (default: 500)
        chunk_overlap:  Overlap characters between chunks (default: 50)

    Returns:
        IngestionResult with counts of documents loaded and chunks created.

    Raises:
        FileNotFoundError: if documents_dir doesn't exist
        ValueError:        if no documents are found
        RuntimeError:      if embedding or insertion fails
    """
    # Step 1: Load all JSON documents from disk
    docs = await load_documents(documents_dir)
    if not docs:
        raise ValueError(f"No documents found in {documents_dir}")
    logger.info("Step 1/4 — Loaded %d documents from disk", len(docs))

    # Step 2: Chunk documents for embedding
    chunks = await chunk_documents(docs, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("Chunking produced no chunks — check input data")
    logger.info("Step 2/4 — Created %d chunks", len(chunks))

    # Step 3: Embed all chunk contents via Azure OpenAI
    texts = [chunk["content"] for chunk in chunks]
    embeddings = await embed_texts(embedder, texts)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {len(embeddings)} embeddings "
            f"for {len(chunks)} chunks"
        )
    logger.info("Step 3/4 — Generated %d embeddings", len(embeddings))

    # Step 4: Wipe existing data and insert new chunks
    # Idempotent: full wipe-and-replace ensures no duplicates or stale data
    await db.execute(delete(Document))
    logger.info("Step 4/4 — Cleared existing documents table")

    # Build ORM objects for bulk insert
    db_objects = []
    for chunk, embedding in zip(chunks, embeddings):
        db_objects.append(
            Document(
                content=chunk["content"],
                metadata_=chunk["metadata"],
                embedding=embedding,
            )
        )

    # add_all + commit in one transaction — if anything fails, nothing is inserted
    db.add_all(db_objects)
    await db.commit()

    logger.info(
        "Ingestion complete — %d documents loaded, %d chunks inserted",
        len(docs),
        len(chunks),
    )

    return IngestionResult(
        documents_loaded=len(docs),
        chunks_created=len(chunks),
    )
