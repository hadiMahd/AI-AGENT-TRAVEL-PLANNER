"""
Document chunker — splits loaded documents into smaller chunks for embedding.

Why chunking?
- Embedding models have token limits; chunking keeps each piece within bounds
- Smaller chunks improve retrieval precision — the LLM gets the most relevant
  snippet instead of a long document that may cover multiple topics
- Overlap between chunks prevents information loss at chunk boundaries

Chunking rationale for this project:
- 19 destination docs, each ~300-400 characters
- chunk_size=500 means most docs fit in a SINGLE chunk (no split needed)
- chunk_overlap=50 provides continuity for the rare doc exceeding 500 chars
- No meaningful information loss since RecursiveCharacterTextSplitter
  respects sentence/paragraph boundaries
- Result: ~20-25 total chunks from 19 documents

Uses LangChain's RecursiveCharacterTextSplitter which splits on
["\\n\\n", "\\n", ". ", " ", ""] — prioritizing natural text boundaries.
"""

import logging

import anyio
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


async def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split loaded documents into chunks for embedding.

    Args:
        documents:     List of dicts from load_documents(), each with
                       "content" (str) and "metadata" (dict)
        chunk_size:    Maximum characters per chunk.
                       Default: 500 (from RAG_CHUNK_SIZE setting)
        chunk_overlap: Overlap characters between consecutive chunks.
                       Default: 50 (from RAG_CHUNK_OVERLAP setting)

    Returns:
        List of dicts, each with:
          - "content":  str  — the chunk text
          - "metadata": dict — inherited from source doc + "chunk_index" added
    """
    # Initialize the splitter with the configured parameters
    # RecursiveCharacterTextSplitter tries splitting on double-newline,
    # then single newline, then ". ", then " ", then character-by-character
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        # RecursiveCharacterTextSplitter.split_text is synchronous,
        # so we offload to a thread to stay async-safe
        splits = await anyio.to_thread.run_sync(splitter.split_text, content)

        for idx, split_text in enumerate(splits):
            # Each chunk inherits the source doc's metadata
            # plus a chunk_index for traceability back to the original doc
            chunk_meta = dict(metadata)
            chunk_meta["chunk_index"] = idx

            chunks.append({"content": split_text, "metadata": chunk_meta})

    logger.info(
        "Chunked %d documents into %d chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
