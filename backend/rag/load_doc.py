"""
Document loader — reads destination JSON files from the documents/ directory.

Each JSON file has the shape:
    {
      "text": "Albania is a budget-friendly destination...",
      "metadata": {
        "country": "Albania",
        "continent": "Europe",
        "style": "budget",
        "latitude": 41.1533,
        "longitude": 20.1683
      }
    }

This module:
1. Globs all *.json files in the configured documents directory
2. Reads and parses each file
3. Returns a list of dicts with "content" and "metadata" keys,
   ready for the chunker

The "text" field from JSON is mapped to "content" to match
the documents table column name.
"""

import json
import logging
from pathlib import Path

import anyio

logger = logging.getLogger(__name__)


async def load_documents(documents_dir: str) -> list[dict]:
    """
    Load all JSON documents from the given directory.

    Args:
        documents_dir: Absolute or relative path to the folder
                       containing destination JSON files.
                       - Local dev:  "../documents" (relative to backend/)
                       - Docker:      "/app/documents" (mounted volume)

    Returns:
        A list of dicts, each with:
          - "content":  str  — the destination description text
          - "metadata": dict — source metadata + "source_file" added

    Raises:
        FileNotFoundError: if the directory does not exist
    """
    docs_path = Path(documents_dir).resolve()

    if not docs_path.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_path}")

    # Glob all JSON files — sorted for deterministic order
    json_files = sorted(docs_path.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", docs_path)
        return []

    logger.info("Found %d JSON files in %s", len(json_files), docs_path)

    documents: list[dict] = []

    for json_file in json_files:
        # Use anyio.to_thread.run_sync because json.load + Path.read_text
        # are synchronous I/O — we offload to a thread to stay async-safe
        raw = await anyio.to_thread.run_sync(lambda f=json_file: f.read_text(encoding="utf-8"))
        data = json.loads(raw)

        # Extract the text field and rename to "content"
        # to match the documents table column name
        content = data.get("text", "")
        if not content:
            logger.warning("Skipping %s — no 'text' field", json_file.name)
            continue

        # Copy metadata and inject the source filename for traceability
        metadata = dict(data.get("metadata", {}))
        metadata["source_file"] = json_file.name

        documents.append({"content": content, "metadata": metadata})

    logger.info("Loaded %d documents from disk", len(documents))
    return documents
