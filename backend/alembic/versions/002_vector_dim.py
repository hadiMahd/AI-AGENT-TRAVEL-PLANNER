"""change embedding dimension from 1536 to 768

Revision ID: 002_vector_dim
Revises: 001_initial
Create Date: 2026-04-28

Migrates the documents.embedding column from vector(1536) to vector(768)
to match Gemini's text-embedding-004 output dimensionality.

Steps:
1. Nullify existing embeddings (1536-dim vectors are incompatible with 768-dim)
2. Drop the HNSW index that depends on the embedding column
3. Alter the column type from vector(1536) to vector(768)
4. Recreate the HNSW index with the new dimensionality

After running this migration, re-run POST /rag/ingest to regenerate
all embeddings using Gemini's embedding model.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_vector_dim"
down_revision: Union[str, Sequence[str], None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Change embedding dimension from 1536 (Azure) to 768 (Gemini)."""
    # Step 1: Nullify existing embeddings — 1536-dim vectors cannot
    # coexist with a vector(768) column type
    op.execute("UPDATE documents SET embedding = NULL")

    # Step 2: Drop the HNSW index — it depends on the column's type
    op.execute("DROP INDEX IF EXISTS idx_documents_embedding")

    # Step 3: Alter the column dimensionality
    op.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768)")

    # Step 4: Recreate the HNSW index for fast cosine similarity search
    op.execute(
        "CREATE INDEX idx_documents_embedding "
        "ON documents USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Revert embedding dimension from 768 (Gemini) to 1536 (Azure)."""
    op.execute("UPDATE documents SET embedding = NULL")
    op.execute("DROP INDEX IF EXISTS idx_documents_embedding")
    op.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(1536)")
    op.execute(
        "CREATE INDEX idx_documents_embedding "
        "ON documents USING hnsw (embedding vector_cosine_ops)"
    )
