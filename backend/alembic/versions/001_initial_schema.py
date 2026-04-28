"""initial schema — baseline matching db-schema.sql

Revision ID: 001_initial
Revises: None
Create Date: 2026-04-28

This is a baseline migration that represents the schema as created
by db-schema.sql. If the DB already has all tables (from the Docker
init script), mark this as applied with:
    alembic stamp 001_initial

Do NOT run `alembic upgrade head` on a DB that was initialized by
db-schema.sql — the tables already exist and this migration would fail.
Only run it on a completely empty database.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables and indexes matching db-schema.sql."""
    # Enable required extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')

    # 1. Users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # 2. Agent Runs
    op.create_table(
        "agent_runs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("response", sa.Text),
        sa.Column("prompt_tokens", sa.Integer, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(10, 6), server_default="0.0"),
        sa.Column("status", sa.String(50), server_default="completed"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_agent_runs_user_id", "agent_runs", ["user_id"])
    op.create_index("idx_agent_runs_created_at", "agent_runs", ["created_at"])

    # 3. Tool Logs
    op.create_table(
        "tool_logs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("input_payload", postgresql.JSONB),
        sa.Column("output_payload", postgresql.JSONB),
        sa.Column("status", sa.String(50), server_default="success"),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_tool_logs_run_id", "tool_logs", ["run_id"])

    # 4. Documents (RAG + pgvector) — using vector(1536) for the baseline
    # since the existing DB was created with 1536-dim embeddings
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1536)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute(
        "CREATE INDEX idx_documents_embedding ON documents USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX idx_documents_metadata ON documents USING gin (metadata)"
    )

    # 5. Delivery Logs
    op.create_table(
        "delivery_logs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("recipient_email", sa.String(255), nullable=False),
        sa.Column("subject", sa.String(255)),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_message", sa.Text),
        sa.Column("provider_message_id", sa.String(255)),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_delivery_logs_run_id", "delivery_logs", ["run_id"])
    op.create_index("idx_delivery_logs_user_id", "delivery_logs", ["user_id"])


def downgrade() -> None:
    """Drop all tables (reverse of initial schema creation)."""
    op.drop_table("delivery_logs")
    op.drop_table("documents")
    op.drop_table("tool_logs")
    op.drop_table("agent_runs")
    op.drop_table("users")
