"""
SQLAlchemy ORM models for the Smart Travel Planner database.

Maps to the tables defined in db-schema.sql:
- users:         authentication & scoping (all data scoped per user)
- agent_runs:    who asked what, the answer, token cost, status
- tool_logs:     which tools fired, payloads, latency per run
- documents:     RAG knowledge chunks with pgvector embeddings
- delivery_logs: Resend/email audit trail per run

All models inherit from Base which provides a shared metadata registry
used by Alembic for autogenerate support.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models — provides a shared metadata registry."""

    pass


class User(Base):
    """
    Auth & scoping — every data row in the system is scoped to a user.

    - email:          unique login identifier
    - password_hash:  bcrypt/argon2 hash (never store plaintext)
    - created_at:     registration timestamp
    - updated_at:     last profile update timestamp
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    created_at: Mapped[str | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[str | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class AgentRun(Base):
    """
    Who asked what, the agent's answer, token cost, and run status.

    - query:              the user's natural language query
    - response:           the agent's final synthesized answer
    - prompt_tokens:      input tokens consumed (for cost tracking)
    - completion_tokens:  output tokens generated
    - cost_usd:           calculated cost per query (per INSTRUCTIONS.md)
    - status:             'completed', 'failed', etc.
    """

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
    )
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    response: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(
        default=0,
    )
    completion_tokens: Mapped[int | None] = mapped_column(
        default=0,
    )
    cost_usd: Mapped[float | None] = mapped_column(
        Numeric(10, 6),
        default=0.0,
    )
    status: Mapped[str | None] = mapped_column(
        String(50),
        default="completed",
    )
    created_at: Mapped[str | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )


class ToolLog(Base):
    """
    Which tools fired during an agent run, their inputs/outputs, and latency.

    - tool_name:       e.g., "rag_retriever", "ml_predictor", "weather_fetcher"
    - input_payload:   JSONB — the tool's Pydantic-validated input
    - output_payload:  JSONB — the tool's structured output
    - status:          'success' or 'error' (tool errors go to LLM, not raised)
    - latency_ms:      wall-clock time for the tool call (for perf tracking)
    """

    __tablename__ = "tool_logs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    input_payload: Mapped[dict | None] = mapped_column(JSONB)
    output_payload: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str | None] = mapped_column(
        String(50),
        default="success",
    )
    latency_ms: Mapped[int | None] = mapped_column()
    created_at: Mapped[str | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Document(Base):
    """
    RAG knowledge store — each row is a text chunk with its embedding.

    Schema mirrors the `documents` table in db-schema.sql:
    - id:          auto-generated UUID primary key
    - content:     the text chunk that will be retrieved and fed to the LLM
    - metadata:    JSONB blob carrying source info (country, style, continent, etc.)
                   plus chunk-level metadata (chunk_index, source_file)
    - embedding:   1536-dim vector from Azure OpenAI text-embedding-3-small
    - created_at:  timestamp for audit / staleness checks

    Indexes (defined in db-schema.sql, not here):
    - HNSW on `embedding` for fast cosine similarity search
    - GIN  on `metadata` for fast JSONB key/value filtering
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        server_default="{}",
    )
    embedding: Mapped[list | None] = mapped_column(
        Vector(1536),
        nullable=True,
        comment="1536-dim embedding from Azure OpenAI text-embedding-3-small",
    )
    created_at: Mapped[str | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class DeliveryLog(Base):
    """
    Resend/Email audit trail — tracks webhook/email delivery per agent run.

    - recipient_email:     who received the plan
    - subject:            email subject line
    - status:             'sent', 'failed', 'bounced', etc.
    - error_message:      populated on failure (for debugging)
    - provider_message_id: Resend's message ID (for tracking in their dashboard)
    - latency_ms:         time from send request to provider response
    """

    __tablename__ = "delivery_logs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
    )
    recipient_email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    subject: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    provider_message_id: Mapped[str | None] = mapped_column(String(255))
    latency_ms: Mapped[int | None] = mapped_column()
    created_at: Mapped[str | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
