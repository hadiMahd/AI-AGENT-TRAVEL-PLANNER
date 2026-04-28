"""
Alembic environment configuration — connects migrations to our ORM models.

Key customizations vs the default template:
1. target_metadata: set to our Base.metadata so autogenerate can detect
   schema changes from the ORM models in models/alchemy.py
2. Database URL: read from the DATABASE_URL env var (or .env file)
   instead of the hardcoded sqlalchemy.url in alembic.ini
3. pgvector Vector type: registered via render_item so Alembic can
   generate migrations that include the vector() column type
4. Async-compatible: uses the sync engine (Alembic doesn't support async),
   but reads the same DATABASE_URL that the async app uses
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from models.alchemy import Base

# Alembic Config object — provides access to values in alembic.ini
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at our ORM metadata so autogenerate can detect changes
target_metadata = Base.metadata

# Override the sqlalchemy.url from alembic.ini with the real DATABASE_URL.
# alembic.ini has a placeholder — we always read from the environment.
# This ensures migrations use the same DB as the running application.
db_url = os.getenv("DATABASE_URL", "")
if db_url:
    # Alembic's default engine is sync. Our DATABASE_URL uses asyncpg:
    #   postgresql+asyncpg://user:pass@host:5432/db
    # We replace the async driver with the sync driver (psycopg2) so
    # Alembic's synchronous engine can connect.
    sync_url = db_url.replace("+asyncpg", "")
    config.set_main_option("sqlalchemy.url", sync_url)


# -- pgvector type rendering --------------------------------------------------
# Alembic doesn't know about pgvector's Vector type by default.
# Without this, autogenerate would crash or produce incorrect migrations.
# We tell Alembic to render `Vector(N)` as a custom type string that
# the migration template can import and use.
from pgvector.sqlalchemy import Vector


def render_item(type_, obj, autogen_context):
    """Custom renderer for pgvector's Vector type in migration files."""
    if type_ == "type" and isinstance(obj, Vector):
        # Render as: sa.Vector(dim) — we'll add the import in the migration
        return f"pgvector.sqlalchemy.Vector({obj.dim})", "import pgvector.sqlalchemy"
    return None


# -----------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() emit SQL strings to the script output.
    Useful for generating migration scripts without a live DB connection.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the context.
    Uses NullPool so each migration gets a fresh connection
    (avoids connection reuse issues in long migration chains).
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_item=render_item,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
