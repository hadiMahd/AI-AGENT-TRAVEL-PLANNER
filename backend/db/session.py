from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import get_settings

_engine = None
_async_session_local = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _async_session_local
    if _async_session_local is None:
        _async_session_local = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_local


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    session_factory = _get_session_factory()
    async with session_factory() as session:
        yield session


async def init_db() -> None:
    _get_engine()


async def dispose_db() -> None:
    global _engine, _async_session_local
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_local = None


async def check_db_health() -> bool:
    engine = _get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
