from __future__ import annotations

import asyncio
from logging.config import fileConfig
from pathlib import Path
from typing import Any

from alembic import context
from alembic.runtime.environment import NameFilterParentNames, NameFilterType
from sqlalchemy import pool
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.ext.asyncio import create_async_engine

from neuro_alignment.config import get_settings
from neuro_alignment.storage import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
target_metadata = Base.metadata


def include_name(
    name: str | None,
    type_: NameFilterType,
    _parent_names: NameFilterParentNames,
) -> bool:
    """Limit reflection to tables owned by this application's metadata."""
    if type_ == "table":
        return name in target_metadata.tables
    return True


def include_object(
    _object: Any,
    name: str | None,
    type_: str,
    _reflected: bool,
    _compare_to: Any,
) -> bool:
    """Exclude PostgreSQL-only indexes from SQLite drift comparisons."""
    if type_ == "index" and name == "ix_behavioral_memories_embedding_hnsw":
        return context.get_context().dialect.name == "postgresql"
    return True


def is_sqlite_url(url: str) -> bool:
    return make_url(url).get_backend_name() == "sqlite"


def ensure_sqlite_parent(url: str) -> None:
    parsed_url = make_url(url)
    if parsed_url.get_backend_name() != "sqlite":
        return
    database_path = parsed_url.database
    if database_path and database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=is_sqlite_url(settings.database_url),
        include_name=include_name,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations using an established database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=connection.dialect.name == "sqlite",
        include_name=include_name,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create a temporary asynchronous engine and run migrations."""
    ensure_sqlite_parent(settings.database_url)
    connectable = create_async_engine(
        settings.database_url,
        poolclass=pool.NullPool,
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations against the configured database."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
