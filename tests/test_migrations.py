from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pytest import MonkeyPatch
from sqlalchemy import create_engine, inspect, text

from neuro_alignment.config import get_settings

PROJECT_ROOT = Path(__file__).parents[1]
BASE_REVISION = "e4d7a9136c84"
HEAD_REVISION = "7a2f4c9d1e30"
APPLICATION_TABLES = {
    "daily_plans",
    "domain_events",
    "inbound_events",
    "outbox",
    "behavioral_memories",
}


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def alembic_config(*, output_buffer: StringIO | None = None) -> Config:
    return Config(
        file_=PROJECT_ROOT / "alembic.ini",
        toml_file=PROJECT_ROOT / "pyproject.toml",
        output_buffer=output_buffer,
    )


def use_database(monkeypatch: MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()


def test_revision_graph_has_one_head() -> None:
    script = ScriptDirectory.from_config(alembic_config())

    assert script.get_heads() == [HEAD_REVISION]
    assert script.get_base() == BASE_REVISION


def test_sqlite_upgrade_check_and_downgrade_round_trip(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    database_path = tmp_path / "nested" / "migration.db"
    use_database(monkeypatch, f"sqlite+aiosqlite:///{database_path}")
    config = alembic_config()

    command.upgrade(config, BASE_REVISION)

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        inspector = inspect(engine)
        assert "ix_domain_events_user_task_occurred_at" not in {
            index["name"] for index in inspector.get_indexes("domain_events")
        }
        assert "ix_outbox_status_created_at" not in {
            index["name"] for index in inspector.get_indexes("outbox")
        }
    finally:
        engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        inspector = inspect(engine)
        assert set(inspector.get_table_names()) == APPLICATION_TABLES | {"alembic_version"}
        assert inspector.get_pk_constraint("daily_plans")["constrained_columns"] == ["id"]
        assert {index["name"] for index in inspector.get_indexes("domain_events")} >= {
            "ix_domain_events_user_task_occurred_at"
        }
        assert {index["name"] for index in inspector.get_indexes("outbox")} >= {
            "ix_outbox_status_created_at"
        }
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                HEAD_REVISION
            )
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE checkpoints (thread_id TEXT PRIMARY KEY)"))
    finally:
        engine.dispose()

    command.check(config)
    command.upgrade(config, "head")
    command.check(config)
    command.downgrade(config, "base")

    engine = create_engine(f"sqlite:///{database_path}")
    try:
        remaining_tables = set(inspect(engine).get_table_names())
        assert not remaining_tables.intersection(APPLICATION_TABLES)
        assert "checkpoints" in remaining_tables
    finally:
        engine.dispose()

    command.upgrade(config, "head")


def test_postgresql_offline_ddl_uses_jsonb_and_query_indexes(
    monkeypatch: MonkeyPatch,
) -> None:
    use_database(
        monkeypatch,
        "postgresql+psycopg://migration_user:migration_pass@localhost/migration_test",
    )
    output = StringIO()

    command.upgrade(alembic_config(output_buffer=output), "head", sql=True)

    ddl = output.getvalue()
    assert ddl.count(" TYPE JSONB USING ") == 4
    assert "ALTER COLUMN plan TYPE JSONB USING plan::jsonb" in ddl
    assert "ALTER COLUMN payload TYPE JSONB USING payload::jsonb" in ddl
    assert "ALTER COLUMN buttons TYPE JSONB USING buttons::jsonb" in ddl
    assert "CREATE INDEX ix_domain_events_user_task_occurred_at" in ddl
    assert "CREATE INDEX ix_outbox_status_created_at" in ddl
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert "embedding VECTOR(32) NOT NULL" in ddl
    assert "CREATE INDEX ix_behavioral_memories_embedding_hnsw" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl

    downgrade_output = StringIO()
    command.downgrade(
        alembic_config(output_buffer=downgrade_output),
        f"{HEAD_REVISION}:{BASE_REVISION}",
        sql=True,
    )

    downgrade_ddl = downgrade_output.getvalue()
    assert downgrade_ddl.count(" TYPE JSON USING ") == 4
    assert "ALTER COLUMN plan TYPE JSON USING plan::json" in downgrade_ddl
