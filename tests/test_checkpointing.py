from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from neuro_alignment.checkpointing import CheckpointManager
from neuro_alignment.config import Settings


class ProbeState(TypedDict):
    value: int


@pytest.mark.asyncio
async def test_sqlite_checkpoint_survives_manager_restart(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        checkpoint_backend="sqlite",
        checkpoint_sqlite_path=tmp_path / "checkpoints.db",
    )
    builder = StateGraph(ProbeState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    config = {"configurable": {"thread_id": "sqlite-restart-probe"}}

    first_manager = CheckpointManager(settings)
    first_saver = await first_manager.start()
    first_graph = builder.compile(checkpointer=first_saver)
    assert await first_graph.ainvoke({"value": 9}, config) == {"value": 10}
    await first_manager.close()

    second_manager = CheckpointManager(settings)
    second_saver = await second_manager.start()
    second_graph = builder.compile(checkpointer=second_saver)
    snapshot = await second_graph.aget_state(config)
    assert snapshot.values == {"value": 10}
    await second_manager.close()


@pytest.mark.asyncio
async def test_postgres_checkpoint_requires_explicit_connection_string() -> None:
    manager = CheckpointManager(
        Settings(
            _env_file=None,
            app_env="test",
            checkpoint_backend="postgres",
            checkpoint_postgres_url=None,
        )
    )

    with pytest.raises(ValueError, match="CHECKPOINT_POSTGRES_URL or a PostgreSQL DATABASE_URL"):
        await manager.start()

    await manager.close()


def increment(state: ProbeState) -> dict[str, int]:
    return {"value": state["value"] + 1}
