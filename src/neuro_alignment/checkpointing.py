from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from neuro_alignment.config import Settings

CheckpointSaver = BaseCheckpointSaver[Any]


@dataclass(slots=True)
class CheckpointManager:
    """Own exactly one LangGraph checkpointer for the application lifespan."""

    settings: Settings
    saver: CheckpointSaver | None = field(default=None, init=False)
    _exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack, init=False)

    @property
    def is_started(self) -> bool:
        return self.saver is not None

    async def start(self) -> CheckpointSaver:
        if self.saver is not None:
            return self.saver

        backend = self.settings.checkpoint_backend
        if backend == "memory":
            saver: CheckpointSaver = InMemorySaver()
        elif backend == "sqlite":
            path = self.settings.checkpoint_sqlite_path.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            sqlite_saver = await self._exit_stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(path))
            )
            await sqlite_saver.setup()
            saver = sqlite_saver
        else:
            connection_string = self.settings.checkpoint_postgres_url
            if not connection_string:
                raise ValueError(
                    "CHECKPOINT_POSTGRES_URL is required when CHECKPOINT_BACKEND=postgres."
                )
            postgres_saver = await self._exit_stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(connection_string)
            )
            await postgres_saver.setup()
            saver = postgres_saver

        self.saver = saver
        return saver

    async def close(self) -> None:
        self.saver = None
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
