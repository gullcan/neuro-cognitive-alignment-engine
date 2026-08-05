from __future__ import annotations

from pathlib import Path

import pytest

from neuro_alignment.config import Settings
from neuro_alignment.delivery import OutboxDispatcher
from neuro_alignment.domain import OutboundMessage
from neuro_alignment.storage import Database, OutboxRepository


class RecordingChannel:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.messages: list[OutboundMessage] = []

    async def send(self, message: OutboundMessage) -> str:
        self.messages.append(message)
        if len(self.messages) <= self.failures:
            raise RuntimeError("provider unavailable")
        return f"telegram-{len(self.messages)}"


@pytest.mark.asyncio
async def test_dispatcher_delivers_each_claimed_message_once(tmp_path: Path) -> None:
    database, outbox = await build_outbox(tmp_path)
    channel = RecordingChannel()
    dispatcher = OutboxDispatcher(build_settings(), outbox, channel)
    await outbox.enqueue(
        [
            OutboundMessage(
                idempotency_key="delivery-success-1",
                chat_id="12345",
                text="feedback",
            )
        ]
    )
    try:
        first, second = await dispatcher.deliver(), await dispatcher.deliver()

        assert first.model_dump() == {
            "enabled": True,
            "claimed": 1,
            "sent": 1,
            "failed": 0,
            "dead_lettered": 0,
        }
        assert second.claimed == 0
        assert len(channel.messages) == 1
        assert await outbox.pending() == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_dispatcher_dead_letters_after_max_attempts(tmp_path: Path) -> None:
    database, outbox = await build_outbox(tmp_path)
    channel = RecordingChannel(failures=10)
    settings = build_settings(outbox_max_attempts=2)
    dispatcher = OutboxDispatcher(settings, outbox, channel)
    await outbox.enqueue(
        [
            OutboundMessage(
                idempotency_key="delivery-failure-1",
                chat_id="12345",
                text="feedback",
            )
        ]
    )
    try:
        first = await dispatcher.deliver()
        assert first.failed == 1
        assert first.dead_lettered == 0
        assert len(await outbox.pending()) == 1

        second = await dispatcher.deliver()
        assert second.failed == 1
        assert second.dead_lettered == 1
        assert await outbox.pending() == []

        third = await dispatcher.deliver()
        assert third.claimed == 0
        assert len(channel.messages) == 2
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_active_delivery_lease_is_not_claimed_twice(tmp_path: Path) -> None:
    database, outbox = await build_outbox(tmp_path)
    await outbox.enqueue(
        [
            OutboundMessage(
                idempotency_key="lease-1",
                chat_id="12345",
                text="feedback",
            )
        ]
    )
    try:
        first = await outbox.claim_delivery_batch(
            limit=10,
            max_attempts=5,
            lease_seconds=120,
        )
        second = await outbox.claim_delivery_batch(
            limit=10,
            max_attempts=5,
            lease_seconds=120,
        )

        assert len(first) == 1
        assert second == []
    finally:
        await database.close()


async def build_outbox(tmp_path: Path) -> tuple[Database, OutboxRepository]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'delivery.db'}")
    await database.create_schema()
    return database, OutboxRepository(database)


def build_settings(*, outbox_max_attempts: int = 5) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        checkpoint_backend="memory",
        telegram_delivery_enabled=True,
        outbox_max_attempts=outbox_max_attempts,
    )
