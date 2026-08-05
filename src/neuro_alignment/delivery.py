from __future__ import annotations

import asyncio
from typing import Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field

from neuro_alignment.config import Settings
from neuro_alignment.domain import OutboundMessage
from neuro_alignment.storage import OutboxRepository

logger = structlog.get_logger()


class MessageDeliveryChannel(Protocol):
    async def send(self, message: OutboundMessage) -> str: ...


class OutboxDeliveryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    claimed: int = Field(ge=0)
    sent: int = Field(ge=0)
    failed: int = Field(ge=0)
    dead_lettered: int = Field(ge=0)


class OutboxDispatcher:
    """Lease and deliver transactional outbox records through Telegram."""

    def __init__(
        self,
        settings: Settings,
        outbox: OutboxRepository,
        channel: MessageDeliveryChannel,
    ) -> None:
        self.settings = settings
        self.outbox = outbox
        self.channel = channel
        self._local_worker_lock = asyncio.Lock()

    async def deliver(self, *, limit: int | None = None) -> OutboxDeliveryReport:
        if not self.settings.telegram_delivery_enabled:
            return OutboxDeliveryReport(
                enabled=False,
                claimed=0,
                sent=0,
                failed=0,
                dead_lettered=0,
            )

        batch_limit = limit or self.settings.outbox_batch_size
        if not 1 <= batch_limit <= 100:
            raise ValueError("Outbox delivery limit must be between 1 and 100.")

        async with self._local_worker_lock:
            records = await self.outbox.claim_delivery_batch(
                limit=batch_limit,
                max_attempts=self.settings.outbox_max_attempts,
                lease_seconds=self.settings.outbox_lease_seconds,
            )
            sent = 0
            failed = 0
            dead_lettered = 0
            for record_id, message in records:
                try:
                    provider_message_id = await self.channel.send(message)
                except Exception as error:
                    failed += 1
                    exhausted = await self.outbox.mark_failed(
                        record_id,
                        error,
                        max_attempts=self.settings.outbox_max_attempts,
                    )
                    dead_lettered += int(exhausted)
                    await logger.aerror(
                        "outbox_delivery_failed",
                        record_id=record_id,
                        idempotency_key=message.idempotency_key,
                        error_type=type(error).__name__,
                        exhausted=exhausted,
                    )
                else:
                    await self.outbox.mark_sent(record_id, provider_message_id)
                    sent += 1

        return OutboxDeliveryReport(
            enabled=True,
            claimed=len(records),
            sent=sent,
            failed=failed,
            dead_lettered=dead_lettered,
        )
