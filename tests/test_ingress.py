from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from neuro_alignment.api import create_app
from neuro_alignment.config import Settings
from neuro_alignment.domain import NotionTask, OutboundMessage
from neuro_alignment.runtime import AppServices


class StubNotionClient:
    def __init__(self, tasks: list[NotionTask]) -> None:
        self.tasks = tasks
        self.requested_dates: list[date] = []

    async def fetch_daily_tasks(self, target_date: date) -> list[NotionTask]:
        self.requested_dates.append(target_date)
        return self.tasks


class StubTelegramClient:
    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []
        self.answered_callbacks: list[str] = []

    async def send(self, message: OutboundMessage) -> str:
        self.sent.append(message)
        return f"telegram-{len(self.sent)}"

    async def answer_callback_query(
        self,
        callback_query_id: str,
        *,
        text: str = "İşleniyor…",
    ) -> None:
        self.answered_callbacks.append(callback_query_id)


@pytest.mark.asyncio
async def test_telegram_webhook_authenticates_processes_and_deduplicates(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path)
    services = await build_services(settings)
    application = create_app(settings=settings, service_builder=lambda _settings: services)
    update = {
        "update_id": 7001,
        "message": {
            "message_id": 50,
            "date": 1785906000,
            "chat": {"id": 12345},
            "from": {"id": 88},
            "text": "Bugünkü başlangıç direncimi kaydediyorum.",
        },
    }

    async with (
        application.router.lifespan_context(application),
        build_client(application) as client,
    ):
        unauthorized = await client.post("/v1/webhooks/telegram", json=update)
        assert unauthorized.status_code == 401

        unauthorized_chat_update = {
            **update,
            "update_id": 7000,
            "message": {**update["message"], "chat": {"id": 99999}},
        }
        unauthorized_chat = await client.post(
            "/v1/webhooks/telegram",
            json=unauthorized_chat_update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )
        assert unauthorized_chat.status_code == 403

        first = await client.post(
            "/v1/webhooks/telegram",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )
        assert first.status_code == 200
        assert first.json()["status"] == "checkin_recorded"
        assert first.json()["delivery"]["enabled"] is False

        duplicate = await client.post(
            "/v1/webhooks/telegram",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["status"] == "duplicate"
        assert duplicate.json()["duplicate"] is True


@pytest.mark.asyncio
async def test_telegram_task_callback_queues_feedback_and_acks_button(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path)
    services = await build_services(settings)
    telegram = StubTelegramClient()
    services.telegram = telegram  # type: ignore[assignment]
    application = create_app(settings=settings, service_builder=lambda _settings: services)
    update = {
        "update_id": 7002,
        "callback_query": {
            "id": "callback-7002",
            "from": {"id": 88},
            "message": {"chat": {"id": 12345}},
            "data": "task:started:notion-task-1",
        },
    }

    async with (
        application.router.lifespan_context(application),
        build_client(application) as client,
    ):
        response = await client.post(
            "/v1/webhooks/telegram",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-secret"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "feedback_generated"
        assert response.json()["queued_messages"] == 1
        assert telegram.answered_callbacks == ["callback-7002"]
        assert len(await services.outbox.pending()) == 1


@pytest.mark.asyncio
async def test_scheduler_requires_internal_key_and_builds_daily_plan(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    services = await build_services(settings)
    notion = StubNotionClient(
        [
            NotionTask(
                page_id="notion-task-1",
                title="Webhook entegrasyonunu tamamla",
                scheduled_date=date(2026, 8, 5),
                commitment_tier="Core",
                priority="P1",
                definition_of_done="Entegrasyon testleri geçiyor",
                minimum_action="Webhook contract testini aç",
            )
        ]
    )
    services.notion = notion  # type: ignore[assignment]
    application = create_app(settings=settings, service_builder=lambda _settings: services)
    body = {"plan_date": "2026-08-05", "request_id": "manual-2026-08-05"}

    async with (
        application.router.lifespan_context(application),
        build_client(application) as client,
    ):
        unauthorized = await client.post(
            "/v1/internal/scheduler/daily-plan",
            json=body,
            headers={"X-Internal-Api-Key": "wrong"},
        )
        assert unauthorized.status_code == 401

        response = await client.post(
            "/v1/internal/scheduler/daily-plan",
            json=body,
            headers={"X-Internal-Api-Key": "internal-secret"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "plan_created"
        assert response.json()["queued_messages"] == 1
        assert notion.requested_dates == [date(2026, 8, 5)]


@pytest.mark.asyncio
async def test_internal_outbox_endpoint_delivers_with_leased_worker(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, telegram_delivery_enabled=True)
    services = await build_services(settings)
    telegram = StubTelegramClient()
    services.telegram = telegram  # type: ignore[assignment]
    await services.outbox.enqueue(
        [
            OutboundMessage(
                idempotency_key="api-delivery-1",
                chat_id="12345",
                text="Teslim edilecek mesaj",
            )
        ]
    )
    application = create_app(settings=settings, service_builder=lambda _settings: services)

    async with (
        application.router.lifespan_context(application),
        build_client(application) as client,
    ):
        response = await client.post(
            "/v1/internal/outbox/deliver",
            json={"limit": 10},
            headers={"X-Internal-Api-Key": "internal-secret"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "enabled": True,
            "claimed": 1,
            "sent": 1,
            "failed": 0,
            "dead_lettered": 0,
        }
        assert len(telegram.sent) == 1
        assert await services.outbox.pending() == []


async def build_services(settings: Settings) -> AppServices:
    services = AppServices.build(settings)
    await services.database.create_schema()
    return services


def build_client(application: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    )


def build_settings(
    tmp_path: Path,
    *,
    telegram_delivery_enabled: bool = False,
) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'ingress.db'}",
        checkpoint_backend="memory",
        telegram_webhook_secret="telegram-secret",
        telegram_chat_id="12345",
        telegram_delivery_enabled=telegram_delivery_enabled,
        internal_api_key="internal-secret",
        openai_api_key=None,
    )
