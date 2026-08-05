from __future__ import annotations

import json
import traceback

import httpx
import pytest

from neuro_alignment.config import Settings
from neuro_alignment.domain import OutboundMessage
from neuro_alignment.integrations import TelegramClient, TelegramDeliveryError


@pytest.mark.asyncio
async def test_telegram_http_failure_never_exposes_bot_token() -> None:
    bot_token = "123456:super-secret-token"

    def fail_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500, request=request, json={"ok": False})

    settings = Settings(
        _env_file=None,
        app_env="test",
        telegram_delivery_enabled=True,
        telegram_bot_token=bot_token,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_request)) as http_client:
        client = TelegramClient(settings, http_client)

        with pytest.raises(TelegramDeliveryError) as captured:
            await client.send(
                OutboundMessage(
                    idempotency_key="secret-test",
                    chat_id="12345",
                    text="message",
                )
            )

    assert bot_token not in str(captured.value)
    assert "HTTPStatusError" in str(captured.value)
    rendered_traceback = "".join(traceback.format_exception(captured.value))
    assert bot_token not in rendered_traceback


@pytest.mark.asyncio
async def test_telegram_omits_empty_reply_markup() -> None:
    captured_json: dict[str, object] | None = None

    def accept_request(request: httpx.Request) -> httpx.Response:
        nonlocal captured_json
        captured_json = json.loads(request.content)
        return httpx.Response(
            status_code=200,
            request=request,
            json={"ok": True, "result": {"message_id": 42}},
        )

    settings = Settings(
        _env_file=None,
        app_env="test",
        telegram_delivery_enabled=True,
        telegram_bot_token="test-token",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(accept_request)) as http_client:
        client = TelegramClient(settings, http_client)
        message_id = await client.send(
            OutboundMessage(
                idempotency_key="empty-markup-test",
                chat_id="12345",
                text="message",
            )
        )

    assert message_id == "42"
    assert captured_json == {"chat_id": "12345", "text": "message"}
