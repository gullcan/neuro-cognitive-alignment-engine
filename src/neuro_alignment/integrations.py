from __future__ import annotations

import hashlib
import hmac
from datetime import date, datetime
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import httpx
import structlog

from neuro_alignment.config import Settings
from neuro_alignment.domain import (
    EventSource,
    InboundEventType,
    NormalizedInboundEvent,
    NotionTask,
    OutboundMessage,
    TaskAction,
)

logger = structlog.get_logger()


class IntegrationConfigurationError(RuntimeError):
    pass


class NotionSchemaError(RuntimeError):
    pass


class TelegramDeliveryError(RuntimeError):
    """A sanitized Telegram failure that never includes the bot-token URL."""


class NotionClient:
    REQUIRED_PROPERTIES: ClassVar[set[str]] = {
        "Task",
        "Window",
        "Status",
        "Commitment Tier",
        "Priority",
        "Definition of Done",
        "Minimum Action",
    }

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.http_client = http_client

    async def fetch_daily_tasks(self, target_date: date) -> list[NotionTask]:
        if not self.settings.notion_configured:
            raise IntegrationConfigurationError(
                "NOTION_API_TOKEN and NOTION_DATA_SOURCE_ID are required."
            )
        token = self.settings.notion_api_token
        assert token is not None
        assert self.settings.notion_data_source_id is not None

        endpoint = (
            f"https://api.notion.com/v1/data_sources/{self.settings.notion_data_source_id}/query"
        )
        headers = {
            "Authorization": f"Bearer {token.get_secret_value()}",
            "Notion-Version": self.settings.notion_api_version,
            "Content-Type": "application/json",
        }
        query = self._daily_query(target_date)
        response = await self.http_client.post(endpoint, headers=headers, json=query)
        response.raise_for_status()
        payload = response.json()
        tasks = [self._parse_page(page, target_date) for page in payload.get("results", [])]

        while payload.get("has_more") and payload.get("next_cursor"):
            next_query = {**query, "start_cursor": payload["next_cursor"]}
            response = await self.http_client.post(endpoint, headers=headers, json=next_query)
            response.raise_for_status()
            payload = response.json()
            tasks.extend(self._parse_page(page, target_date) for page in payload.get("results", []))
        return tasks

    @staticmethod
    def _daily_query(target_date: date) -> dict[str, Any]:
        """Build one immutable query contract reused across every result page."""
        return {
            "filter": {
                "and": [
                    {
                        "property": "Window",
                        "date": {"equals": target_date.isoformat()},
                    },
                    {
                        "property": "Status",
                        "status": {"does_not_equal": "Archived"},
                    },
                ]
            },
            "sorts": [
                {"property": "Window", "direction": "ascending"},
                {"property": "Priority", "direction": "ascending"},
            ],
            "page_size": 100,
        }

    def _parse_page(self, page: dict[str, Any], target_date: date) -> NotionTask:
        properties = page.get("properties", {})
        missing = sorted(self.REQUIRED_PROPERTIES - properties.keys())
        if missing:
            raise NotionSchemaError(
                "Notion Commitments schema is missing properties: " + ", ".join(missing)
            )

        window = properties["Window"].get("date") or {}
        start_raw = window.get("start")
        end_raw = window.get("end")
        scheduled_start = self._parse_datetime(start_raw)
        scheduled_end = self._parse_datetime(end_raw)

        evidence_raw = properties.get("Evidence", {}).get("url")
        return NotionTask(
            page_id=page["id"],
            title=self._rich_text(properties["Task"].get("title", [])) or "Untitled",
            scheduled_date=target_date,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            status=self._option_name(properties["Status"], "status") or "Planned",
            commitment_tier=(
                self._option_name(properties["Commitment Tier"], "select") or "Flexible"
            ),
            priority=self._option_name(properties["Priority"], "select") or "P2",
            project_ids=[
                relation["id"] for relation in properties.get("Project", {}).get("relation", [])
            ],
            definition_of_done=self._rich_text(
                properties["Definition of Done"].get("rich_text", [])
            ),
            minimum_action=self._rich_text(properties["Minimum Action"].get("rich_text", [])),
            estimated_minutes=properties.get("Estimated Minutes", {}).get("number"),
            cognitive_load=properties.get("Cognitive Load", {}).get("number"),
            context_cue=self._context_value(properties.get("Context Cue", {})),
            evidence_required=bool(properties.get("Evidence Required", {}).get("checkbox", False)),
            evidence_url=evidence_raw,
            skip_reason=self._option_name(
                properties.get("Skip Reason", {}),
                "select",
            ),
            last_edited_time=self._parse_datetime(page.get("last_edited_time")),
        )

    @staticmethod
    def _rich_text(items: list[dict[str, Any]]) -> str:
        return "".join(item.get("plain_text", "") for item in items).strip()

    @staticmethod
    def _option_name(prop: dict[str, Any], option_type: str) -> str | None:
        option = prop.get(option_type)
        return option.get("name") if option else None

    @classmethod
    def _context_value(cls, prop: dict[str, Any]) -> str:
        if prop.get("type") == "select":
            return cls._option_name(prop, "select") or ""
        return cls._rich_text(prop.get("rich_text", []))

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value or "T" not in value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


class TelegramUpdateParser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def normalize(self, update: dict[str, Any]) -> NormalizedInboundEvent:
        update_id = str(update["update_id"])
        callback = update.get("callback_query")
        if callback:
            chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
            self._validate_chat(chat_id)
            action, task_id, metadata = self._parse_callback(callback.get("data", ""))
            return NormalizedInboundEvent(
                event_id=update_id,
                event_type=InboundEventType.TELEGRAM_ACTION,
                source=EventSource.TELEGRAM,
                user_id=self.settings.default_user_id,
                occurred_at=datetime.now(self.settings.tz),
                task_id=task_id,
                action=action,
                payload={
                    "callback_query_id": callback.get("id"),
                    "chat_id": chat_id,
                    "telegram_user_id": callback.get("from", {}).get("id"),
                    **metadata,
                },
            )

        message = update.get("message")
        if message:
            chat_id = str(message.get("chat", {}).get("id", ""))
            self._validate_chat(chat_id)
            occurred_at = datetime.fromtimestamp(
                message.get("date", int(datetime.now().timestamp())),
                tz=ZoneInfo("UTC"),
            )
            return NormalizedInboundEvent(
                event_id=update_id,
                event_type=InboundEventType.TELEGRAM_MESSAGE,
                source=EventSource.TELEGRAM,
                user_id=self.settings.default_user_id,
                occurred_at=occurred_at,
                text=message.get("text", ""),
                payload={
                    "message_id": message.get("message_id"),
                    "chat_id": chat_id,
                    "telegram_user_id": message.get("from", {}).get("id"),
                },
            )

        raise ValueError("Unsupported Telegram update type.")

    def _validate_chat(self, chat_id: str) -> None:
        configured = self.settings.telegram_chat_id
        if configured and chat_id != configured:
            raise PermissionError("Telegram chat is not authorized.")

    @staticmethod
    def _parse_callback(data: str) -> tuple[TaskAction, str | None, dict[str, str]]:
        parts = data.split(":", maxsplit=2)
        if len(parts) != 3:
            raise ValueError("Malformed Telegram callback data.")
        scope, action_name, reference = parts

        if scope == "plan":
            action_map = {
                "approve": TaskAction.PLAN_APPROVED,
                "reject": TaskAction.PLAN_REJECTED,
            }
            if action_name not in action_map:
                raise ValueError("Unsupported plan action.")
            return action_map[action_name], None, {"approval_token": reference}

        if scope == "task":
            action_map = {
                "started": TaskAction.STARTED,
                "completed": TaskAction.COMPLETED,
                "blocked": TaskAction.BLOCKED,
                "skipped": TaskAction.SKIPPED,
                "rescheduled": TaskAction.RESCHEDULED,
            }
            if action_name not in action_map:
                raise ValueError("Unsupported task action.")
            return action_map[action_name], reference, {}

        raise ValueError("Unsupported callback scope.")


class TelegramClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.http_client = http_client

    async def send(self, message: OutboundMessage) -> str:
        if not self.settings.telegram_delivery_enabled:
            await logger.ainfo(
                "telegram_delivery_skipped",
                idempotency_key=message.idempotency_key,
                chat_id=message.chat_id,
            )
            return f"dry-run:{message.idempotency_key}"

        if not self.settings.telegram_bot_token:
            raise IntegrationConfigurationError("TELEGRAM_BOT_TOKEN is required.")

        reply_markup: dict[str, Any] | None = None
        if message.buttons:
            reply_markup = {
                "inline_keyboard": [
                    [button.model_dump(mode="json") for button in row] for row in message.buttons
                ]
            }
        request_body: dict[str, Any] = {
            "chat_id": message.chat_id,
            "text": message.text,
        }
        if reply_markup is not None:
            request_body["reply_markup"] = reply_markup
        payload = await self._post("sendMessage", request_body)
        result = payload.get("result")
        if not isinstance(result, dict) or "message_id" not in result:
            raise TelegramDeliveryError("Telegram sendMessage returned no message_id.")
        return str(result["message_id"])

    async def answer_callback_query(
        self,
        callback_query_id: str,
        *,
        text: str = "İşleniyor…",
    ) -> None:
        """Stop Telegram's inline-button progress indicator promptly."""
        if not self.settings.telegram_delivery_enabled:
            await logger.ainfo(
                "telegram_callback_answer_skipped",
                callback_query_id=callback_query_id,
            )
            return
        await self._post(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query_id,
                "text": text[:200],
                "show_alert": False,
            },
        )

    async def _post(self, method: str, json_body: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.telegram_bot_token:
            raise IntegrationConfigurationError("TELEGRAM_BOT_TOKEN is required.")
        token = self.settings.telegram_bot_token.get_secret_value()
        try:
            response = await self.http_client.post(
                f"https://api.telegram.org/bot{token}/{method}",
                json=json_body,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise TelegramDeliveryError(
                f"Telegram {method} request failed ({type(error).__name__})."
            ) from None
        try:
            payload = response.json()
        except ValueError as error:
            raise TelegramDeliveryError(
                f"Telegram {method} returned an invalid response."
            ) from error
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise TelegramDeliveryError(f"Telegram rejected {method}.")
        return payload


def verify_notion_signature(
    raw_body: bytes,
    received_signature: str | None,
    verification_token: str | None,
) -> bool:
    if not verification_token:
        return False
    if not received_signature:
        return False
    digest = hmac.new(
        verification_token.encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    expected = f"sha256={digest}"
    return hmac.compare_digest(expected, received_signature)


def verify_telegram_secret(received: str | None, expected: str | None) -> bool:
    if not expected:
        return False
    if not received:
        return False
    return hmac.compare_digest(received, expected)
