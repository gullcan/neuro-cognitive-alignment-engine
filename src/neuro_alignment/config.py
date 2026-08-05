from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["local", "test", "production"] = "local"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    readiness_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    timezone: str = "Europe/Istanbul"
    default_user_id: str = "owner"

    database_url: str = "sqlite+aiosqlite:///./data/neuro_alignment.db"

    checkpoint_backend: Literal["memory", "sqlite", "postgres"] = "sqlite"
    checkpoint_sqlite_path: Path = Path("./data/neuro_alignment_checkpoints.db")
    checkpoint_postgres_url: str | None = None

    notion_api_token: SecretStr | None = None
    notion_data_source_id: str | None = None
    notion_verification_token: SecretStr | None = None
    notion_api_version: str = "2026-03-11"

    telegram_bot_token: SecretStr | None = None
    telegram_webhook_secret: SecretStr | None = None
    telegram_chat_id: str | None = None
    telegram_delivery_enabled: bool = False

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-terra"

    internal_api_key: SecretStr = Field(default=SecretStr("change-me"))

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def notion_configured(self) -> bool:
        return bool(self.notion_api_token and self.notion_data_source_id)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@lru_cache
def get_settings() -> Settings:
    return Settings()
