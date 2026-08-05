from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator, model_validator
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

    outbox_batch_size: int = Field(default=20, ge=1, le=100)
    outbox_max_attempts: int = Field(default=5, ge=1, le=20)
    outbox_lease_seconds: int = Field(default=120, ge=10, le=3600)

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-terra"

    internal_api_key: SecretStr = Field(default=SecretStr("change-me"))

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        """Select psycopg's async SQLAlchemy dialect for platform Postgres URLs."""
        if isinstance(value, str) and value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        if isinstance(value, str) and value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        return value

    @field_validator("telegram_webhook_secret")
    @classmethod
    def validate_telegram_webhook_secret(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if not 1 <= len(secret) <= 256 or re.fullmatch(r"[A-Za-z0-9_-]+", secret) is None:
            raise ValueError(
                "TELEGRAM_WEBHOOK_SECRET must contain 1-256 letters, digits, "
                "underscores, or hyphens."
            )
        return value

    @field_validator(
        "notion_api_token",
        "notion_verification_token",
        "telegram_bot_token",
        "telegram_webhook_secret",
        "openai_api_key",
        mode="before",
    )
    @classmethod
    def normalize_empty_secret(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_production_safety(self) -> Settings:
        if self.app_env != "production":
            return self
        if self.checkpoint_backend != "postgres":
            raise ValueError("Production requires CHECKPOINT_BACKEND=postgres.")
        if self.postgres_checkpoint_url is None:
            raise ValueError(
                "Production requires CHECKPOINT_POSTGRES_URL or a PostgreSQL DATABASE_URL."
            )
        if self.internal_api_key.get_secret_value() == "change-me":
            raise ValueError("Production requires a non-default INTERNAL_API_KEY.")
        if self.telegram_delivery_enabled and not self.telegram_bot_token:
            raise ValueError("Telegram delivery requires TELEGRAM_BOT_TOKEN.")
        if self.telegram_bot_token and not self.telegram_webhook_secret:
            raise ValueError("Telegram webhooks require TELEGRAM_WEBHOOK_SECRET.")
        return self

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def notion_configured(self) -> bool:
        return bool(self.notion_api_token and self.notion_data_source_id)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def postgres_checkpoint_url(self) -> str | None:
        """Return psycopg-compatible checkpoint connection information.

        A dedicated URL remains supported, but small deployments can safely reuse the
        operational PostgreSQL database without duplicating one secret in the platform UI.
        """
        connection_string = self.checkpoint_postgres_url
        if connection_string is None and self.database_url.startswith("postgresql+psycopg://"):
            connection_string = self.database_url
        if connection_string is None:
            return None
        return connection_string.replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
