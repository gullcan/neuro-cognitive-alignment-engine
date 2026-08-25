from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuro_alignment.config import Settings


def test_empty_optional_secrets_are_normalized_to_none() -> None:
    settings = Settings(
        _env_file=None,
        notion_api_token="",
        telegram_bot_token="",
        telegram_webhook_secret="",
        openai_api_key="",
        groq_api_key="",
    )

    assert settings.notion_api_token is None
    assert settings.telegram_bot_token is None
    assert settings.telegram_webhook_secret is None
    assert settings.openai_api_key is None
    assert settings.groq_api_key is None


def test_telegram_webhook_secret_rejects_unsupported_characters() -> None:
    with pytest.raises(ValidationError, match="TELEGRAM_WEBHOOK_SECRET"):
        Settings(
            _env_file=None,
            telegram_webhook_secret="contains spaces",
        )


def test_render_postgres_url_uses_async_psycopg_dialect() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://user:password@database:5432/app",
    )

    assert settings.database_url == "postgresql+psycopg://user:password@database:5432/app"


def test_explicit_sqlalchemy_database_driver_is_preserved() -> None:
    database_url = "postgresql+psycopg://user:password@database:5432/app"

    settings = Settings(_env_file=None, database_url=database_url)

    assert settings.database_url == database_url


def test_postgres_checkpoint_url_falls_back_to_operational_database() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://user:password@database:5432/app?sslmode=require",
        checkpoint_backend="postgres",
    )

    assert settings.postgres_checkpoint_url == (
        "postgresql://user:password@database:5432/app?sslmode=require"
    )


def test_explicit_checkpoint_url_takes_precedence() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://user:password@database:5432/app",
        checkpoint_backend="postgres",
        checkpoint_postgres_url="postgresql://checkpoint:password@database:5432/checkpoints",
    )

    assert settings.postgres_checkpoint_url == (
        "postgresql://checkpoint:password@database:5432/checkpoints"
    )


def test_production_rejects_non_durable_checkpoint_backend() -> None:
    with pytest.raises(ValidationError, match="CHECKPOINT_BACKEND=postgres"):
        Settings(
            _env_file=None,
            app_env="production",
            checkpoint_backend="sqlite",
            internal_api_key="secure-internal-key",
        )
