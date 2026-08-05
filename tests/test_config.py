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
    )

    assert settings.notion_api_token is None
    assert settings.telegram_bot_token is None
    assert settings.telegram_webhook_secret is None
    assert settings.openai_api_key is None


def test_telegram_webhook_secret_rejects_unsupported_characters() -> None:
    with pytest.raises(ValidationError, match="TELEGRAM_WEBHOOK_SECRET"):
        Settings(
            _env_file=None,
            telegram_webhook_secret="contains spaces",
        )


def test_production_rejects_non_durable_checkpoint_backend() -> None:
    with pytest.raises(ValidationError, match="CHECKPOINT_BACKEND=postgres"):
        Settings(
            _env_file=None,
            app_env="production",
            checkpoint_backend="sqlite",
            internal_api_key="secure-internal-key",
        )
