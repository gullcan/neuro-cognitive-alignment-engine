from __future__ import annotations

import os
from collections.abc import Mapping

from alembic import command
from alembic.config import Config


def resolve_port(environment: Mapping[str, str]) -> str:
    """Return Render's public port after enforcing Uvicorn's valid range."""
    raw_port = environment.get("PORT", "10000")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise ValueError("PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    return str(port)


def start() -> None:
    """Apply idempotent migrations, then replace this process with Uvicorn."""
    command.upgrade(Config("alembic.ini"), "head")
    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "neuro_alignment.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            resolve_port(os.environ),
        ],
    )
