from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from alembic.config import Config

from neuro_alignment import deployment


def test_resolve_port_uses_render_default() -> None:
    assert deployment.resolve_port({}) == "10000"


@pytest.mark.parametrize("value", ["zero", "0", "65536"])
def test_resolve_port_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="PORT"):
        deployment.resolve_port({"PORT": value})


def test_start_runs_migrations_before_replacing_process(monkeypatch: pytest.MonkeyPatch) -> None:
    upgrade = Mock()
    execvp = Mock()
    monkeypatch.setattr(deployment.command, "upgrade", upgrade)
    monkeypatch.setattr(deployment.os, "execvp", execvp)
    monkeypatch.setenv("PORT", "10000")

    deployment.start()

    migration_config = upgrade.call_args.args[0]
    assert isinstance(migration_config, Config)
    script_location = migration_config.get_main_option("script_location")
    assert Path(script_location).resolve() == Path("migrations").resolve()
    assert upgrade.call_args.args[1] == "head"
    execvp.assert_called_once_with(
        "uvicorn",
        [
            "uvicorn",
            "neuro_alignment.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "10000",
        ],
    )
