from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pytest import MonkeyPatch

from neuro_alignment.api import create_app
from neuro_alignment.config import Settings
from neuro_alignment.intelligence import RuleBasedIntelligenceProvider
from neuro_alignment.storage import Database


@pytest.mark.asyncio
async def test_health_endpoints_and_lifespan(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    application = create_app(settings=settings)

    async with application.router.lifespan_context(application):
        services = application.state.services
        assert isinstance(services.intelligence, RuleBasedIntelligenceProvider)
        assert services.workflow_ready
        assert not services.http_client.is_closed

        async with build_client(application) as client:
            live_response = await client.get("/health/live")
            assert live_response.status_code == 200
            assert live_response.json() == {
                "status": "ok",
                "service": "neuro-cognitive-alignment-engine",
                "version": "0.1.0",
                "environment": "test",
                "checks": {"process": "ok"},
            }

            ready_response = await client.get("/health/ready")
            assert ready_response.status_code == 200
            assert ready_response.json()["checks"] == {
                "database": "ok",
                "workflow": "ok",
            }

    assert services.http_client.is_closed
    assert not hasattr(application.state, "services")


@pytest.mark.asyncio
async def test_readiness_returns_503_without_leaking_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def fail_ping(_database: Database) -> None:
        raise RuntimeError("sensitive database details")

    monkeypatch.setattr(Database, "ping", fail_ping)
    application = create_app(settings=build_test_settings(tmp_path))

    async with (
        application.router.lifespan_context(application),
        build_client(application) as client,
    ):
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "neuro-cognitive-alignment-engine",
        "version": "0.1.0",
        "environment": "test",
        "checks": {"database": "unavailable", "workflow": "ok"},
    }
    assert "sensitive" not in response.text


@pytest.mark.asyncio
async def test_openapi_exposes_health_contract(tmp_path: Path) -> None:
    application = create_app(settings=build_test_settings(tmp_path))

    async with (
        application.router.lifespan_context(application),
        build_client(application) as client,
    ):
        schema = (await client.get("/openapi.json")).json()

    assert set(schema["paths"]) >= {
        "/health/live",
        "/health/ready",
        "/v1/webhooks/telegram",
        "/v1/internal/scheduler/daily-plan",
        "/v1/internal/outbox/deliver",
    }
    assert schema["info"]["version"] == "0.1.0"


def build_client(application: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    )


def build_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'api.db'}",
        checkpoint_backend="memory",
        openai_api_key=None,
    )
