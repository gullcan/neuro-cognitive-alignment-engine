from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import structlog
import uvicorn
from fastapi import Depends, FastAPI, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from neuro_alignment import __version__
from neuro_alignment.config import Settings, get_settings
from neuro_alignment.runtime import AppServices

logger = structlog.get_logger()

SERVICE_NAME = "neuro-cognitive-alignment-engine"
ServiceBuilder = Callable[[Settings], AppServices]


class HealthResponse(BaseModel):
    """Stable health payload for local tools and deployment probes."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "not_ready"]
    service: str = SERVICE_NAME
    version: str = __version__
    environment: str
    checks: dict[str, str] = Field(default_factory=dict)


def get_services(request: Request) -> AppServices:
    """Resolve the application-scoped dependency container."""
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, AppServices):
        raise RuntimeError("Application services are not initialized.")
    return services


Services = Annotated[AppServices, Depends(get_services)]


def create_app(
    *,
    settings: Settings | None = None,
    service_builder: ServiceBuilder = AppServices.build,
) -> FastAPI:
    """Create an isolated FastAPI application for production or tests."""
    configured_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        services = service_builder(configured_settings)
        try:
            await services.start()
            application.state.services = services
            await logger.ainfo(
                "application_started",
                environment=configured_settings.app_env,
                version=__version__,
                checkpoint_backend=configured_settings.checkpoint_backend,
            )
            yield
        finally:
            await services.close()
            if hasattr(application.state, "services"):
                del application.state.services
            await logger.ainfo("application_stopped")

    application = FastAPI(
        title="Neuro-Cognitive Alignment Engine",
        description=(
            "Stateful, evidence-aware intent-action alignment API. "
            "It does not provide medical diagnosis or measure neurological activity."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    @application.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["health"],
        summary="Process liveness",
    )
    async def liveness() -> HealthResponse:
        return HealthResponse(
            status="ok",
            environment=configured_settings.app_env,
            checks={"process": "ok"},
        )

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": HealthResponse,
                "description": "A required dependency is unavailable.",
            }
        },
        tags=["health"],
        summary="Dependency readiness",
    )
    async def readiness(response: Response, services: Services) -> HealthResponse:
        if not services.workflow_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="not_ready",
                environment=configured_settings.app_env,
                checks={"database": "unknown", "workflow": "unavailable"},
            )
        try:
            async with asyncio.timeout(configured_settings.readiness_timeout_seconds):
                await services.database.ping()
        except Exception as error:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            await logger.awarning(
                "readiness_check_failed",
                dependency="database",
                error_type=type(error).__name__,
            )
            return HealthResponse(
                status="not_ready",
                environment=configured_settings.app_env,
                checks={"database": "unavailable", "workflow": "ok"},
            )

        return HealthResponse(
            status="ok",
            environment=configured_settings.app_env,
            checks={"database": "ok", "workflow": "ok"},
        )

    return application


app = create_app()


def run() -> None:
    """Run the API through the installed console command."""
    settings = get_settings()
    uvicorn.run(
        "neuro_alignment.api:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
    )
