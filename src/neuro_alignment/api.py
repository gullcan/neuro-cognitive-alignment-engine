from __future__ import annotations

import asyncio
import hmac
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Annotated, Any, Literal

import structlog
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from neuro_alignment import __version__
from neuro_alignment.config import Settings, get_settings
from neuro_alignment.delivery import OutboxDeliveryReport
from neuro_alignment.domain import (
    EventSource,
    InboundEventType,
    NormalizedInboundEvent,
    ProcessResult,
    TaskAction,
)
from neuro_alignment.integrations import (
    IntegrationConfigurationError,
    verify_telegram_secret,
)
from neuro_alignment.runtime import AppServices
from neuro_alignment.workflow import WorkflowInputError

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


class EventProcessingResponse(BaseModel):
    """Public result contract shared by webhook and scheduler operations."""

    model_config = ConfigDict(extra="forbid")

    status: str
    event_id: str | None = None
    thread_id: str | None = None
    duplicate: bool = False
    queued_messages: int = Field(default=0, ge=0)
    delivery: OutboxDeliveryReport | None = None

    @classmethod
    def from_result(
        cls,
        result: ProcessResult,
        delivery: OutboxDeliveryReport,
    ) -> EventProcessingResponse:
        return cls(
            status=result.status,
            event_id=result.event_id,
            thread_id=result.thread_id,
            duplicate=result.duplicate,
            queued_messages=result.queued_messages,
            delivery=delivery,
        )


class DailyPlanTriggerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_date: date | None = None
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


class TaskMonitorTriggerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime | None = None
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


class OutboxDeliveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int | None = Field(default=None, ge=1, le=100)


def get_services(request: Request) -> AppServices:
    """Resolve the application-scoped dependency container."""
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, AppServices):
        raise RuntimeError("Application services are not initialized.")
    return services


Services = Annotated[AppServices, Depends(get_services)]
TelegramSecretHeader = Annotated[
    str | None,
    Header(alias="X-Telegram-Bot-Api-Secret-Token"),
]
InternalApiKeyHeader = Annotated[str | None, Header(alias="X-Internal-Api-Key")]


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
                intelligence_provider=type(services.intelligence).__name__,
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

    @application.post(
        "/v1/webhooks/telegram",
        response_model=EventProcessingResponse,
        tags=["webhooks"],
        summary="Receive one authenticated Telegram update",
    )
    async def telegram_webhook(
        update: dict[str, Any],
        services: Services,
        telegram_secret: TelegramSecretHeader = None,
    ) -> EventProcessingResponse:
        expected_secret = (
            configured_settings.telegram_webhook_secret.get_secret_value()
            if configured_settings.telegram_webhook_secret
            else None
        )
        if not verify_telegram_secret(telegram_secret, expected_secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Telegram webhook secret.",
            )

        try:
            event = services.telegram_updates.normalize(update)
        except PermissionError as error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Telegram chat is not authorized.",
            ) from error
        except (KeyError, TypeError, ValueError) as error:
            if str(error) == "Unsupported Telegram update type.":
                return EventProcessingResponse(status="ignored")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malformed Telegram update.",
            ) from error

        callback_query_id = event.payload.get("callback_query_id")
        is_plan_decision = event.action in {
            TaskAction.PLAN_APPROVED,
            TaskAction.PLAN_REJECTED,
        }
        if callback_query_id and not is_plan_decision:
            try:
                await services.telegram.answer_callback_query(str(callback_query_id))
            except Exception as error:
                await logger.awarning(
                    "telegram_callback_answer_failed",
                    event_id=event.event_id,
                    error_type=type(error).__name__,
                )

        try:
            result = await process_and_deliver(event, services)
        except (LookupError, WorkflowInputError):
            # Invalid or already-opposed plan decisions are permanent input errors;
            # returning 2xx prevents Telegram from retrying the same callback forever.
            result = EventProcessingResponse(
                status="rejected",
                event_id=event.event_id,
                thread_id=services.workflow.thread_id_for(event) if services.workflow else None,
            )
        if callback_query_id and is_plan_decision:
            callback_text = plan_callback_text(result)
            try:
                await services.telegram.answer_callback_query(
                    str(callback_query_id),
                    text=callback_text,
                )
            except Exception as error:
                await logger.awarning(
                    "telegram_callback_answer_failed",
                    event_id=event.event_id,
                    error_type=type(error).__name__,
                )
            message_id = event.payload.get("message_id")
            chat_id = event.payload.get("chat_id")
            if isinstance(message_id, int) and chat_id:
                try:
                    await services.telegram.clear_inline_keyboard(str(chat_id), message_id)
                except Exception as error:
                    await logger.awarning(
                        "telegram_keyboard_clear_failed",
                        event_id=event.event_id,
                        error_type=type(error).__name__,
                    )
        return result

    @application.post(
        "/v1/internal/scheduler/daily-plan",
        response_model=EventProcessingResponse,
        tags=["internal"],
        summary="Trigger the daily planning workflow",
    )
    async def trigger_daily_plan(
        request_body: DailyPlanTriggerRequest,
        services: Services,
        internal_api_key: InternalApiKeyHeader = None,
    ) -> EventProcessingResponse:
        require_internal_api_key(internal_api_key, configured_settings)
        target_date = request_body.plan_date or datetime.now(configured_settings.tz).date()
        source_event_id = request_body.request_id or target_date.isoformat()
        event = NormalizedInboundEvent(
            event_id=f"daily-plan:{configured_settings.default_user_id}:{source_event_id}",
            event_type=InboundEventType.DAILY_PLAN_REQUESTED,
            source=EventSource.SCHEDULER,
            user_id=configured_settings.default_user_id,
            occurred_at=datetime.now(configured_settings.tz),
            payload={"plan_date": target_date.isoformat()},
        )
        try:
            return await process_and_deliver(event, services)
        except IntegrationConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Notion integration is not configured or available.",
            ) from error

    @application.post(
        "/v1/internal/scheduler/task-monitor",
        response_model=EventProcessingResponse,
        tags=["internal"],
        summary="Run one idempotent task monitoring cycle",
    )
    async def trigger_task_monitor(
        request_body: TaskMonitorTriggerRequest,
        services: Services,
        internal_api_key: InternalApiKeyHeader = None,
    ) -> EventProcessingResponse:
        require_internal_api_key(internal_api_key, configured_settings)
        observed_at = request_body.observed_at or datetime.now(configured_settings.tz)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=configured_settings.tz)
        else:
            observed_at = observed_at.astimezone(configured_settings.tz)
        source_event_id = request_body.request_id or observed_at.strftime("%Y%m%d-%H%M")
        event = NormalizedInboundEvent(
            event_id=f"task-monitor:{configured_settings.default_user_id}:{source_event_id}",
            event_type=InboundEventType.TASK_MONITOR_TICK,
            source=EventSource.SCHEDULER,
            user_id=configured_settings.default_user_id,
            occurred_at=observed_at,
            payload={"plan_date": observed_at.date().isoformat()},
        )
        return await process_and_deliver(event, services)

    @application.post(
        "/v1/internal/outbox/deliver",
        response_model=OutboxDeliveryReport,
        tags=["internal"],
        summary="Deliver one leased outbox batch",
    )
    async def deliver_outbox(
        request_body: OutboxDeliveryRequest,
        services: Services,
        internal_api_key: InternalApiKeyHeader = None,
    ) -> OutboxDeliveryReport:
        require_internal_api_key(internal_api_key, configured_settings)
        if services.dispatcher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Outbox dispatcher is not initialized.",
            )
        return await services.dispatcher.deliver(limit=request_body.limit)

    return application


async def process_and_deliver(
    event: NormalizedInboundEvent,
    services: AppServices,
) -> EventProcessingResponse:
    if services.workflow is None or services.dispatcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow runtime is not initialized.",
        )
    result = await services.workflow.process(event)
    delivery = await services.dispatcher.deliver()
    return EventProcessingResponse.from_result(result, delivery)


def require_internal_api_key(received: str | None, settings: Settings) -> None:
    expected = settings.internal_api_key.get_secret_value()
    if received is None or not hmac.compare_digest(received, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key.",
        )


def plan_callback_text(result: EventProcessingResponse) -> str:
    if result.status == "rejected":
        return "Bu seçim daha önce yapıldı."
    if result.status == "plan_approved":
        return "Bugünün planı hazır." if result.queued_messages else "Bu plan zaten hazır."
    if result.status == "plan_rejected":
        return (
            "Tamam, planı düzenleyebilirsin."
            if result.queued_messages
            else "Bu planı daha önce düzenlemeye ayırdın."
        )
    if result.status == "duplicate":
        return "Bu işlem daha önce işlendi."
    return "Tamamdır."


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
