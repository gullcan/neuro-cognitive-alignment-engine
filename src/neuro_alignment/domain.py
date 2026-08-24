from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class EventSource(StrEnum):
    SCHEDULER = "scheduler"
    NOTION = "notion"
    TELEGRAM = "telegram"
    INTERNAL = "internal"


class InboundEventType(StrEnum):
    DAILY_PLAN_REQUESTED = "scheduler.daily_plan_requested"
    NOTION_CHANGED = "notion.commitment_changed"
    TELEGRAM_ACTION = "telegram.action_received"
    TELEGRAM_MESSAGE = "telegram.message_received"


class TaskAction(StrEnum):
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    STARTED = "started"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    RESCHEDULED = "rescheduled"


class DomainEventType(StrEnum):
    PLAN_CREATED = "plan.created"
    PLAN_CONFIRMED = "plan.confirmed"
    PLAN_REJECTED = "plan.rejected"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_BLOCKED = "task.blocked"
    TASK_SKIPPED = "task.skipped"
    TASK_RESCHEDULED = "task.rescheduled"
    CHECKIN_RECEIVED = "checkin.received"
    FEEDBACK_SENT = "feedback.sent"


ACTION_TO_DOMAIN_EVENT: dict[TaskAction, DomainEventType] = {
    TaskAction.PLAN_APPROVED: DomainEventType.PLAN_CONFIRMED,
    TaskAction.PLAN_REJECTED: DomainEventType.PLAN_REJECTED,
    TaskAction.STARTED: DomainEventType.TASK_STARTED,
    TaskAction.COMPLETED: DomainEventType.TASK_COMPLETED,
    TaskAction.BLOCKED: DomainEventType.TASK_BLOCKED,
    TaskAction.SKIPPED: DomainEventType.TASK_SKIPPED,
    TaskAction.RESCHEDULED: DomainEventType.TASK_RESCHEDULED,
}


class NormalizedInboundEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: InboundEventType
    source: EventSource
    user_id: str
    occurred_at: datetime
    task_id: str | None = None
    action: TaskAction | None = None
    text: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class NotionTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str
    title: str
    scheduled_date: date
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    status: str = "Planned"
    commitment_tier: str = "Flexible"
    priority: str = "P2"
    project_ids: list[str] = Field(default_factory=list)
    definition_of_done: str = ""
    minimum_action: str = ""
    estimated_minutes: int | None = None
    cognitive_load: int | None = Field(default=None, ge=1, le=5)
    context_cue: str = ""
    evidence_required: bool = False
    evidence_url: HttpUrl | None = None
    skip_reason: str | None = None
    last_edited_time: datetime | None = None


class DailyPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    order: int = Field(ge=1)
    scheduled_start: datetime | None = None
    estimated_minutes: int | None = None
    commitment_tier: str
    priority: str
    cognitive_load: int | None = None
    definition_of_done: str
    minimum_action: str
    rationale: str


class DailyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_date: date
    headline: str
    items: list[DailyPlanItem]
    capacity_warning: str | None = None
    generated_by: str


class EvidenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    occurred_at: datetime
    task_id: str | None = None
    source: str


class SimilarBehaviorEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    task_id: str
    task_title: str
    action: TaskAction
    occurred_at: datetime
    similarity: float = Field(ge=0, le=1)
    context: dict[str, Any] = Field(default_factory=dict)


class BehaviorEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    total_events: int
    counts: dict[str, int] = Field(default_factory=dict)
    recent_events: list[EvidenceEvent] = Field(default_factory=list)
    similar_episodes: list[SimilarBehaviorEpisode] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    has_sufficient_history: bool = False


class NeuroFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_evidence: str
    behavioral_pattern: str
    interpretation: str
    neuro_context: str
    word_action_gap: str
    immediate_intervention: str
    follow_up_minutes: int = Field(ge=1, le=1440)
    evidence_request: str
    confidence: float = Field(ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    reasons: list[str] = Field(default_factory=list)


class InlineButton(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    callback_data: str = Field(max_length=64)


class OutboundMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str
    chat_id: str
    text: str
    buttons: list[list[InlineButton]] = Field(default_factory=list)


class ProcessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    event_id: str
    thread_id: str
    duplicate: bool = False
    interrupted: bool = False
    queued_messages: int = 0
