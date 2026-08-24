from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal, Protocol, TypedDict, cast

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from neuro_alignment.checkpointing import CheckpointSaver
from neuro_alignment.config import Settings
from neuro_alignment.domain import (
    ACTION_TO_DOMAIN_EVENT,
    BehaviorEvidence,
    CritiqueResult,
    DailyPlan,
    DailyPlanItem,
    DomainEventType,
    InboundEventType,
    InlineButton,
    NeuroFeedback,
    NormalizedInboundEvent,
    NotionTask,
    OutboundMessage,
    ProcessResult,
    TaskAction,
    TaskDayActivity,
)
from neuro_alignment.intelligence import IntelligenceProvider
from neuro_alignment.memory import build_behavior_context
from neuro_alignment.storage import (
    EventRepository,
    MemoryRepository,
    OutboxRepository,
    PlanRepository,
)

logger = structlog.get_logger()

WorkflowRoute = Literal["daily_plan", "task_monitor", "plan_decision", "behavior", "checkin"]
RouteDecision = Literal[
    "duplicate",
    "daily_plan",
    "task_monitor",
    "plan_decision",
    "behavior",
    "checkin",
]
FeedbackDecision = Literal["approved", "retry"]


class WorkflowState(TypedDict):
    """Serializable state persisted after each LangGraph super-step."""

    event: dict[str, Any]
    thread_id: str
    route: WorkflowRoute | Literal["unrouted"]
    duplicate: bool
    tasks: list[dict[str, Any]]
    plan: dict[str, Any] | None
    approval_token: str | None
    task_activity: dict[str, dict[str, Any]]
    evidence: dict[str, Any] | None
    behavior_context: dict[str, Any] | None
    behavior_vector: list[float]
    task_title: str | None
    feedback: dict[str, Any] | None
    critique: dict[str, Any] | None
    critique_notes: list[str]
    feedback_attempts: int
    queued_messages: int
    status: str


StateUpdate = dict[str, Any]
CompiledWorkflow = CompiledStateGraph[
    WorkflowState,
    None,
    WorkflowState,
    WorkflowState,
]


class CommitmentSource(Protocol):
    async def fetch_daily_tasks(self, target_date: date) -> list[NotionTask]: ...


@dataclass(frozen=True, slots=True)
class WorkflowDependencies:
    settings: Settings
    events: EventRepository
    memory: MemoryRepository
    plans: PlanRepository
    outbox: OutboxRepository
    notion: CommitmentSource
    intelligence: IntelligenceProvider


class UnsafeFeedbackError(RuntimeError):
    """Raised when generated feedback fails the bounded safety review twice."""


class WorkflowInputError(ValueError):
    """A permanent event-contract error that a transport must not retry."""


class WorkflowEngine:
    """Compile and execute the event-driven neuro-alignment state machine."""

    def __init__(
        self,
        dependencies: WorkflowDependencies,
        checkpointer: CheckpointSaver,
    ) -> None:
        self.dependencies = dependencies
        self.graph = self._compile(checkpointer)

    async def process(
        self,
        event: NormalizedInboundEvent,
        *,
        thread_id: str | None = None,
    ) -> ProcessResult:
        resolved_thread_id = thread_id or self.thread_id_for(event)
        initial_state = self._initial_state(event, resolved_thread_id)
        config: RunnableConfig = {
            "configurable": {"thread_id": resolved_thread_id},
            "tags": ["neuro-alignment", event.event_type.value],
            "metadata": {
                "event_id": event.event_id,
                "user_id": event.user_id,
            },
        }

        try:
            output = await self.graph.ainvoke(initial_state, config)
        except Exception as error:
            try:
                await self.dependencies.events.fail_inbound(event, error)
            except Exception as persistence_error:
                await logger.aerror(
                    "inbound_failure_record_failed",
                    event_id=event.event_id,
                    error_type=type(persistence_error).__name__,
                )
            raise

        state = cast(WorkflowState, output)
        duplicate = state["duplicate"]
        return ProcessResult(
            status="duplicate" if duplicate else state["status"],
            event_id=event.event_id,
            thread_id=resolved_thread_id,
            duplicate=duplicate,
            queued_messages=state["queued_messages"],
        )

    def thread_id_for(self, event: NormalizedInboundEvent) -> str:
        """Keep execution history isolated by business aggregate."""
        if event.event_type in {
            InboundEventType.DAILY_PLAN_REQUESTED,
            InboundEventType.TASK_MONITOR_TICK,
            InboundEventType.NOTION_CHANGED,
        }:
            return f"daily:{event.user_id}:{self._plan_date(event).isoformat()}"
        if event.task_id:
            return f"task:{event.user_id}:{event.task_id}"
        if event.action in {TaskAction.PLAN_APPROVED, TaskAction.PLAN_REJECTED}:
            token = str(event.payload.get("approval_token", event.event_id))
            return f"plan-decision:{event.user_id}:{token}"
        return f"checkin:{event.user_id}:{self._local_date(event).isoformat()}"

    def _compile(self, checkpointer: CheckpointSaver) -> CompiledWorkflow:
        builder = StateGraph(WorkflowState)
        builder.add_node("claim_event", self._claim_event)

        builder.add_node("load_commitments", self._load_commitments)
        builder.add_node("planner_agent", self._planner_agent)
        builder.add_node("persist_plan", self._persist_plan)
        builder.add_node("queue_plan", self._queue_plan)

        builder.add_node("load_approved_plan", self._load_approved_plan)
        builder.add_node("load_task_activity", self._load_task_activity)
        builder.add_node("queue_monitor_messages", self._queue_monitor_messages)

        builder.add_node("apply_plan_decision", self._apply_plan_decision)

        builder.add_node("record_behavior", self._record_behavior)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("neuro_behavioral_agent", self._neuro_behavioral_agent)
        builder.add_node("safety_critic", self._safety_critic)
        builder.add_node("queue_feedback", self._queue_feedback)

        builder.add_node("record_checkin", self._record_checkin)
        builder.add_node("queue_checkin_response", self._queue_checkin_response)
        builder.add_node("complete_event", self._complete_event)

        builder.add_edge(START, "claim_event")
        builder.add_conditional_edges(
            "claim_event",
            self._route_after_claim,
            {
                "duplicate": END,
                "daily_plan": "load_commitments",
                "task_monitor": "load_approved_plan",
                "plan_decision": "apply_plan_decision",
                "behavior": "record_behavior",
                "checkin": "record_checkin",
            },
        )

        builder.add_edge("load_commitments", "planner_agent")
        builder.add_edge("planner_agent", "persist_plan")
        builder.add_edge("persist_plan", "queue_plan")
        builder.add_edge("queue_plan", "complete_event")

        builder.add_edge("load_approved_plan", "load_task_activity")
        builder.add_edge("load_task_activity", "queue_monitor_messages")
        builder.add_edge("queue_monitor_messages", "complete_event")

        builder.add_edge("apply_plan_decision", "complete_event")

        builder.add_edge("record_behavior", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "neuro_behavioral_agent")
        builder.add_edge("neuro_behavioral_agent", "safety_critic")
        builder.add_conditional_edges(
            "safety_critic",
            self._route_after_critique,
            {
                "approved": "queue_feedback",
                "retry": "neuro_behavioral_agent",
            },
        )
        builder.add_edge("queue_feedback", "complete_event")

        builder.add_edge("record_checkin", "queue_checkin_response")
        builder.add_edge("queue_checkin_response", "complete_event")
        builder.add_edge("complete_event", END)
        return builder.compile(
            checkpointer=checkpointer,
            name="neuro-cognitive-alignment-workflow",
        )

    async def _claim_event(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        route = self._select_route(event)
        claimed = await self.dependencies.events.claim_inbound(event)
        return {
            "route": route,
            "duplicate": not claimed,
        }

    @staticmethod
    def _route_after_claim(state: WorkflowState) -> RouteDecision:
        if state["duplicate"]:
            return "duplicate"
        route = state["route"]
        if route == "unrouted":
            raise RuntimeError("Workflow route was not resolved.")
        return route

    async def _load_commitments(self, state: WorkflowState) -> StateUpdate:
        tasks = await self.dependencies.notion.fetch_daily_tasks(
            self._plan_date(self._event(state))
        )
        return {"tasks": [task.model_dump(mode="json") for task in tasks]}

    async def _planner_agent(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        tasks = [NotionTask.model_validate(task) for task in state["tasks"]]
        plan = await self.dependencies.intelligence.build_daily_plan(
            tasks=tasks,
            plan_date=self._plan_date(event).isoformat(),
            user_id=event.user_id,
        )
        return {"plan": plan.model_dump(mode="json")}

    async def _persist_plan(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        plan = self._plan(state)
        approval_token = await self.dependencies.plans.save(
            event.user_id,
            plan,
            state["thread_id"],
        )
        await self.dependencies.events.append_domain_event(
            event_type=DomainEventType.PLAN_CREATED.value,
            user_id=event.user_id,
            task_id=None,
            source=event.source.value,
            occurred_at=event.occurred_at,
            payload={
                "inbound_event_id": event.event_id,
                "plan_date": plan.plan_date.isoformat(),
                "item_count": len(plan.items),
                "thread_id": state["thread_id"],
            },
            idempotency_key=f"{event.source.value}:{event.event_id}:plan-created",
        )
        return {"approval_token": approval_token}

    async def _queue_plan(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        chat_id = self._chat_id(event)
        queued = 0
        if chat_id:
            token = state["approval_token"]
            if token is None:
                raise RuntimeError("Plan approval token was not generated.")
            message = OutboundMessage(
                idempotency_key=f"daily-plan:{event.event_id}",
                chat_id=chat_id,
                text=self._format_plan(self._plan(state)),
                buttons=[
                    [
                        InlineButton(
                            text="Planı onayla",
                            callback_data=f"plan:approve:{token}",
                        ),
                        InlineButton(
                            text="Planı reddet",
                            callback_data=f"plan:reject:{token}",
                        ),
                    ]
                ],
            )
            queued = await self.dependencies.outbox.enqueue([message])
        return {"queued_messages": queued, "status": "plan_created"}

    async def _load_approved_plan(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        resolved = await self.dependencies.plans.approved_plan(
            event.user_id,
            self._plan_date(event),
        )
        if resolved is None:
            return {
                "plan": None,
                "approval_token": None,
                "status": "monitor_no_approved_plan",
            }
        approval_token, plan = resolved
        return {
            "plan": plan.model_dump(mode="json"),
            "approval_token": approval_token,
        }

    async def _load_task_activity(self, state: WorkflowState) -> StateUpdate:
        raw_plan = state["plan"]
        if raw_plan is None:
            return {"task_activity": {}}
        plan = DailyPlan.model_validate(raw_plan)
        period_start = datetime.combine(
            plan.plan_date,
            time.min,
            tzinfo=self.dependencies.settings.tz,
        ).astimezone(UTC)
        period_end = period_start + timedelta(days=1)
        activity = await self.dependencies.events.task_activity_for_period(
            user_id=self._event(state).user_id,
            task_ids=[item.task_id for item in plan.items],
            period_start=period_start,
            period_end=period_end,
        )
        return {
            "task_activity": {
                task_id: value.model_dump(mode="json") for task_id, value in activity.items()
            }
        }

    async def _queue_monitor_messages(self, state: WorkflowState) -> StateUpdate:
        raw_plan = state["plan"]
        token = state["approval_token"]
        chat_id = self._chat_id(self._event(state))
        if raw_plan is None or token is None or not chat_id:
            return {"queued_messages": 0, "status": state["status"]}

        plan = DailyPlan.model_validate(raw_plan)
        now = self._event(state).occurred_at.astimezone(self.dependencies.settings.tz)
        activities = {
            task_id: TaskDayActivity.model_validate(value)
            for task_id, value in state["task_activity"].items()
        }
        messages: list[OutboundMessage] = []
        terminal_actions = {
            TaskAction.COMPLETED,
            TaskAction.SKIPPED,
            TaskAction.RESCHEDULED,
        }

        for item in plan.items:
            scheduled_start = self._scheduled_start(item)
            if scheduled_start is None or now < scheduled_start:
                continue
            activity = activities.get(item.task_id)
            latest_action = activity.latest_action if activity else None
            if activity and activity.counts.get("task.completed", 0) > 0:
                continue
            if latest_action in terminal_actions:
                continue

            if latest_action is None:
                grace_end = scheduled_start + timedelta(
                    minutes=self.dependencies.settings.task_start_grace_minutes
                )
                if now < grace_end:
                    kind = "due"
                    text = self._format_due_reminder(item, scheduled_start)
                else:
                    kind = "start-check"
                    text = self._format_start_check(item, scheduled_start)
                messages.append(
                    self._task_monitor_message(
                        idempotency_key=f"task-{kind}:{token}:{item.task_id}",
                        chat_id=chat_id,
                        item=item,
                        text=text,
                    )
                )
                continue

            latest_at = activity.latest_action_at if activity else None
            if latest_at is None:
                continue
            latest_local = latest_at.astimezone(self.dependencies.settings.tz)
            if latest_action == TaskAction.STARTED:
                expected_minutes = (
                    item.estimated_minutes
                    or self.dependencies.settings.task_progress_default_minutes
                )
                if now >= latest_local + timedelta(minutes=expected_minutes):
                    messages.append(
                        self._task_monitor_message(
                            idempotency_key=f"task-progress-check:{token}:{item.task_id}",
                            chat_id=chat_id,
                            item=item,
                            text=self._format_progress_check(item, expected_minutes),
                        )
                    )
            elif latest_action == TaskAction.BLOCKED and now >= latest_local + timedelta(
                minutes=self.dependencies.settings.blocked_follow_up_minutes
            ):
                messages.append(
                    self._task_monitor_message(
                        idempotency_key=f"task-blocked-check:{token}:{item.task_id}",
                        chat_id=chat_id,
                        item=item,
                        text=self._format_blocked_check(item),
                    )
                )

        open_items = [
            item
            for item in plan.items
            if activities.get(item.task_id) is None
            or activities[item.task_id].counts.get("task.completed", 0) == 0
        ]
        completed_count = len(plan.items) - len(open_items)
        if plan.items and now.hour >= self.dependencies.settings.day_summary_hour:
            messages.append(
                self._day_status_message(
                    idempotency_key=f"day-summary:{token}",
                    chat_id=chat_id,
                    heading="GÜNÜN SON KONTROLÜ",
                    plan=plan,
                    open_items=open_items,
                    completed_count=completed_count,
                )
            )
        elif plan.items and now.hour >= self.dependencies.settings.day_recovery_hour:
            messages.append(
                self._day_status_message(
                    idempotency_key=f"day-recovery:{token}",
                    chat_id=chat_id,
                    heading="GÜN SONU YAKLAŞIYOR",
                    plan=plan,
                    open_items=open_items,
                    completed_count=completed_count,
                )
            )

        queued = await self.dependencies.outbox.enqueue(messages)
        return {"queued_messages": queued, "status": "task_monitor_completed"}

    async def _apply_plan_decision(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        action = event.action
        if action not in {TaskAction.PLAN_APPROVED, TaskAction.PLAN_REJECTED}:
            raise WorkflowInputError(
                "Plan decision route requires an approval or rejection action."
            )
        approval_token = str(event.payload.get("approval_token", ""))
        if not approval_token:
            raise WorkflowInputError("Plan decision is missing approval_token.")
        resolved = await self.dependencies.plans.plan_by_approval_token(
            event.user_id,
            approval_token,
        )
        if resolved is None:
            raise LookupError("Plan approval token is invalid or expired.")
        _plan_thread_id, plan_date, current_status, plan = resolved
        status = "approved" if action == TaskAction.PLAN_APPROVED else "rejected"
        if current_status not in {"pending", status}:
            raise WorkflowInputError(
                f"Plan is already {current_status}; it cannot be changed to {status}."
            )
        await self.dependencies.events.append_domain_event(
            event_type=ACTION_TO_DOMAIN_EVENT[action].value,
            user_id=event.user_id,
            task_id=None,
            source=event.source.value,
            occurred_at=event.occurred_at,
            payload={
                "inbound_event_id": event.event_id,
                "plan_date": plan_date.isoformat(),
                "approval_token": approval_token,
            },
            idempotency_key=f"{event.source.value}:{event.event_id}:plan-decision",
        )
        if current_status == "pending":
            await self.dependencies.plans.update_status(event.user_id, plan_date, status)

        queued = 0
        chat_id = self._chat_id(event)
        if chat_id:
            decision_text = (
                "Plan onaylandı. Saatli görevler zamanı geldiğinde açılacak; "
                "sistem başlangıç ve ilerleme kanıtını gün boyunca takip edecek."
                if status == "approved"
                else "Plan reddedildi. Yeni plan oluşturulmadan taahhüt seti aktif sayılmayacak."
            )
            messages = [
                OutboundMessage(
                    idempotency_key=f"plan-decision:{approval_token}:{status}",
                    chat_id=chat_id,
                    text=decision_text,
                )
            ]
            if status == "approved":
                messages.extend(
                    self._task_control_message(
                        chat_id=chat_id,
                        approval_token=approval_token,
                        item=item,
                    )
                    for item in plan.items
                    if self._scheduled_start(item) is None
                )
            queued = await self.dependencies.outbox.enqueue(messages)
        return {
            "queued_messages": queued,
            "status": f"plan_{status}",
        }

    @staticmethod
    def _task_control_message(
        *,
        chat_id: str,
        approval_token: str,
        item: DailyPlanItem,
    ) -> OutboundMessage:
        definition = item.definition_of_done or "Somut tamamlanma kanıtını bildir."
        minimum_action = item.minimum_action or "İlk fiziksel adımı belirle ve başlat."
        return OutboundMessage(
            idempotency_key=f"plan-task:{approval_token}:{item.task_id}",
            chat_id=chat_id,
            text=(
                f"AKTİF TAAHHÜT {item.order}\n"
                f"{item.title}\n\n"
                f"Minimum eylem: {minimum_action}\n"
                f"Tamamlanma kanıtı: {definition}\n\n"
                "Durumu yalnızca gözlenebilir eylemine göre seç."
            ),
            buttons=WorkflowEngine._task_action_buttons(item.task_id),
        )

    @staticmethod
    def _task_action_buttons(task_id: str) -> list[list[InlineButton]]:
        return [
            [
                InlineButton(
                    text="Başlattım",
                    callback_data=f"task:started:{task_id}",
                ),
                InlineButton(
                    text="Tamamladım",
                    callback_data=f"task:completed:{task_id}",
                ),
            ],
            [
                InlineButton(
                    text="Engellendim",
                    callback_data=f"task:blocked:{task_id}",
                ),
                InlineButton(
                    text="Atladım",
                    callback_data=f"task:skipped:{task_id}",
                ),
            ],
            [
                InlineButton(
                    text="Erteledim",
                    callback_data=f"task:rescheduled:{task_id}",
                )
            ],
        ]

    async def _record_behavior(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        if event.action is None or event.action not in ACTION_TO_DOMAIN_EVENT:
            raise WorkflowInputError("Behavior route requires a supported task action.")
        if event.task_id is None:
            raise WorkflowInputError("Behavior route requires task_id.")
        await self.dependencies.events.append_domain_event(
            event_type=ACTION_TO_DOMAIN_EVENT[event.action].value,
            user_id=event.user_id,
            task_id=event.task_id,
            source=event.source.value,
            occurred_at=event.occurred_at,
            payload={
                "inbound_event_id": event.event_id,
                "self_report": event.payload,
            },
            idempotency_key=f"{event.source.value}:{event.event_id}:task-action",
        )
        item = await self.dependencies.plans.task_item(event.user_id, event.task_id)
        task_title = item.title if item else str(event.payload.get("task_title") or event.task_id)
        context, vector = build_behavior_context(
            event,
            item,
            self.dependencies.settings.tz,
        )
        await self.dependencies.memory.store_episode(
            source_event_id=event.event_id,
            user_id=event.user_id,
            task_id=event.task_id,
            task_title=task_title,
            action=event.action,
            occurred_at=event.occurred_at,
            context=context,
            embedding=vector,
        )
        return {
            "behavior_context": context,
            "behavior_vector": vector,
            "task_title": task_title,
        }

    async def _retrieve_evidence(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        if event.task_id is None:
            raise WorkflowInputError("Evidence retrieval requires task_id.")
        evidence = await self.dependencies.events.build_evidence(
            user_id=event.user_id,
            task_id=event.task_id,
        )
        similar = await self.dependencies.memory.similar_episodes(
            user_id=event.user_id,
            embedding=state["behavior_vector"],
            exclude_source_event_id=event.event_id,
        )
        evidence = evidence.model_copy(
            update={
                "similar_episodes": similar,
                "evidence_refs": evidence.evidence_refs
                + [episode.memory_id for episode in similar],
                "has_sufficient_history": evidence.has_sufficient_history or len(similar) >= 3,
            }
        )
        return {
            "evidence": evidence.model_dump(mode="json"),
        }

    async def _neuro_behavioral_agent(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        evidence = self._evidence(state)
        feedback = await self.dependencies.intelligence.generate_feedback(
            event=event,
            task_title=state["task_title"] or event.task_id or "Görev",
            evidence=evidence,
            critique_notes=state["critique_notes"],
        )
        return {
            "feedback": feedback.model_dump(mode="json"),
            "feedback_attempts": state["feedback_attempts"] + 1,
        }

    async def _safety_critic(self, state: WorkflowState) -> StateUpdate:
        feedback = self._feedback(state)
        evidence = self._evidence(state)
        reasons: list[str] = []

        unknown_refs = set(feedback.evidence_refs) - set(evidence.evidence_refs)
        if unknown_refs:
            reasons.append("Feedback contains evidence references that were not retrieved.")
        if not evidence.has_sufficient_history and feedback.confidence > 0.8:
            reasons.append("Confidence exceeds the permitted level for sparse history.")

        combined_text = " ".join(
            (
                feedback.observed_evidence,
                feedback.behavioral_pattern,
                feedback.interpretation,
                feedback.neuro_context,
                feedback.word_action_gap,
                feedback.immediate_intervention,
            )
        ).casefold()
        forbidden_claims = {
            "bana ihtiyacın var",
            "dopamin seviyeni ölçtüm",
            "dopaminini yükselteceğim",
            "dopamin reseptörlerin zayıf",
            "prefrontal korteksin güçlendi",
            "sadece benim onayım",
            "yürütücü işlev bozukluğun var",
            "sende adhd var",
            "klinik tanın",
        }
        matched_claims = sorted(claim for claim in forbidden_claims if claim in combined_text)
        if matched_claims:
            reasons.append(
                "Feedback makes an unsupported biological or clinical claim: "
                + ", ".join(matched_claims)
            )

        critique = CritiqueResult(passed=not reasons, reasons=reasons)
        return {
            "critique": critique.model_dump(mode="json"),
            "critique_notes": reasons,
        }

    @staticmethod
    def _route_after_critique(state: WorkflowState) -> FeedbackDecision:
        raw_critique = state["critique"]
        if raw_critique is None:
            raise RuntimeError("Feedback critique is missing.")
        critique = CritiqueResult.model_validate(raw_critique)
        if critique.passed:
            return "approved"
        if state["feedback_attempts"] < 2:
            return "retry"
        raise UnsafeFeedbackError("Feedback failed the safety review after one revision.")

    async def _queue_feedback(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        queued = 0
        chat_id = self._chat_id(event)
        if chat_id:
            queued = await self.dependencies.outbox.enqueue(
                [
                    OutboundMessage(
                        idempotency_key=f"behavior-feedback:{event.event_id}",
                        chat_id=chat_id,
                        text=self._format_feedback(self._feedback(state)),
                    )
                ]
            )
        return {"queued_messages": queued, "status": "feedback_generated"}

    async def _record_checkin(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        await self.dependencies.events.append_domain_event(
            event_type=DomainEventType.CHECKIN_RECEIVED.value,
            user_id=event.user_id,
            task_id=event.task_id,
            source=event.source.value,
            occurred_at=event.occurred_at,
            payload={
                "inbound_event_id": event.event_id,
                "text": event.text or "",
            },
            idempotency_key=f"{event.source.value}:{event.event_id}:checkin",
        )
        return {"status": "checkin_recorded"}

    async def _queue_checkin_response(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        chat_id = self._chat_id(event)
        if not chat_id:
            return {"queued_messages": 0}

        queued = await self.dependencies.outbox.enqueue(
            [
                OutboundMessage(
                    idempotency_key=f"checkin-response:{event.event_id}",
                    chat_id=chat_id,
                    text=(
                        "CHECK-IN KAYDEDİLDİ\n\n"
                        "Bu bildirim bir niyet beyanıdır; henüz davranış kanıtı değildir. "
                        "Söz-eylem tutarlılığı, planlanan görevin başlatılması ve "
                        "tamamlanmasıyla ölçülecek.\n\n"
                        "Şimdi seçtiğin görevin minimum eylemini başlat ve sonucu "
                        "görev düğmesiyle bildir."
                    ),
                )
            ]
        )
        return {"queued_messages": queued}

    async def _complete_event(self, state: WorkflowState) -> StateUpdate:
        await self.dependencies.events.complete_inbound(self._event(state))
        return {}

    @staticmethod
    def _select_route(event: NormalizedInboundEvent) -> WorkflowRoute:
        if event.event_type in {
            InboundEventType.DAILY_PLAN_REQUESTED,
            InboundEventType.NOTION_CHANGED,
        }:
            return "daily_plan"
        if event.event_type == InboundEventType.TASK_MONITOR_TICK:
            return "task_monitor"
        if event.event_type == InboundEventType.TELEGRAM_MESSAGE:
            return "checkin"
        if event.event_type == InboundEventType.TELEGRAM_ACTION:
            if event.action in {TaskAction.PLAN_APPROVED, TaskAction.PLAN_REJECTED}:
                return "plan_decision"
            if event.action in ACTION_TO_DOMAIN_EVENT:
                return "behavior"
        raise WorkflowInputError(f"Unsupported event/action combination: {event.event_type}")

    def _plan_date(self, event: NormalizedInboundEvent) -> date:
        raw_date = event.payload.get("plan_date")
        if isinstance(raw_date, str):
            return date.fromisoformat(raw_date)
        return self._local_date(event)

    def _local_date(self, event: NormalizedInboundEvent) -> date:
        return event.occurred_at.astimezone(self.dependencies.settings.tz).date()

    def _chat_id(self, event: NormalizedInboundEvent) -> str:
        value = event.payload.get("chat_id") or self.dependencies.settings.telegram_chat_id
        return str(value) if value else ""

    @staticmethod
    def _initial_state(
        event: NormalizedInboundEvent,
        thread_id: str,
    ) -> WorkflowState:
        # Every volatile field is overwritten so a reused thread cannot leak the
        # previous invocation's branch data into the next event.
        return WorkflowState(
            event=event.model_dump(mode="json"),
            thread_id=thread_id,
            route="unrouted",
            duplicate=False,
            tasks=[],
            plan=None,
            approval_token=None,
            task_activity={},
            evidence=None,
            behavior_context=None,
            behavior_vector=[],
            task_title=None,
            feedback=None,
            critique=None,
            critique_notes=[],
            feedback_attempts=0,
            queued_messages=0,
            status="processing",
        )

    @staticmethod
    def _event(state: WorkflowState) -> NormalizedInboundEvent:
        return NormalizedInboundEvent.model_validate(state["event"])

    @staticmethod
    def _plan(state: WorkflowState) -> DailyPlan:
        raw_plan = state["plan"]
        if raw_plan is None:
            raise RuntimeError("Daily plan is missing from workflow state.")
        return DailyPlan.model_validate(raw_plan)

    @staticmethod
    def _evidence(state: WorkflowState) -> BehaviorEvidence:
        raw_evidence = state["evidence"]
        if raw_evidence is None:
            raise RuntimeError("Behavior evidence is missing from workflow state.")
        return BehaviorEvidence.model_validate(raw_evidence)

    @staticmethod
    def _feedback(state: WorkflowState) -> NeuroFeedback:
        raw_feedback = state["feedback"]
        if raw_feedback is None:
            raise RuntimeError("Neuro-behavioral feedback is missing from workflow state.")
        return NeuroFeedback.model_validate(raw_feedback)

    @staticmethod
    def _format_plan(plan: DailyPlan) -> str:
        lines = [f"GÜNLÜK TAAHHÜT HARİTASI — {plan.plan_date}", plan.headline, ""]
        for item in plan.items:
            duration = f" · {item.estimated_minutes} dk" if item.estimated_minutes else ""
            lines.append(
                f"{item.order}. {item.title} [{item.commitment_tier}/{item.priority}{duration}]"
            )
            if item.minimum_action:
                lines.append(f"   Minimum eylem: {item.minimum_action}")
        if plan.capacity_warning:
            lines.extend(("", f"Kapasite uyarısı: {plan.capacity_warning}"))
        lines.extend(("", "Bu plan ancak onayından sonra aktif taahhüt sayılacak."))
        return "\n".join(lines)[:4096]

    def _scheduled_start(self, item: DailyPlanItem) -> datetime | None:
        scheduled_start = item.scheduled_start
        if scheduled_start is None:
            return None
        if scheduled_start.tzinfo is None:
            return scheduled_start.replace(tzinfo=self.dependencies.settings.tz)
        return scheduled_start.astimezone(self.dependencies.settings.tz)

    @classmethod
    def _task_monitor_message(
        cls,
        *,
        idempotency_key: str,
        chat_id: str,
        item: DailyPlanItem,
        text: str,
    ) -> OutboundMessage:
        return OutboundMessage(
            idempotency_key=idempotency_key,
            chat_id=chat_id,
            text=text,
            buttons=cls._task_action_buttons(item.task_id),
        )

    @staticmethod
    def _format_due_reminder(item: DailyPlanItem, scheduled_start: datetime) -> str:
        minimum_action = item.minimum_action or "İlk fiziksel adımı belirle ve başlat."
        return (
            f"SAATİ GELDİ · {scheduled_start:%H:%M}\n{item.title}\n\n"
            f"Şimdi yalnızca Minimum Action'ı başlat: {minimum_action} "
            "Başladığında 'Başlattım'a bas; niyet değil, başlangıç kaydı oluştur."
        )

    @staticmethod
    def _format_start_check(item: DailyPlanItem, scheduled_start: datetime) -> str:
        minimum_action = item.minimum_action or "İlk fiziksel adımı belirle ve başlat."
        return (
            f"BAŞLANGIÇ KONTROLÜ · {scheduled_start:%H:%M}\n{item.title}\n\n"
            "Planlanan saat geçti ve henüz davranış kaydı yok. Sessizlik görevi ilerletmez: "
            f"şimdi {minimum_action} Sonra gerçek durumu aşağıdan bildir."
        )

    @staticmethod
    def _format_progress_check(item: DailyPlanItem, expected_minutes: int) -> str:
        definition = item.definition_of_done or "Somut tamamlanma kanıtını bildir."
        return (
            f"İLERLEME KONTROLÜ · {item.title}\n\n"
            f"Başlangıç kaydından {expected_minutes} dakika geçti; sonuç kaydı henüz yok. "
            f"Bittiyse kanıtı tamamla: {definition} Devam etmiyorsa gerçek engeli seç."
        )

    @staticmethod
    def _format_blocked_check(item: DailyPlanItem) -> str:
        minimum_action = item.minimum_action or "İlk fiziksel adımı belirle ve başlat."
        return (
            f"ENGEL TAKİBİ · {item.title}\n\n"
            "Engel kaydından sonra yeni davranış kanıtı gelmedi. Engel hâlâ gerçekse "
            f"somutlaştır; çözüldüyse zinciri yeniden başlat: {minimum_action}"
        )

    @classmethod
    def _day_status_message(
        cls,
        *,
        idempotency_key: str,
        chat_id: str,
        heading: str,
        plan: DailyPlan,
        open_items: list[DailyPlanItem],
        completed_count: int,
    ) -> OutboundMessage:
        if not open_items:
            text = (
                f"{heading}\n{completed_count}/{len(plan.items)} taahhüt tamamlandı. "
                "Bugünün planını eksiksiz davranış kanıtıyla kapattın; hangi başlangıç "
                "ipucunun en iyi çalıştığını yarın yeniden kullanmak için kaydet."
            )
            buttons: list[list[InlineButton]] = []
        else:
            visible_titles = ", ".join(item.title for item in open_items[:4])
            remaining_count = len(open_items) - 4
            if remaining_count > 0:
                visible_titles += f" ve {remaining_count} görev daha"
            first = open_items[0]
            minimum_action = first.minimum_action or "İlk fiziksel adımı belirle ve başlat."
            text = (
                f"{heading}\n{completed_count}/{len(plan.items)} taahhüt tamamlandı. "
                f"Açık kalanlar: {visible_titles}. Şimdi ilk açık görevin zincirini kapat: "
                f"{minimum_action}"
            )
            buttons = cls._task_action_buttons(first.task_id)
        return OutboundMessage(
            idempotency_key=idempotency_key,
            chat_id=chat_id,
            text=text[:4096],
            buttons=buttons,
        )

    @staticmethod
    def _format_feedback(feedback: NeuroFeedback) -> str:
        public_parts = (
            feedback.word_action_gap,
            feedback.immediate_intervention,
            feedback.evidence_request,
        )
        return " ".join(" ".join(part.split()) for part in public_parts)
