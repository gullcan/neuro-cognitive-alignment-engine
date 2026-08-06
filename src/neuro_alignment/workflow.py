from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
    DomainEventType,
    InboundEventType,
    InlineButton,
    NeuroFeedback,
    NormalizedInboundEvent,
    NotionTask,
    OutboundMessage,
    ProcessResult,
    TaskAction,
)
from neuro_alignment.intelligence import IntelligenceProvider
from neuro_alignment.storage import EventRepository, OutboxRepository, PlanRepository

logger = structlog.get_logger()

WorkflowRoute = Literal["daily_plan", "plan_decision", "behavior", "checkin"]
RouteDecision = Literal["duplicate", "daily_plan", "plan_decision", "behavior", "checkin"]
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
    evidence: dict[str, Any] | None
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
                "plan_decision": "apply_plan_decision",
                "behavior": "record_behavior",
                "checkin": "record_checkin",
            },
        )

        builder.add_edge("load_commitments", "planner_agent")
        builder.add_edge("planner_agent", "persist_plan")
        builder.add_edge("persist_plan", "queue_plan")
        builder.add_edge("queue_plan", "complete_event")

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
        _plan_thread_id, plan_date, current_status = resolved
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
                "Plan onaylandı. Gün içindeki davranış olayları bu taahhütlere bağlanacak."
                if status == "approved"
                else "Plan reddedildi. Yeni plan oluşturulmadan taahhüt seti aktif sayılmayacak."
            )
            queued = await self.dependencies.outbox.enqueue(
                [
                    OutboundMessage(
                        idempotency_key=f"plan-decision:{event.event_id}",
                        chat_id=chat_id,
                        text=decision_text,
                    )
                ]
            )
        return {
            "queued_messages": queued,
            "status": f"plan_{status}",
        }

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
        return {}

    async def _retrieve_evidence(self, state: WorkflowState) -> StateUpdate:
        event = self._event(state)
        if event.task_id is None:
            raise WorkflowInputError("Evidence retrieval requires task_id.")
        evidence = await self.dependencies.events.build_evidence(
            user_id=event.user_id,
            task_id=event.task_id,
        )
        task_title = await self.dependencies.plans.task_title(
            event.user_id,
            event.task_id,
        )
        if task_title is None:
            task_title = str(event.payload.get("task_title") or event.task_id)
        return {
            "evidence": evidence.model_dump(mode="json"),
            "task_title": task_title,
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
            "dopamin seviyeni ölçtüm",
            "dopamin reseptörlerin zayıf",
            "prefrontal korteksin güçlendi",
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
            evidence=None,
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

    @staticmethod
    def _format_feedback(feedback: NeuroFeedback) -> str:
        text = "\n\n".join(
            (
                f"GÖZLENEN KANIT\n{feedback.observed_evidence}",
                f"DAVRANIŞ ÖRÜNTÜSÜ\n{feedback.behavioral_pattern}",
                f"YORUM SINIRI\n{feedback.interpretation}",
                f"NÖRO-BİLİŞSEL BAĞLAM\n{feedback.neuro_context}",
                f"SÖZ–EYLEM AÇIĞI\n{feedback.word_action_gap}",
                f"ŞİMDİKİ MÜDAHALE\n{feedback.immediate_intervention}",
                f"KANIT İSTEĞİ\n{feedback.evidence_request}",
                f"Takip: {feedback.follow_up_minutes} dakika · Güven: {feedback.confidence:.0%}",
            )
        )
        return text[:4096]
