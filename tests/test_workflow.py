from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from neuro_alignment.config import Settings
from neuro_alignment.domain import (
    BehaviorEvidence,
    DailyPlan,
    EventSource,
    InboundEventType,
    NeuroFeedback,
    NormalizedInboundEvent,
    NotionTask,
    TaskAction,
)
from neuro_alignment.intelligence import RuleBasedIntelligenceProvider
from neuro_alignment.storage import (
    Database,
    EventRepository,
    MemoryRepository,
    OutboxRepository,
    PlanRepository,
)
from neuro_alignment.workflow import (
    UnsafeFeedbackError,
    WorkflowDependencies,
    WorkflowEngine,
)


class StubCommitmentSource:
    def __init__(self, tasks: list[NotionTask]) -> None:
        self.tasks = tasks
        self.requested_dates: list[date] = []

    async def fetch_daily_tasks(self, target_date: date) -> list[NotionTask]:
        self.requested_dates.append(target_date)
        return self.tasks


@dataclass(slots=True)
class WorkflowTestRuntime:
    engine: WorkflowEngine
    database: Database
    events: EventRepository
    memory: MemoryRepository
    plans: PlanRepository
    outbox: OutboxRepository
    notion: StubCommitmentSource


@pytest.mark.asyncio
async def test_daily_plan_is_persisted_queued_and_idempotent(tmp_path: Path) -> None:
    task = NotionTask(
        page_id="notion-task-1",
        title="LangGraph state tasarımını tamamla",
        scheduled_date=date(2026, 8, 5),
        commitment_tier="Core",
        priority="P1",
        definition_of_done="State ve route testleri geçiyor",
        minimum_action="State şemasını aç",
        estimated_minutes=90,
        cognitive_load=4,
    )
    async with build_runtime(tmp_path, tasks=[task]) as runtime:
        event = make_event(
            event_id="scheduler-2026-08-05",
            event_type=InboundEventType.DAILY_PLAN_REQUESTED,
            source=EventSource.SCHEDULER,
            payload={"plan_date": "2026-08-05"},
        )

        result = await runtime.engine.process(event)

        assert result.status == "plan_created"
        assert result.thread_id == "daily:owner:2026-08-05"
        assert result.queued_messages == 1
        assert runtime.notion.requested_dates == [date(2026, 8, 5)]

        pending = await runtime.outbox.pending()
        assert len(pending) == 1
        plan_message = pending[0][1]
        assert "GÜNLÜK TAAHHÜT HARİTASI" in plan_message.text
        assert "LangGraph state tasarımını tamamla" in plan_message.text
        assert [button.text for button in plan_message.buttons[0]] == [
            "Planı onayla",
            "Planı reddet",
        ]

        duplicate = await runtime.engine.process(event)
        assert duplicate.duplicate
        assert duplicate.status == "duplicate"
        assert duplicate.queued_messages == 0
        assert len(await runtime.outbox.pending()) == 1


@pytest.mark.asyncio
async def test_same_day_plan_refresh_gets_a_new_content_bound_approval_token(
    tmp_path: Path,
) -> None:
    original = NotionTask(
        page_id="refresh-task",
        title="İlk plan sürümü",
        scheduled_date=date(2026, 8, 5),
        commitment_tier="Core",
        priority="P1",
        definition_of_done="İlk sürüm tamamlandı",
        minimum_action="İlk sürümü aç",
    )
    async with build_runtime(tmp_path, tasks=[original]) as runtime:
        await runtime.engine.process(
            make_event(
                event_id="plan-version-1",
                event_type=InboundEventType.DAILY_PLAN_REQUESTED,
                source=EventSource.SCHEDULER,
                payload={"plan_date": "2026-08-05"},
            )
        )
        first_message = (await runtime.outbox.pending())[-1][1]
        first_token = first_message.buttons[0][0].callback_data.rsplit(":", 1)[1]

        runtime.notion.tasks = [original.model_copy(update={"title": "İkinci plan sürümü"})]
        await runtime.engine.process(
            make_event(
                event_id="plan-version-2",
                event_type=InboundEventType.DAILY_PLAN_REQUESTED,
                source=EventSource.SCHEDULER,
                payload={"plan_date": "2026-08-05"},
            )
        )
        second_message = (await runtime.outbox.pending())[-1][1]
        second_token = second_message.buttons[0][0].callback_data.rsplit(":", 1)[1]

        assert first_token != second_token
        assert "İkinci plan sürümü" in second_message.text


@pytest.mark.asyncio
async def test_plan_approval_follows_its_own_deterministic_branch(tmp_path: Path) -> None:
    task = NotionTask(
        page_id="notion-task-approval",
        title="Onaylanan görevi başlat",
        scheduled_date=date(2026, 8, 5),
        commitment_tier="Core",
        priority="P1",
        definition_of_done="Davranış düğmeleri görünür",
        minimum_action="Başlattım düğmesine bas",
    )
    async with build_runtime(tmp_path, tasks=[task]) as runtime:
        plan_event = make_event(
            event_id="plan-request-1",
            event_type=InboundEventType.DAILY_PLAN_REQUESTED,
            source=EventSource.SCHEDULER,
            payload={"plan_date": "2026-08-05"},
        )
        await runtime.engine.process(plan_event)
        plan_message = (await runtime.outbox.pending())[0][1]
        token = plan_message.buttons[0][0].callback_data.rsplit(":", maxsplit=1)[1]
        approval_event = make_event(
            event_id="telegram-update-100",
            event_type=InboundEventType.TELEGRAM_ACTION,
            source=EventSource.TELEGRAM,
            action=TaskAction.PLAN_APPROVED,
            payload={"approval_token": token, "chat_id": "12345"},
        )

        result = await runtime.engine.process(approval_event)

        assert result.status == "plan_approved"
        assert result.thread_id == f"plan-decision:owner:{token}"
        assert result.queued_messages == 2
        assert await runtime.plans.resolve_approval_token("owner", token) is None
        messages = [message for _record_id, message in await runtime.outbox.pending()]
        assert len(messages) == 3
        task_message = messages[-1]
        assert "AKTİF TAAHHÜT 1" in task_message.text
        assert "Onaylanan görevi başlat" in task_message.text
        assert [[button.text for button in row] for row in task_message.buttons] == [
            ["Başlattım", "Tamamladım"],
            ["Engellendim", "Atladım"],
            ["Erteledim"],
        ]
        assert task_message.buttons[0][0].callback_data == ("task:started:notion-task-approval")


@pytest.mark.asyncio
async def test_monitor_controls_every_scheduled_task_and_closes_the_day(tmp_path: Path) -> None:
    first_task = NotionTask(
        page_id="scheduled-task-1",
        title="İlk entegrasyonu doğrula",
        scheduled_date=date(2026, 8, 5),
        scheduled_start=datetime(2026, 8, 5, 9, 0, tzinfo=UTC),
        commitment_tier="Core",
        priority="P1",
        definition_of_done="İlk test geçti",
        minimum_action="İlk test dosyasını aç.",
        estimated_minutes=30,
    )
    second_task = NotionTask(
        page_id="scheduled-task-2",
        title="İkinci entegrasyonu doğrula",
        scheduled_date=date(2026, 8, 5),
        scheduled_start=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        commitment_tier="Core",
        priority="P1",
        definition_of_done="İkinci test geçti",
        minimum_action="İkinci test dosyasını aç.",
        estimated_minutes=30,
    )
    async with build_runtime(tmp_path, tasks=[first_task, second_task]) as runtime:
        await runtime.engine.process(
            make_event(
                event_id="scheduled-plan",
                event_type=InboundEventType.DAILY_PLAN_REQUESTED,
                source=EventSource.SCHEDULER,
                payload={"plan_date": "2026-08-05"},
                occurred_at=datetime(2026, 8, 5, 8, 0, tzinfo=UTC),
            )
        )
        plan_message = (await runtime.outbox.pending())[0][1]
        token = plan_message.buttons[0][0].callback_data.rsplit(":", maxsplit=1)[1]
        approval = await runtime.engine.process(
            make_event(
                event_id="scheduled-plan-approval",
                event_type=InboundEventType.TELEGRAM_ACTION,
                source=EventSource.TELEGRAM,
                action=TaskAction.PLAN_APPROVED,
                payload={"approval_token": token, "chat_id": "12345"},
                occurred_at=datetime(2026, 8, 5, 8, 5, tzinfo=UTC),
            )
        )
        assert approval.queued_messages == 1

        due_first = await runtime.engine.process(
            monitor_event("monitor-1205", datetime(2026, 8, 5, 9, 5, tzinfo=UTC))
        )
        assert due_first.queued_messages == 1
        duplicate = await runtime.engine.process(
            monitor_event("monitor-1205", datetime(2026, 8, 5, 9, 5, tzinfo=UTC))
        )
        assert duplicate.duplicate

        overdue_first = await runtime.engine.process(
            monitor_event("monitor-1220", datetime(2026, 8, 5, 9, 20, tzinfo=UTC))
        )
        assert overdue_first.queued_messages == 1
        await runtime.engine.process(
            make_event(
                event_id="complete-first",
                event_type=InboundEventType.TELEGRAM_ACTION,
                source=EventSource.TELEGRAM,
                task_id=first_task.page_id,
                action=TaskAction.COMPLETED,
                payload={"chat_id": "12345"},
                occurred_at=datetime(2026, 8, 5, 9, 21, tzinfo=UTC),
            )
        )

        due_second = await runtime.engine.process(
            monitor_event("monitor-1305", datetime(2026, 8, 5, 10, 5, tzinfo=UTC))
        )
        assert due_second.queued_messages == 1
        await runtime.engine.process(
            make_event(
                event_id="start-second",
                event_type=InboundEventType.TELEGRAM_ACTION,
                source=EventSource.TELEGRAM,
                task_id=second_task.page_id,
                action=TaskAction.STARTED,
                payload={"chat_id": "12345"},
                occurred_at=datetime(2026, 8, 5, 10, 6, tzinfo=UTC),
            )
        )
        progress_second = await runtime.engine.process(
            monitor_event("monitor-1340", datetime(2026, 8, 5, 10, 40, tzinfo=UTC))
        )
        assert progress_second.queued_messages == 1
        await runtime.engine.process(
            make_event(
                event_id="complete-second",
                event_type=InboundEventType.TELEGRAM_ACTION,
                source=EventSource.TELEGRAM,
                task_id=second_task.page_id,
                action=TaskAction.COMPLETED,
                payload={"chat_id": "12345"},
                occurred_at=datetime(2026, 8, 5, 16, 30, tzinfo=UTC),
            )
        )

        recovery = await runtime.engine.process(
            monitor_event("monitor-2000", datetime(2026, 8, 5, 17, 0, tzinfo=UTC))
        )
        final = await runtime.engine.process(
            monitor_event("monitor-2300", datetime(2026, 8, 5, 20, 0, tzinfo=UTC))
        )
        assert recovery.queued_messages == 1
        assert final.queued_messages == 1

        messages = [message for _record_id, message in await runtime.outbox.pending()]
        assert any("SAATİ GELDİ · 12:00" in message.text for message in messages)
        assert any("BAŞLANGIÇ KONTROLÜ · 12:00" in message.text for message in messages)
        assert any("SAATİ GELDİ · 13:00" in message.text for message in messages)
        assert any("İLERLEME KONTROLÜ" in message.text for message in messages)
        assert any("GÜN SONU YAKLAŞIYOR\n2/2" in message.text for message in messages)
        assert any("GÜNÜN SON KONTROLÜ\n2/2" in message.text for message in messages)


@pytest.mark.asyncio
async def test_text_checkin_is_recorded_and_receives_evidence_bounded_response(
    tmp_path: Path,
) -> None:
    async with build_runtime(tmp_path, tasks=[]) as runtime:
        event = make_event(
            event_id="telegram-checkin-1",
            event_type=InboundEventType.TELEGRAM_MESSAGE,
            source=EventSource.TELEGRAM,
            payload={"chat_id": "12345"},
        ).model_copy(update={"text": "enerji 7, odak 8"})

        result = await runtime.engine.process(event)

        assert result.status == "checkin_recorded"
        assert result.queued_messages == 1
        pending = await runtime.outbox.pending()
        assert len(pending) == 1
        assert pending[0][1].chat_id == "12345"
        assert "CHECK-IN KAYDEDİLDİ" in pending[0][1].text
        assert "henüz davranış kanıtı değildir" in pending[0][1].text


@pytest.mark.asyncio
async def test_behavior_feedback_uses_evidence_and_keeps_threads_isolated(
    tmp_path: Path,
) -> None:
    async with build_runtime(tmp_path, tasks=[]) as runtime:
        first = make_event(
            event_id="telegram-task-1",
            event_type=InboundEventType.TELEGRAM_ACTION,
            source=EventSource.TELEGRAM,
            task_id="task-a",
            action=TaskAction.STARTED,
            payload={"chat_id": "12345", "task_title": "Mimari testi"},
        )
        second = make_event(
            event_id="telegram-task-2",
            event_type=InboundEventType.TELEGRAM_ACTION,
            source=EventSource.TELEGRAM,
            task_id="task-b",
            action=TaskAction.COMPLETED,
            payload={"chat_id": "12345", "task_title": "Notion şeması"},
        )

        first_result = await runtime.engine.process(first)
        second_result = await runtime.engine.process(second)

        assert first_result.status == "feedback_generated"
        assert second_result.status == "feedback_generated"
        assert first_result.thread_id == "task:owner:task-a"
        assert second_result.thread_id == "task:owner:task-b"

        first_snapshot = await runtime.engine.graph.aget_state(
            {"configurable": {"thread_id": first_result.thread_id}}
        )
        second_snapshot = await runtime.engine.graph.aget_state(
            {"configurable": {"thread_id": second_result.thread_id}}
        )
        assert first_snapshot.values["event"]["event_id"] == "telegram-task-1"
        assert second_snapshot.values["event"]["event_id"] == "telegram-task-2"
        assert first_snapshot.values["evidence"]["counts"] == {"task.started": 1}
        assert second_snapshot.values["evidence"]["counts"] == {"task.completed": 1}
        similar = second_snapshot.values["evidence"]["similar_episodes"]
        assert len(similar) == 1
        assert similar[0]["task_id"] == "task-a"
        assert similar[0]["action"] == "started"
        assert 0 < similar[0]["similarity"] <= 1

        messages = [message for _record_id, message in await runtime.outbox.pending()]
        assert len(messages) == 2
        assert all("GÖZLENEN KANIT" not in message.text for message in messages)
        assert all("NÖRO-BİLİŞSEL BAĞLAM" not in message.text for message in messages)
        assert all("\n" not in message.text for message in messages)
        assert all(len(message.text) <= 902 for message in messages)
        assert "niyet bitti; davranış başladı" in messages[0].text
        assert "Kendine güvenmek için gereken kanıtı sen ürettin" in messages[1].text

        internal_feedback = NeuroFeedback.model_validate(second_snapshot.values["feedback"])
        assert "Güncel davranış olayı" in internal_feedback.observed_evidence
        assert "öğrenme süreçlerini" in internal_feedback.neuro_context


@pytest.mark.asyncio
async def test_skipped_feedback_is_short_personalized_and_actionable() -> None:
    provider = RuleBasedIntelligenceProvider()
    feedback = await provider.generate_feedback(
        event=make_event(
            event_id="telegram-task-skipped",
            event_type=InboundEventType.TELEGRAM_ACTION,
            source=EventSource.TELEGRAM,
            task_id="task-skipped",
            action=TaskAction.SKIPPED,
            payload={"chat_id": "12345", "task_title": "Notion entegrasyonunu doğrula"},
        ),
        task_title="Notion entegrasyonunu doğrula",
        evidence=BehaviorEvidence(
            task_id="task-skipped",
            total_events=6,
            counts={"task.completed": 1, "task.skipped": 1, "task.rescheduled": 3},
            has_sufficient_history=True,
        ),
        critique_notes=[],
    )

    message = WorkflowEngine._format_feedback(feedback)

    assert "atlama/erteleme kaydı 4'e çıktı" in message
    assert "10 dakikalık Minimum Action'ı şimdi uygula" in message
    assert "doğrulanabilir engeli bildir" in message
    assert "\n" not in message
    assert "GÖZLENEN KANIT" not in message
    assert len(message) <= 902


@pytest.mark.asyncio
async def test_safety_critic_retries_once_then_fails_closed(tmp_path: Path) -> None:
    unsafe_provider = AlwaysUnsafeIntelligenceProvider()
    async with build_runtime(tmp_path, tasks=[], intelligence=unsafe_provider) as runtime:
        event = make_event(
            event_id="unsafe-feedback-1",
            event_type=InboundEventType.TELEGRAM_ACTION,
            source=EventSource.TELEGRAM,
            task_id="task-unsafe",
            action=TaskAction.BLOCKED,
            payload={"chat_id": "12345"},
        )

        with pytest.raises(UnsafeFeedbackError):
            await runtime.engine.process(event)

        assert unsafe_provider.feedback_calls == 2
        assert await runtime.outbox.pending() == []
        evidence = await runtime.events.build_evidence(
            user_id="owner",
            task_id="task-unsafe",
        )
        assert evidence.total_events == 1

        with pytest.raises(UnsafeFeedbackError):
            await runtime.engine.process(event)

        assert unsafe_provider.feedback_calls == 4
        retried_evidence = await runtime.events.build_evidence(
            user_id="owner",
            task_id="task-unsafe",
        )
        assert retried_evidence.total_events == 1


class AlwaysUnsafeIntelligenceProvider:
    def __init__(self) -> None:
        self.feedback_calls = 0

    async def build_daily_plan(
        self,
        *,
        tasks: Sequence[NotionTask],
        plan_date: str,
        user_id: str,
    ) -> DailyPlan:
        raise AssertionError("Daily planning is not expected in this test.")

    async def generate_feedback(
        self,
        *,
        event: NormalizedInboundEvent,
        task_title: str,
        evidence: BehaviorEvidence,
        critique_notes: list[str],
    ) -> NeuroFeedback:
        self.feedback_calls += 1
        return NeuroFeedback(
            observed_evidence="Engel bildirildi.",
            behavioral_pattern="Tek olay var.",
            interpretation="Dopamin seviyeni ölçtüm.",
            neuro_context="Bu klinik bir ölçümdür.",
            word_action_gap="Eylem durdu.",
            immediate_intervention="Bir dakika başla.",
            follow_up_minutes=1,
            evidence_request="Çıktıyı gönder.",
            confidence=0.4,
            evidence_refs=evidence.evidence_refs,
        )


@asynccontextmanager
async def build_runtime(
    tmp_path: Path,
    *,
    tasks: list[NotionTask],
    intelligence: RuleBasedIntelligenceProvider | AlwaysUnsafeIntelligenceProvider | None = None,
) -> AsyncIterator[WorkflowTestRuntime]:
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'workflow.db'}",
        checkpoint_backend="memory",
        telegram_chat_id="12345",
        openai_api_key=None,
    )
    database = Database(settings.database_url)
    await database.create_schema()
    events = EventRepository(database)
    memory = MemoryRepository(database)
    plans = PlanRepository(database)
    outbox = OutboxRepository(database)
    notion = StubCommitmentSource(tasks)
    engine = WorkflowEngine(
        WorkflowDependencies(
            settings=settings,
            events=events,
            memory=memory,
            plans=plans,
            outbox=outbox,
            notion=notion,
            intelligence=intelligence or RuleBasedIntelligenceProvider(),
        ),
        InMemorySaver(),
    )
    try:
        yield WorkflowTestRuntime(
            engine=engine,
            database=database,
            events=events,
            memory=memory,
            plans=plans,
            outbox=outbox,
            notion=notion,
        )
    finally:
        await database.close()


def make_event(
    *,
    event_id: str,
    event_type: InboundEventType,
    source: EventSource,
    task_id: str | None = None,
    action: TaskAction | None = None,
    payload: dict[str, object] | None = None,
    occurred_at: datetime | None = None,
) -> NormalizedInboundEvent:
    return NormalizedInboundEvent(
        event_id=event_id,
        event_type=event_type,
        source=source,
        user_id="owner",
        occurred_at=occurred_at or datetime(2026, 8, 5, 8, 0, tzinfo=UTC),
        task_id=task_id,
        action=action,
        payload=payload or {},
    )


def monitor_event(event_id: str, occurred_at: datetime) -> NormalizedInboundEvent:
    return make_event(
        event_id=event_id,
        event_type=InboundEventType.TASK_MONITOR_TICK,
        source=EventSource.SCHEDULER,
        payload={"plan_date": "2026-08-05"},
        occurred_at=occurred_at,
    )
