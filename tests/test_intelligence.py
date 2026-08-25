from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime

import pytest

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
from neuro_alignment.intelligence import (
    ResilientIntelligenceProvider,
    RuleBasedIntelligenceProvider,
)


class FailingIntelligenceProvider:
    async def build_daily_plan(
        self,
        *,
        tasks: Sequence[NotionTask],
        plan_date: str,
        user_id: str,
    ) -> DailyPlan:
        raise RuntimeError("simulated free-tier outage")

    async def generate_feedback(
        self,
        *,
        event: NormalizedInboundEvent,
        task_title: str,
        evidence: BehaviorEvidence,
        critique_notes: list[str],
    ) -> NeuroFeedback:
        raise RuntimeError("simulated rate limit")


@pytest.mark.asyncio
async def test_resilient_provider_falls_back_for_plan_and_feedback() -> None:
    provider = ResilientIntelligenceProvider(
        FailingIntelligenceProvider(),
        RuleBasedIntelligenceProvider(),
    )
    task = NotionTask(
        page_id="fallback-task",
        title="LLM yedeğini doğrula",
        scheduled_date=date(2026, 8, 25),
        commitment_tier="Core",
        priority="P1",
        definition_of_done="Yedek sağlayıcı plan üretti",
        minimum_action="Testi çalıştır",
    )
    plan = await provider.build_daily_plan(
        tasks=[task],
        plan_date="2026-08-25",
        user_id="owner",
    )
    event = NormalizedInboundEvent(
        event_id="fallback-feedback",
        event_type=InboundEventType.TELEGRAM_ACTION,
        source=EventSource.TELEGRAM,
        user_id="owner",
        occurred_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        task_id=task.page_id,
        action=TaskAction.STARTED,
    )
    feedback = await provider.generate_feedback(
        event=event,
        task_title=task.title,
        evidence=BehaviorEvidence(task_id=task.page_id, total_events=1),
        critique_notes=[],
    )

    assert plan.generated_by == "rule-based-local"
    assert plan.items[0].task_id == task.page_id
    assert "Başladın" in feedback.word_action_gap
