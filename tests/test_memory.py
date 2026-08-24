from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from neuro_alignment.domain import (
    DailyPlanItem,
    EventSource,
    InboundEventType,
    NormalizedInboundEvent,
    TaskAction,
)
from neuro_alignment.memory import BEHAVIOR_VECTOR_DIMENSIONS, build_behavior_context
from neuro_alignment.storage import Database, MemoryRepository


def test_behavior_context_vector_is_explainable_normalized_and_deterministic() -> None:
    event = _event("event-1", TaskAction.BLOCKED, hour=9)
    item = _item("task-1", cognitive_load=4)

    first_context, first_vector = build_behavior_context(event, item, ZoneInfo("Europe/Istanbul"))
    second_context, second_vector = build_behavior_context(event, item, ZoneInfo("Europe/Istanbul"))

    assert first_context == second_context
    assert first_vector == second_vector
    assert len(first_vector) == BEHAVIOR_VECTOR_DIMENSIONS
    assert sum(value * value for value in first_vector) == pytest.approx(1.0)
    assert first_context["action"] == "blocked"
    assert first_context["commitment_tier"] == "Core"
    assert first_context["priority"] == "P1"


@pytest.mark.asyncio
async def test_memory_repository_finds_similar_cross_task_episode(tmp_path: Path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}")
    await database.create_schema()
    repository = MemoryRepository(database)
    try:
        first_event = _event("event-1", TaskAction.BLOCKED, hour=9)
        first_context, first_vector = build_behavior_context(
            first_event,
            _item("task-1", cognitive_load=4),
            ZoneInfo("Europe/Istanbul"),
        )
        await repository.store_episode(
            source_event_id=first_event.event_id,
            user_id="owner",
            task_id="task-1",
            task_title="İlk zor görev",
            action=TaskAction.BLOCKED,
            occurred_at=first_event.occurred_at,
            context=first_context,
            embedding=first_vector,
        )

        query_event = _event("event-2", TaskAction.BLOCKED, hour=9)
        _, query_vector = build_behavior_context(
            query_event,
            _item("task-2", cognitive_load=4),
            ZoneInfo("Europe/Istanbul"),
        )
        matches = await repository.similar_episodes(
            user_id="owner",
            embedding=query_vector,
            exclude_source_event_id=query_event.event_id,
        )

        assert len(matches) == 1
        assert matches[0].task_id == "task-1"
        assert matches[0].action == TaskAction.BLOCKED
        assert matches[0].similarity == pytest.approx(1.0)
    finally:
        await database.close()


def _event(event_id: str, action: TaskAction, *, hour: int) -> NormalizedInboundEvent:
    return NormalizedInboundEvent(
        event_id=event_id,
        event_type=InboundEventType.TELEGRAM_ACTION,
        source=EventSource.TELEGRAM,
        user_id="owner",
        occurred_at=datetime(2026, 8, 24, hour, tzinfo=UTC),
        task_id=f"task-{event_id}",
        action=action,
    )


def _item(task_id: str, *, cognitive_load: int) -> DailyPlanItem:
    return DailyPlanItem(
        task_id=task_id,
        title=f"Görev {task_id}",
        order=1,
        commitment_tier="Core",
        priority="P1",
        cognitive_load=cognitive_load,
        definition_of_done="Çıktı kaydedildi",
        minimum_action="Dosyayı aç",
        rationale="Test bağlamı",
    )
