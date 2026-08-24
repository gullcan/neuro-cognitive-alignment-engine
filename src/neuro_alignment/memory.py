from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from neuro_alignment.domain import DailyPlanItem, NormalizedInboundEvent, TaskAction

BEHAVIOR_VECTOR_DIMENSIONS = 32

ACTION_INDEX = {
    TaskAction.STARTED: 0,
    TaskAction.COMPLETED: 1,
    TaskAction.BLOCKED: 2,
    TaskAction.SKIPPED: 3,
    TaskAction.RESCHEDULED: 4,
}
COMMITMENT_INDEX = {"Core": 5, "Flexible": 6, "Optional": 7}
PRIORITY_INDEX = {"P1": 8, "P2": 9, "P3": 10}


def build_behavior_context(
    event: NormalizedInboundEvent,
    item: DailyPlanItem | None,
    timezone: ZoneInfo,
) -> tuple[dict[str, Any], list[float]]:
    """Encode observable planning context without claiming semantic understanding."""
    if event.action not in ACTION_INDEX:
        raise ValueError("Behavior memory requires a task behavior action.")

    local_time = event.occurred_at.astimezone(timezone)
    commitment_tier = item.commitment_tier if item else "Unknown"
    priority = item.priority if item else "Unknown"
    cognitive_load = item.cognitive_load if item else None
    estimated_minutes = item.estimated_minutes if item else None
    evidence_required = bool(item and item.definition_of_done)
    time_bucket = _time_bucket(local_time)

    context: dict[str, Any] = {
        "action": event.action.value,
        "commitment_tier": commitment_tier,
        "priority": priority,
        "weekday": local_time.weekday(),
        "hour": local_time.hour,
        "time_bucket": time_bucket,
        "cognitive_load": cognitive_load,
        "estimated_minutes": estimated_minutes,
        "evidence_required": evidence_required,
    }

    vector = [0.0] * BEHAVIOR_VECTOR_DIMENSIONS
    vector[ACTION_INDEX[event.action]] = 1.0
    if commitment_tier in COMMITMENT_INDEX:
        vector[COMMITMENT_INDEX[commitment_tier]] = 1.0
    if priority in PRIORITY_INDEX:
        vector[PRIORITY_INDEX[priority]] = 1.0
    vector[11 + local_time.weekday()] = 1.0
    vector[18 + time_bucket] = 1.0
    vector[22] = min(max((cognitive_load or 0) / 5, 0), 1)
    vector[23] = min(max((estimated_minutes or 0) / 240, 0), 1)
    vector[24] = 1.0 if evidence_required else 0.0
    vector[25] = 1.0 if local_time.weekday() >= 5 else 0.0
    vector[26] = {"Core": 1.0, "Flexible": 0.6, "Optional": 0.3}.get(commitment_tier, 0.0)
    vector[27] = {"P1": 1.0, "P2": 0.6, "P3": 0.3}.get(priority, 0.0)
    vector[28] = local_time.hour / 23
    vector[29] = 1.0
    return context, _normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True)) / denominator))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _time_bucket(value: datetime) -> int:
    if value.hour < 6:
        return 0
    if value.hour < 12:
        return 1
    if value.hour < 18:
        return 2
    return 3
