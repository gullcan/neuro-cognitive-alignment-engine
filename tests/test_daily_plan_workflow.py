from __future__ import annotations

from pathlib import Path


def test_daily_plan_workflow_is_scheduled_secure_and_idempotent() -> None:
    workflow = Path(".github/workflows/daily-plan.yml").read_text()

    assert 'cron: "35 7 * * *"' in workflow
    assert 'timezone: "Europe/Istanbul"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "secrets.RENDER_INTERNAL_API_KEY" in workflow
    assert 'request_id="github-actions-${plan_date}"' in workflow
    assert 'request_id="github-actions-manual-${plan_date}-${GITHUB_RUN_ID}"' in workflow
    assert "--retry-all-errors" in workflow
    assert "X-Internal-Api-Key: ${INTERNAL_API_KEY}" in workflow
    assert "change-me" not in workflow


def test_task_monitor_workflow_runs_every_fifteen_minutes_with_same_secret() -> None:
    workflow = Path(".github/workflows/task-monitor.yml").read_text()

    assert 'cron: "*/15 * * * *"' in workflow
    assert 'timezone: "Europe/Istanbul"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "secrets.RENDER_INTERNAL_API_KEY" in workflow
    assert "/v1/internal/scheduler/daily-plan" in workflow
    assert "/v1/internal/scheduler/task-monitor" in workflow
    assert 'sync_request_id="github-actions-sync-${cycle_id}"' in workflow
    assert 'plan_date="$(TZ=Europe/Istanbul date +%F)"' in workflow
    assert "--retry-all-errors" in workflow
    assert "X-Internal-Api-Key: ${INTERNAL_API_KEY}" in workflow
    assert "change-me" not in workflow
