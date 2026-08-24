from __future__ import annotations

from pathlib import Path


def test_daily_plan_workflow_is_scheduled_secure_and_idempotent() -> None:
    workflow = Path(".github/workflows/daily-plan.yml").read_text()

    assert 'cron: "35 7 * * *"' in workflow
    assert 'timezone: "Europe/Istanbul"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "secrets.RENDER_INTERNAL_API_KEY" in workflow
    assert 'request_id="github-actions-${plan_date}"' in workflow
    assert "--retry-all-errors" in workflow
    assert "X-Internal-Api-Key: ${INTERNAL_API_KEY}" in workflow
    assert "change-me" not in workflow
