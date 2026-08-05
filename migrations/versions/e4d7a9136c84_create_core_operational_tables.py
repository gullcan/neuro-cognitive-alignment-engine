"""Create core operational tables.

This base revision mirrors the schema formerly produced by ``Base.metadata.create_all``
so an existing, verified legacy database can be stamped here before later upgrades.

Revision ID: e4d7a9136c84
Revises:
Create Date: 2026-08-05 20:27:13.904523

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e4d7a9136c84"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the legacy-compatible operational schema."""
    op.create_table(
        "daily_plans",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=100), nullable=False),
        sa.Column("plan_date", sa.Date(), nullable=False),
        sa.Column("thread_id", sa.String(length=180), nullable=False),
        sa.Column("approval_token", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("plan", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "plan_date",
            name="uq_daily_plan_user_date",
        ),
    )
    op.create_index(
        op.f("ix_daily_plans_approval_token"),
        "daily_plans",
        ["approval_token"],
        unique=True,
    )
    op.create_index(
        op.f("ix_daily_plans_plan_date"),
        "daily_plans",
        ["plan_date"],
        unique=False,
    )
    op.create_index(
        op.f("ix_daily_plans_user_id"),
        "daily_plans",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "domain_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_type", sa.String(length=80), nullable=False),
        sa.Column("user_id", sa.String(length=100), nullable=False),
        sa.Column("task_id", sa.String(length=100), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_domain_events_event_type"),
        "domain_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_domain_events_occurred_at"),
        "domain_events",
        ["occurred_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_domain_events_task_id"),
        "domain_events",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_domain_events_user_id"),
        "domain_events",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "inbound_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("source_event_id", sa.String(length=160), nullable=False),
        sa.Column("event_type", sa.String(length=80), nullable=False),
        sa.Column("user_id", sa.String(length=100), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source",
            "source_event_id",
            name="uq_inbound_source_event",
        ),
    )
    op.create_index(
        op.f("ix_inbound_events_user_id"),
        "inbound_events",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "outbox",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=180), nullable=False),
        sa.Column("chat_id", sa.String(length=100), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("buttons", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("provider_message_id", sa.String(length=100), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_outbox_idempotency_key"),
        "outbox",
        ["idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    """Drop the legacy-compatible operational schema."""
    op.drop_index(op.f("ix_outbox_idempotency_key"), table_name="outbox")
    op.drop_table("outbox")

    op.drop_index(op.f("ix_inbound_events_user_id"), table_name="inbound_events")
    op.drop_table("inbound_events")

    op.drop_index(op.f("ix_domain_events_user_id"), table_name="domain_events")
    op.drop_index(op.f("ix_domain_events_task_id"), table_name="domain_events")
    op.drop_index(op.f("ix_domain_events_occurred_at"), table_name="domain_events")
    op.drop_index(op.f("ix_domain_events_event_type"), table_name="domain_events")
    op.drop_table("domain_events")

    op.drop_index(op.f("ix_daily_plans_user_id"), table_name="daily_plans")
    op.drop_index(op.f("ix_daily_plans_plan_date"), table_name="daily_plans")
    op.drop_index(
        op.f("ix_daily_plans_approval_token"),
        table_name="daily_plans",
    )
    op.drop_table("daily_plans")
