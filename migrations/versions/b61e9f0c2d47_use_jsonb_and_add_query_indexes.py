"""Use JSONB and add query-oriented indexes.

Revision ID: b61e9f0c2d47
Revises: e4d7a9136c84
Create Date: 2026-08-05 22:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b61e9f0c2d47"
down_revision: str | Sequence[str] | None = "e4d7a9136c84"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_COLUMNS = (
    ("daily_plans", "plan"),
    ("domain_events", "payload"),
    ("inbound_events", "payload"),
    ("outbox", "buttons"),
)


def upgrade() -> None:
    """Use PostgreSQL JSONB and add indexes for hot-path queries."""
    if op.get_context().dialect.name == "postgresql":
        for table_name, column_name in JSON_COLUMNS:
            op.alter_column(
                table_name,
                column_name,
                existing_type=sa.JSON(),
                type_=postgresql.JSONB(astext_type=sa.Text()),
                existing_nullable=False,
                postgresql_using=f"{column_name}::jsonb",
            )

    op.create_index(
        "ix_domain_events_user_task_occurred_at",
        "domain_events",
        ["user_id", "task_id", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_outbox_status_created_at",
        "outbox",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Restore generic JSON and remove the query-oriented indexes."""
    op.drop_index("ix_outbox_status_created_at", table_name="outbox")
    op.drop_index(
        "ix_domain_events_user_task_occurred_at",
        table_name="domain_events",
    )

    if op.get_context().dialect.name == "postgresql":
        for table_name, column_name in reversed(JSON_COLUMNS):
            op.alter_column(
                table_name,
                column_name,
                existing_type=postgresql.JSONB(astext_type=sa.Text()),
                type_=sa.JSON(),
                existing_nullable=False,
                postgresql_using=f"{column_name}::json",
            )
