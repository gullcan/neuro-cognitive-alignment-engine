"""Add behavioral vector memory.

Revision ID: 7a2f4c9d1e30
Revises: b61e9f0c2d47
Create Date: 2026-08-24 13:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "7a2f4c9d1e30"
down_revision: str | Sequence[str] | None = "b61e9f0c2d47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VECTOR_DIMENSIONS = 32


def upgrade() -> None:
    """Create explainable cross-task behavioral memory storage."""
    is_postgresql = op.get_context().dialect.name == "postgresql"
    if is_postgresql:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "behavioral_memories",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_event_id", sa.String(length=160), nullable=False),
        sa.Column("user_id", sa.String(length=100), nullable=False),
        sa.Column("task_id", sa.String(length=100), nullable=False),
        sa.Column("task_title", sa.Text(), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "context",
            postgresql.JSONB(astext_type=sa.Text()) if is_postgresql else sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "embedding",
            Vector(VECTOR_DIMENSIONS) if is_postgresql else sa.JSON(),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_behavioral_memories")),
        sa.UniqueConstraint(
            "source_event_id",
            name=op.f("uq_behavioral_memories_source_event_id"),
        ),
    )
    op.create_index(
        op.f("ix_behavioral_memories_action"),
        "behavioral_memories",
        ["action"],
        unique=False,
    )
    op.create_index(
        op.f("ix_behavioral_memories_occurred_at"),
        "behavioral_memories",
        ["occurred_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_behavioral_memories_task_id"),
        "behavioral_memories",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_behavioral_memories_user_id"),
        "behavioral_memories",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_behavioral_memories_user_occurred_at",
        "behavioral_memories",
        ["user_id", "occurred_at"],
        unique=False,
    )
    if is_postgresql:
        op.create_index(
            "ix_behavioral_memories_embedding_hnsw",
            "behavioral_memories",
            ["embedding"],
            unique=False,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


def downgrade() -> None:
    """Remove behavioral memory while retaining the shared vector extension."""
    if op.get_context().dialect.name == "postgresql":
        op.drop_index(
            "ix_behavioral_memories_embedding_hnsw",
            table_name="behavioral_memories",
            postgresql_using="hnsw",
        )
    op.drop_index(
        "ix_behavioral_memories_user_occurred_at",
        table_name="behavioral_memories",
    )
    op.drop_index(op.f("ix_behavioral_memories_user_id"), table_name="behavioral_memories")
    op.drop_index(op.f("ix_behavioral_memories_task_id"), table_name="behavioral_memories")
    op.drop_index(
        op.f("ix_behavioral_memories_occurred_at"),
        table_name="behavioral_memories",
    )
    op.drop_index(op.f("ix_behavioral_memories_action"), table_name="behavioral_memories")
    op.drop_table("behavioral_memories")
