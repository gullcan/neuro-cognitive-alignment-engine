from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from neuro_alignment.domain import (
    BehaviorEvidence,
    DailyPlan,
    EvidenceEvent,
    NormalizedInboundEvent,
    OutboundMessage,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class InboundEventRecord(Base):
    __tablename__ = "inbound_events"
    __table_args__ = (
        UniqueConstraint("source", "source_event_id", name="uq_inbound_source_event"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_event_id: Mapped[str] = mapped_column(String(160), nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="processing", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)


class DomainEventRecord(Base):
    __tablename__ = "domain_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    task_id: Mapped[str | None] = mapped_column(String(100), index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class DailyPlanRecord(Base):
    __tablename__ = "daily_plans"
    __table_args__ = (
        UniqueConstraint("user_id", "plan_date", name="uq_daily_plan_user_date"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    plan_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(180), nullable=False)
    approval_token: Mapped[str] = mapped_column(
        String(24), nullable=False, unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class OutboxRecord(Base):
    __tablename__ = "outbox"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(
        String(180), nullable=False, unique=True, index=True
    )
    chat_id: Mapped[str] = mapped_column(String(100), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    buttons: Mapped[list[list[dict[str, str]]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(String(100))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Database:
    def __init__(self, url: str) -> None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine: AsyncEngine = create_async_engine(
            url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def create_schema(self) -> None:
        if self.engine.url.get_backend_name() == "sqlite":
            database_path = self.engine.url.database
            if database_path and database_path != ":memory:":
                Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session


class EventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def claim_inbound(
        self,
        event: NormalizedInboundEvent,
        *,
        lease_seconds: int = 120,
    ) -> bool:
        now = utc_now()
        lease_until = now + timedelta(seconds=lease_seconds)
        async with self.database.session() as session, session.begin():
            record = await session.scalar(
                select(InboundEventRecord).where(
                    InboundEventRecord.source == event.source.value,
                    InboundEventRecord.source_event_id == event.event_id,
                )
            )
            if record is None:
                session.add(
                    InboundEventRecord(
                        id=str(uuid4()),
                        source=event.source.value,
                        source_event_id=event.event_id,
                        event_type=event.event_type.value,
                        user_id=event.user_id,
                        payload=event.model_dump(mode="json"),
                        status="processing",
                        attempts=1,
                        received_at=now,
                        lease_until=lease_until,
                    )
                )
                return True

            if record.status == "completed":
                return False
            if (
                record.status == "processing"
                and record.lease_until is not None
                and self._as_utc(record.lease_until) > now
            ):
                return False

            record.status = "processing"
            record.attempts += 1
            record.lease_until = lease_until
            record.last_error = None
            return True

    async def complete_inbound(self, event: NormalizedInboundEvent) -> None:
        async with self.database.session() as session, session.begin():
            record = await self._get_inbound(session, event)
            record.status = "completed"
            record.completed_at = utc_now()
            record.lease_until = None

    async def fail_inbound(
        self,
        event: NormalizedInboundEvent,
        error: Exception,
    ) -> None:
        async with self.database.session() as session, session.begin():
            record = await self._get_inbound(session, event)
            record.status = "failed"
            record.last_error = str(error)[:2000]
            record.lease_until = None

    async def append_domain_event(
        self,
        *,
        event_type: str,
        user_id: str,
        task_id: str | None,
        source: str,
        occurred_at: datetime,
        payload: dict[str, Any] | None = None,
    ) -> str:
        event_id = str(uuid4())
        async with self.database.session() as session, session.begin():
            session.add(
                DomainEventRecord(
                    id=event_id,
                    event_type=event_type,
                    user_id=user_id,
                    task_id=task_id,
                    source=source,
                    occurred_at=occurred_at,
                    payload=payload or {},
                )
            )
        return event_id

    async def build_evidence(
        self,
        *,
        user_id: str,
        task_id: str,
        limit: int = 20,
    ) -> BehaviorEvidence:
        async with self.database.session() as session:
            rows = (
                await session.scalars(
                    select(DomainEventRecord)
                    .where(
                        DomainEventRecord.user_id == user_id,
                        DomainEventRecord.task_id == task_id,
                    )
                    .order_by(DomainEventRecord.occurred_at.desc())
                    .limit(limit)
                )
            ).all()

        counts = Counter(row.event_type for row in rows)
        evidence_events = [
            EvidenceEvent(
                event_id=row.id,
                event_type=row.event_type,
                occurred_at=self._as_utc(row.occurred_at),
                task_id=row.task_id,
                source=row.source,
            )
            for row in rows
        ]
        resolved_events = sum(
            counts[event_type]
            for event_type in (
                "task.completed",
                "task.blocked",
                "task.skipped",
                "task.rescheduled",
            )
        )
        return BehaviorEvidence(
            task_id=task_id,
            total_events=len(rows),
            counts=dict(counts),
            recent_events=evidence_events,
            evidence_refs=[row.id for row in rows],
            has_sufficient_history=resolved_events >= 3,
        )

    async def _get_inbound(
        self,
        session: AsyncSession,
        event: NormalizedInboundEvent,
    ) -> InboundEventRecord:
        record = await session.scalar(
            select(InboundEventRecord).where(
                InboundEventRecord.source == event.source.value,
                InboundEventRecord.source_event_id == event.event_id,
            )
        )
        if record is None:
            raise RuntimeError(f"Inbound event disappeared: {event.event_id}")
        return record

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PlanRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def save(
        self,
        user_id: str,
        plan: DailyPlan,
        thread_id: str,
        status: str = "pending",
    ) -> str:
        now = utc_now()
        approval_token = hashlib.sha256(thread_id.encode()).hexdigest()[:20]
        async with self.database.session() as session, session.begin():
            record = await session.scalar(
                select(DailyPlanRecord).where(
                    DailyPlanRecord.user_id == user_id,
                    DailyPlanRecord.plan_date == plan.plan_date,
                )
            )
            if record is None:
                session.add(
                    DailyPlanRecord(
                        id=str(uuid4()),
                        user_id=user_id,
                        plan_date=plan.plan_date,
                        thread_id=thread_id,
                        approval_token=approval_token,
                        status=status,
                        plan=plan.model_dump(mode="json"),
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                record.thread_id = thread_id
                record.approval_token = approval_token
                record.status = status
                record.plan = plan.model_dump(mode="json")
                record.updated_at = now
        return approval_token

    async def update_status(self, user_id: str, plan_date: date, status: str) -> None:
        async with self.database.session() as session, session.begin():
            record = await session.scalar(
                select(DailyPlanRecord).where(
                    DailyPlanRecord.user_id == user_id,
                    DailyPlanRecord.plan_date == plan_date,
                )
            )
            if record is None:
                raise LookupError(f"No plan exists for {user_id} on {plan_date}")
            record.status = status
            record.updated_at = utc_now()

    async def task_title(self, user_id: str, task_id: str) -> str | None:
        async with self.database.session() as session:
            records = (
                await session.scalars(
                    select(DailyPlanRecord)
                    .where(DailyPlanRecord.user_id == user_id)
                    .order_by(DailyPlanRecord.plan_date.desc())
                    .limit(14)
                )
            ).all()
        for record in records:
            plan = DailyPlan.model_validate(record.plan)
            for item in plan.items:
                if item.task_id == task_id:
                    return item.title
        return None

    async def resolve_approval_token(
        self,
        user_id: str,
        approval_token: str,
    ) -> tuple[str, date] | None:
        async with self.database.session() as session:
            record = await session.scalar(
                select(DailyPlanRecord).where(
                    DailyPlanRecord.user_id == user_id,
                    DailyPlanRecord.approval_token == approval_token,
                    DailyPlanRecord.status == "pending",
                )
            )
        if record is None:
            return None
        return record.thread_id, record.plan_date


class OutboxRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def enqueue(self, messages: Sequence[OutboundMessage]) -> int:
        queued = 0
        async with self.database.session() as session, session.begin():
            for message in messages:
                existing = await session.scalar(
                    select(OutboxRecord.id).where(
                        OutboxRecord.idempotency_key == message.idempotency_key
                    )
                )
                if existing is not None:
                    continue
                session.add(
                    OutboxRecord(
                        id=str(uuid4()),
                        idempotency_key=message.idempotency_key,
                        chat_id=message.chat_id,
                        text=message.text,
                        buttons=[
                            [button.model_dump(mode="json") for button in row]
                            for row in message.buttons
                        ],
                        status="pending",
                        attempts=0,
                    )
                )
                queued += 1
        return queued

    async def pending(self, limit: int = 20) -> list[tuple[str, OutboundMessage]]:
        async with self.database.session() as session:
            rows = (
                await session.scalars(
                    select(OutboxRecord)
                    .where(OutboxRecord.status.in_(("pending", "failed")))
                    .order_by(OutboxRecord.created_at)
                    .limit(limit)
                )
            ).all()
        return [
            (
                row.id,
                OutboundMessage(
                    idempotency_key=row.idempotency_key,
                    chat_id=row.chat_id,
                    text=row.text,
                    buttons=row.buttons,
                ),
            )
            for row in rows
        ]

    async def mark_sent(self, record_id: str, provider_message_id: str) -> None:
        async with self.database.session() as session, session.begin():
            record = await session.get(OutboxRecord, record_id)
            if record is None:
                raise LookupError(record_id)
            record.status = "sent"
            record.attempts += 1
            record.provider_message_id = provider_message_id
            record.last_error = None
            record.updated_at = utc_now()

    async def mark_failed(self, record_id: str, error: Exception) -> None:
        async with self.database.session() as session, session.begin():
            record = await session.get(OutboxRecord, record_id)
            if record is None:
                raise LookupError(record_id)
            record.status = "failed"
            record.attempts += 1
            record.last_error = str(error)[:2000]
            record.updated_at = utc_now()
