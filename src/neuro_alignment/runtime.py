from __future__ import annotations

from dataclasses import dataclass

import httpx

from neuro_alignment import __version__
from neuro_alignment.checkpointing import CheckpointManager
from neuro_alignment.config import Settings
from neuro_alignment.integrations import (
    NotionClient,
    TelegramClient,
    TelegramUpdateParser,
)
from neuro_alignment.intelligence import (
    IntelligenceProvider,
    OpenAIIntelligenceProvider,
    RuleBasedIntelligenceProvider,
)
from neuro_alignment.storage import (
    Database,
    EventRepository,
    OutboxRepository,
    PlanRepository,
)
from neuro_alignment.workflow import (
    WorkflowDependencies,
    WorkflowEngine,
)


@dataclass(slots=True)
class AppServices:
    """Application-scoped resources with one coordinated lifecycle."""

    settings: Settings
    database: Database
    http_client: httpx.AsyncClient
    events: EventRepository
    plans: PlanRepository
    outbox: OutboxRepository
    notion: NotionClient
    telegram: TelegramClient
    telegram_updates: TelegramUpdateParser
    intelligence: IntelligenceProvider
    checkpoints: CheckpointManager | None = None
    workflow: WorkflowEngine | None = None

    @classmethod
    def build(cls, settings: Settings) -> AppServices:
        database = Database(settings.database_url)
        http_client = httpx.AsyncClient(
            headers={"User-Agent": f"neuro-cognitive-alignment-engine/{__version__}"},
            timeout=httpx.Timeout(20.0, connect=5.0),
        )
        intelligence: IntelligenceProvider
        if settings.openai_api_key:
            intelligence = OpenAIIntelligenceProvider(settings)
        else:
            intelligence = RuleBasedIntelligenceProvider()

        return cls(
            settings=settings,
            database=database,
            http_client=http_client,
            events=EventRepository(database),
            plans=PlanRepository(database),
            outbox=OutboxRepository(database),
            notion=NotionClient(settings, http_client),
            telegram=TelegramClient(settings, http_client),
            telegram_updates=TelegramUpdateParser(settings),
            intelligence=intelligence,
        )

    async def start(self) -> None:
        """Start persistence resources and compile the application graph once."""
        if self.workflow is not None:
            return
        checkpoints = CheckpointManager(self.settings)
        self.checkpoints = checkpoints
        checkpointer = await checkpoints.start()
        self.workflow = WorkflowEngine(
            WorkflowDependencies(
                settings=self.settings,
                events=self.events,
                plans=self.plans,
                outbox=self.outbox,
                notion=self.notion,
                intelligence=self.intelligence,
            ),
            checkpointer,
        )

    @property
    def workflow_ready(self) -> bool:
        return bool(
            self.workflow is not None
            and self.checkpoints is not None
            and self.checkpoints.is_started
        )

    async def close(self) -> None:
        """Release graph, outbound HTTP, and database resources during shutdown."""
        try:
            if self.checkpoints is not None:
                await self.checkpoints.close()
        finally:
            self.workflow = None
            self.checkpoints = None
            try:
                await self.http_client.aclose()
            finally:
                await self.database.close()
