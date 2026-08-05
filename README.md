# Neuro-Cognitive Alignment Engine

A stateful, evidence-aware intent-action alignment system built with LangGraph, FastAPI, PostgreSQL, Notion, Telegram, OpenAI, and Pinecone.

## Project Vision

This project is designed to analyze the gap between planned commitments and observed actions. It retrieves relevant behavioral history, identifies recurring patterns, and produces direct but evidence-bounded behavioral feedback.

The system is not a generic reminder bot and does not provide medical diagnosis or claim to measure neurological activity.

## Core Capabilities

- Read daily commitments dynamically from Notion
- Receive check-ins and task actions through Telegram
- Orchestrate stateful workflows with LangGraph
- Persist behavioral events and graph checkpoints
- Retrieve semantically similar behavioral episodes
- Generate evidence-grounded neuro-behavioral feedback
- Produce weekly word-action consistency reviews
- Support human-in-the-loop approvals

## Architecture

- Python 3.12
- FastAPI
- LangGraph
- PostgreSQL
- Pinecone
- Notion API
- Telegram Bot API
- OpenAI Responses API

## Development Status

The operational database, FastAPI runtime, and first stateful LangGraph workflow are
implemented. External webhook endpoints and outbox delivery are the next runtime layer.

## LangGraph Runtime

The graph uses deterministic routing for control flow and reserves the language model for
planning and evidence-bounded feedback generation. Its current branches are:

```text
claim inbound event
├── daily plan -> Notion -> Planner Agent -> persist plan -> Telegram outbox
├── plan decision -> approve/reject persisted plan -> Telegram outbox
├── task behavior -> record event -> retrieve evidence -> Neuro-Behavioral Agent
│                    -> Safety Critic -> Telegram outbox
└── check-in -> record self-report
```

Every branch finishes by marking the inbound event complete. Repeated source events are
stopped at the claim node, making downstream writes idempotent. The Safety Critic permits
one model revision and fails closed if unsupported biological or clinical claims remain.

LangGraph checkpoints and behavioral memory serve different purposes:

- Checkpoints persist graph execution state and recovery history per `thread_id`.
- Operational events persist observed behavior used to construct evidence.
- Semantic long-term memory will later retrieve similar episodes across threads.

Checkpoint backends are selected with `CHECKPOINT_BACKEND`: `memory` for tests, `sqlite`
for local experiments, and `postgres` for the durable runtime. PostgreSQL checkpoint tables
are managed by LangGraph and intentionally remain outside Alembic's operational schema.

## Database migrations

Operational PostgreSQL schemas are managed with Alembic:

```bash
uv run alembic upgrade head
uv run alembic check
```

See [`migrations/README`](migrations/README) for revision, rollback, SQLite testing, and
existing-database baseline guidance.

## Local API

Start PostgreSQL, apply migrations, and run the API:

```bash
docker compose up -d postgres
uv run alembic upgrade head
uv run neuro-alignment
```

The interactive OpenAPI interface is available at `http://localhost:8000/docs`.
Deployment probes use separate endpoints:

- `GET /health/live` checks that the API process is running.
- `GET /health/ready` verifies both graph initialization and database connectivity.

## Safety Boundary

Behavioral observations, user reports, statistical patterns, model hypotheses, and general neuroscience explanations are treated as separate evidence levels. Model-generated hypotheses must never be stored or presented as measured biological facts.
