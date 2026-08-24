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

The operational database, FastAPI runtime, stateful LangGraph workflow, authenticated
Telegram ingress, scheduler trigger, and leased outbox delivery are implemented.

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

Runtime entry points are deliberately separated by trust boundary:

- `POST /v1/webhooks/telegram` requires Telegram's
  `X-Telegram-Bot-Api-Secret-Token` header and validates the configured chat.
- `POST /v1/internal/scheduler/daily-plan` requires `X-Internal-Api-Key` and creates
  an idempotent daily planning event.
- `POST /v1/internal/outbox/deliver` requires `X-Internal-Api-Key` and delivers one
  leased Telegram batch.

The repository also includes a zero-cost GitHub Actions scheduler. It triggers the daily
Notion planning endpoint at 07:35 in `Europe/Istanbul`, supports manual runs, and uses a
deterministic daily request ID so repeated invocations remain idempotent. Configure the
repository secret `RENDER_INTERNAL_API_KEY` with the same value as Render's
`INTERNAL_API_KEY`; the secret is never stored in the workflow file.

When Telegram delivery is enabled, workflow requests attempt an outbox delivery after
processing. Failed records are retried up to `OUTBOX_MAX_ATTEMPTS`, and then moved to the
`dead` state. PostgreSQL workers use row locking with `SKIP LOCKED`; a delivery lease also
recovers records left in `sending` by a stopped worker. Telegram does not expose a message
idempotency key, so delivery is intentionally at-least-once across a crash exactly between
the provider accepting a message and the local sent marker being committed.

## Production deployment

The repository includes a zero-cost portfolio Blueprint using a Render Free Docker web
service and Neon Free PostgreSQL. The database persists operational events and durable
LangGraph checkpoints outside Render's ephemeral filesystem. Because free Render services
do not support pre-deploy commands, the single-instance container applies idempotent
Alembic migrations before starting the API.

See [`docs/deployment/render.md`](docs/deployment/render.md) for the dashboard flow,
free-tier constraints, secret-handling boundary, and Telegram activation gate.

## Safety Boundary

Behavioral observations, user reports, statistical patterns, model hypotheses, and general neuroscience explanations are treated as separate evidence levels. Model-generated hypotheses must never be stored or presented as measured biological facts.
