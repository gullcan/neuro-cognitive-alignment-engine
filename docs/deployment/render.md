# Render Production Deployment

The first production topology consists of one always-on Docker web service and one
managed PostgreSQL instance in Render's Frankfurt region. Render terminates TLS at the
edge and supplies the public HTTPS endpoint required by Telegram webhooks.

## Provisioned resources

`render.yaml` provisions:

- a Starter web service built from the repository's pinned Docker image;
- a Basic 256 MB PostgreSQL 17 database with public database access disabled;
- a private database connection shared by the operational SQLAlchemy store and the
  LangGraph PostgreSQL checkpointer;
- an Alembic pre-deploy command that must succeed before a new release starts;
- a readiness health check that verifies the database and compiled graph;
- generated internal API credentials and dashboard-provided Telegram secrets.

These are paid, always-on resources. The selected plans avoid webhook cold starts and the
expiration behavior of a free PostgreSQL database. Confirm the current price shown by
Render before applying the Blueprint.

## Initial dashboard flow

1. Sign in to Render with the GitHub account that can access
   `gullcan/neuro-cognitive-alignment-engine`.
2. Open **New > Blueprint** and select that repository.
3. Render detects `render.yaml`. Review the Frankfurt region and the two selected plans.
4. Supply the three prompted values from the local `.env` file:
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, and `TELEGRAM_CHAT_ID`.
5. Apply the Blueprint and wait until the database and web service are both available.
6. Open the service URL ending in `/health/ready`. Continue only when it returns
   `"status": "ok"` with both database and workflow checks set to `"ok"`.

Never paste `.env` as a whole into Render. It contains local-only settings that would
override the production-safe values declared by the Blueprint.

## Telegram activation boundary

Creating the web service does not register a Telegram webhook. Registration is a separate
controlled step after the readiness check succeeds. Register the HTTPS URL
`/v1/webhooks/telegram` with Telegram's secret-token header and drop any stale pending
updates. Verify Telegram's `getWebhookInfo` response before sending a live test update.

## Later integrations

Notion and OpenAI credentials are intentionally not part of the first deployment gate.
Add them from the Render service's **Environment** page only after the Telegram ingress,
database durability, and observability checks pass. This keeps deployment failures
isolated to one integration boundary at a time.
