# Free Render and Neon Deployment

The zero-cost portfolio topology consists of one Render Free Docker web service and one
Neon Free PostgreSQL project. Render supplies the public HTTPS endpoint required by
Telegram, while Neon keeps operational data and LangGraph checkpoints outside Render's
ephemeral filesystem.

No payment card is required for the selected free plans at the time this guide was
written. Always review both providers' current plan screens before creating resources.

## Accepted constraints

- Render suspends the web service after an idle period. The first Telegram update after
  suspension can take about a minute to wake the service and might be retried by Telegram.
- Neon suspends idle compute and wakes it on the next database connection.
- Render's pre-deploy command is not available to free web services. The single-instance
  container therefore uses a tested Python entrypoint that applies the idempotent Alembic
  upgrade before replacing itself with Uvicorn.
- Render's configurable shutdown delay is not available on the Free plan, so the service
  uses the platform's free-tier shutdown behavior.
- This topology has no uptime SLA, high-availability guarantee, private network, or
  always-on background worker.
- GitHub Actions wakes the service for daily planning and 15-minute cycles that first sync
  today's Notion plan and then monitor its approved tasks. Unchanged syncs remain silent;
  changed plans require a fresh Telegram approval.
  Scheduled delivery is therefore approximate and can be delayed by the workflow queue or
  a Render cold start; persistent idempotency prevents duplicate controls after retries.

These are deployment-tier constraints, not changes to the LangGraph architecture. A later
paid deployment can restore always-on execution and move migrations back to a dedicated
pre-deploy phase without changing the domain model.

## 1. Create the Neon database

1. Create a Neon account and keep the **Free** plan selected.
2. Create a project named `neuro-cognitive-alignment-engine`.
3. Select an AWS European region close to Render Frankfurt when offered.
4. In the project dashboard, click **Connect**.
5. Disable connection pooling and copy the direct PostgreSQL connection string. Direct
   connections are required because the container runs schema migrations at startup.
6. Treat this connection string as a password. Do not paste it into chat or commit it.

The application automatically converts Neon's standard `postgresql://` URL to SQLAlchemy's
async psycopg dialect. LangGraph reuses the same PostgreSQL secret for checkpoints, so the
Render form asks for the database URL only once.

## 2. Apply the Render Blueprint

1. Sign in to Render and open **New > Blueprint**.
2. Select `gullcan/neuro-cognitive-alignment-engine`.
3. Confirm that the web service plan is **Free** and that no Render database is listed.
4. Supply the four prompted values:
   - `DATABASE_URL`: the direct Neon connection string;
   - `TELEGRAM_BOT_TOKEN`: from the local `.env` file;
   - `TELEGRAM_WEBHOOK_SECRET`: from the local `.env` file;
   - `TELEGRAM_CHAT_ID`: from the local `.env` file.
5. Apply the Blueprint and wait for the deploy to finish.
6. Open the service URL ending in `/health/ready`. Continue only when it returns
   `"status": "ok"` and both checks are `"ok"`.

Never import the local `.env` file as a whole. Local SQLite and development values would
override the Blueprint's production-safe configuration.

## 3. Telegram activation boundary

Creating the service does not register the Telegram webhook. Register
`https://<service>.onrender.com/v1/webhooks/telegram` only after readiness succeeds. Use
the configured secret-token header, drop stale pending updates, and verify
`getWebhookInfo` before sending the first live update.

## Deferred integrations

Add Notion and OpenAI credentials later from Render's **Environment** page. Keeping them
outside the first deployment isolates failures to the database and Telegram boundaries.

After Notion is configured, enable both repository workflows. They reuse the existing
`RENDER_INTERNAL_API_KEY` secret: **Daily Notion plan** imports the day at 07:35 Istanbul
time and **Notion task monitor** checks the approved plan every 15 minutes. No additional
Render worker or paid cron service is required.
