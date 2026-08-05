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

The project is currently in the foundation and infrastructure phase.

## Database migrations

Operational PostgreSQL schemas are managed with Alembic:

```bash
uv run alembic upgrade head
uv run alembic check
```

See [`migrations/README`](migrations/README) for revision, rollback, SQLite testing, and
existing-database baseline guidance.

## Safety Boundary

Behavioral observations, user reports, statistical patterns, model hypotheses, and general neuroscience explanations are treated as separate evidence levels. Model-generated hypotheses must never be stored or presented as measured biological facts.
