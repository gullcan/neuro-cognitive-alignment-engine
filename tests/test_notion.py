from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from neuro_alignment.config import Settings
from neuro_alignment.integrations import NotionClient


@pytest.mark.asyncio
async def test_notion_pagination_preserves_daily_filter_and_sort_contract() -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if "start_cursor" not in body:
            payload = {
                "results": [_page("page-1", "İlk görev")],
                "has_more": True,
                "next_cursor": "cursor-2",
            }
        else:
            payload = {
                "results": [_page("page-2", "İkinci görev")],
                "has_more": False,
                "next_cursor": None,
            }
        return httpx.Response(status_code=200, request=request, json=payload)

    settings = Settings(
        _env_file=None,
        app_env="test",
        notion_api_token="secret-token",
        notion_data_source_id="source-id",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        tasks = await NotionClient(settings, http_client).fetch_daily_tasks(date(2026, 8, 24))

    assert [task.title for task in tasks] == ["İlk görev", "İkinci görev"]
    assert len(requests) == 2
    assert requests[1]["filter"] == requests[0]["filter"]
    assert requests[1]["sorts"] == requests[0]["sorts"]
    assert requests[1]["page_size"] == 100
    assert requests[1]["start_cursor"] == "cursor-2"


def _page(page_id: str, title: str) -> dict[str, object]:
    return {
        "id": page_id,
        "last_edited_time": "2026-08-24T08:00:00.000Z",
        "properties": {
            "Task": {"title": [{"plain_text": title}]},
            "Window": {"date": {"start": "2026-08-24T09:00:00+03:00", "end": None}},
            "Status": {"status": {"name": "Planned"}},
            "Commitment Tier": {"select": {"name": "Core"}},
            "Priority": {"select": {"name": "P1"}},
            "Definition of Done": {"rich_text": [{"plain_text": "Çıktı hazır"}]},
            "Minimum Action": {"rich_text": [{"plain_text": "Dosyayı aç"}]},
        },
    }
