"""TDD: end-to-end real-network crawl through /api/crawl/urls (uses local proxy).

This is an integration smoke. It launches actual playwright through the proxy
configured in `settings.proxy.http_proxy` (default 127.0.0.1:10808). Marked
slow + network so default `pytest` doesn't run it. Run via:

    .venv/Scripts/python.exe -m pytest tests/test_crawl_real_network.py -m network -v

Sites picked:
- https://example.com         — minimal HTML, no JS, sanity baseline
- https://httpbin.org/get     — JSON-in-html shape, no JS challenge
- https://www.python.org/     — real CDN site, validates full playwright flow

Contract pinned:
1. API returns 200 + task_id immediately
2. Within 60s, task transitions to completed (or failed with concrete errors)
3. ≥1 of 3 URLs must report status=success and content_length>0
4. Subtask/event/edge observability tables get populated (see test_crawl_task_inline)
5. proxy_used field is non-None (proves Bug #1 fix is wired live)
"""
from __future__ import annotations

import time
from typing import Dict, Any

import pytest
import requests


API_BASE = "http://127.0.0.1:8000"
TIMEOUT_S = 90
POLL_INTERVAL = 2.0

URLS = [
    "https://example.com",
    "https://httpbin.org/get",
    "https://www.python.org/",
]


pytestmark = pytest.mark.network


def _api_alive() -> bool:
    try:
        return requests.get(f"{API_BASE}/openapi.json", timeout=3).status_code == 200
    except Exception:
        return False


def _post_crawl(urls, use_proxy=True, auto_index=False) -> Dict[str, Any]:
    resp = requests.post(
        f"{API_BASE}/api/crawl/urls",
        json={
            "urls": urls,
            "use_proxy": use_proxy,
            "auto_index": auto_index,
            "max_concurrent": 3,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _get_task(task_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE}/api/tasks/{task_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _wait_terminal(task_id: str, timeout=TIMEOUT_S) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = _get_task(task_id)
        if last["status"] in {"completed", "failed", "cancelled", "timeout"}:
            return last
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Task {task_id} did not reach terminal state in {timeout}s, last status={last and last['status']}")


@pytest.fixture(autouse=True)
def _require_api():
    if not _api_alive():
        pytest.skip(f"uvicorn not reachable at {API_BASE}")


def test_real_crawl_through_proxy_end_to_end():
    """Submit 3 real URLs with use_proxy=True; expect ≥1 success + populated observability."""
    submission = _post_crawl(URLS, use_proxy=True, auto_index=False)
    assert submission["success"] is True
    assert submission["total_urls"] == len(URLS)
    task_id = submission["task_id"]

    terminal = _wait_terminal(task_id, timeout=TIMEOUT_S)
    assert terminal["status"] in {"completed", "failed"}, terminal

    records_resp = requests.get(
        f"{API_BASE}/api/crawl/records",
        params={"task_id": int(task_id), "page_size": 50},
        timeout=10,
    )
    records_resp.raise_for_status()
    records = records_resp.json()["records"]
    assert len(records) == len(URLS), records

    successes = [r for r in records if r["status"] == "success"]
    assert successes, (
        f"All {len(records)} URLs failed via proxy {URLS!r}: "
        f"{[(r['url'], r['status'], r.get('error_message')) for r in records]}"
    )

    for r in successes:
        assert r.get("content_length"), f"Success record missing content_length: {r}"
        assert r.get("title"), f"Success record missing title: {r}"

    subtasks_resp = requests.get(f"{API_BASE}/api/tasks/{task_id}/subtasks", timeout=10)
    subtasks = subtasks_resp.json()["subtasks"]
    assert len(subtasks) >= len(URLS) + 2, (
        f"Expected at least {len(URLS) + 2} subtasks (URL leafs + prepare/finalize), got {len(subtasks)}"
    )
    keys = {s["subtask_key"] for s in subtasks}
    assert "workflow.prepare" in keys
    assert "workflow.finalize" in keys

    events_resp = requests.get(f"{API_BASE}/api/tasks/{task_id}/events", timeout=10)
    events = events_resp.json()["events"]
    assert len(events) >= 5, f"Event stream too sparse: {len(events)}"

    result = terminal.get("result") or {}
    proxy_used_any = False
    for bucket in ("success_results", "failed_results", "blocked_results"):
        for item in result.get(bucket, []):
            if item.get("proxy_used"):
                proxy_used_any = True
                break
        if proxy_used_any:
            break
    assert proxy_used_any, (
        f"Expected at least one result.proxy_used to be set when use_proxy=True. "
        f"Result keys={list(result.keys())}"
    )


def test_real_crawl_workflow_endpoint_returns_graph():
    """After a real crawl, /workflow must return nodes + edges grounded in URLs."""
    submission = _post_crawl(URLS[:1], use_proxy=True, auto_index=False)
    task_id = submission["task_id"]
    _wait_terminal(task_id, timeout=TIMEOUT_S)

    wf_resp = requests.get(f"{API_BASE}/api/tasks/{task_id}/workflow", timeout=10)
    assert wf_resp.status_code == 200, wf_resp.text
    wf = wf_resp.json()
    assert wf["task_id"] == int(task_id)
    assert wf["task_type"] == "crawl_urls"
    assert isinstance(wf.get("nodes"), list) and wf["nodes"]
    assert isinstance(wf.get("edges"), list) and wf["edges"]
    node_keys = {n.get("key") or n.get("subtask_key") for n in wf["nodes"]}
    assert "workflow.prepare" in node_keys
    assert "workflow.finalize" in node_keys
