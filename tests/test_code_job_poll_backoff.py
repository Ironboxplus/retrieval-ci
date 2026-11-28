"""Both MCP clients back off while waiting for a running code index job.

Observed 2026-11-09: a client held a running index job and queried /api/code/jobs/{id} about
twice a second for the whole run. The wait loop now starts at 0.5 s and doubles up to 5 s.
The two implementations (single-file plugin and the repo's modular server) must agree.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

import mcp_server.server as server

PLUGIN = Path(__file__).resolve().parents[1] / "mcp_plugin" / "retrieval_mcp.py"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("retrieval_mcp_plugin_poll_backoff", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _wire_job_sequence(mod, statuses):
    """Make the module's job lookup return `statuses` in order; returns the call log."""
    calls: list[str] = []
    remaining = list(statuses)

    def next_job():
        status = remaining.pop(0)
        calls.append(status)
        return {"job_id": "backend-job", "status": status, "progress": 0.1}

    if mod is server:
        class Client:
            async def code_index_job(self, _job_id, **_kwargs):
                return next_job()

        mod._client = Client()
    else:
        async def get_job(path, params=None):
            assert path == "/api/code/jobs/backend-job"
            return next_job()

        mod._get = get_job
    return calls


@pytest.fixture(params=["plugin", "server"])
def client_module(request, monkeypatch):
    if request.param == "server":
        monkeypatch.setattr(server, "_client", server._client)  # restored after the test swaps it
        return server
    return _load_plugin()


def _record_sleeps(monkeypatch):
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return sleeps


def test_running_job_is_polled_with_exponential_backoff_capped_at_five_seconds(client_module, monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    calls = _wire_job_sequence(client_module, ["running"] * 8 + ["completed"])

    job = asyncio.run(client_module._wait_code_job("backend-job"))

    assert job["status"] == "completed"
    assert len(calls) == 9
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0, 5.0]


def test_first_poll_after_submission_is_still_prompt(client_module, monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    _wire_job_sequence(client_module, ["queued", "completed"])

    asyncio.run(client_module._wait_code_job("backend-job"))

    assert sleeps == [0.5]


def test_poll_interval_constants_are_shared_by_both_clients():
    plugin = _load_plugin()
    assert plugin._CODE_JOB_POLL_SECONDS == server._CODE_JOB_POLL_SECONDS == 0.5
    assert plugin._CODE_JOB_POLL_MAX_SECONDS == server._CODE_JOB_POLL_MAX_SECONDS == 5.0
