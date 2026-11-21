"""Real HTTP cancellation gate for queued exact match-paper work."""
from __future__ import annotations

import argparse
import json
import time

import httpx


def _submit(client: httpx.Client, paper_id: str, chunks: int) -> int:
    response = client.post("/api/match-doc/jobs", json={
        "input_type": "paper_id",
        "paper_id": paper_id,
        "top_k": 3,
        "max_query_chunks": chunks,
    })
    response.raise_for_status()
    return int(response.json()["task_id"])


def _wait_terminal(client: httpx.Client, task_id: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/match-doc/jobs/{task_id}")
        response.raise_for_status()
        task = response.json()
        if task.get("status") in {"completed", "failed", "cancelled", "timeout"}:
            return task
        time.sleep(0.1)
    raise RuntimeError(f"task {task_id} did not become terminal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=30.0) as client:
        running_id = _submit(client, args.paper_id, args.chunks)
        cancelled_id = _submit(client, args.paper_id, args.chunks)
        cancel_response = client.post(f"/api/match-doc/jobs/{cancelled_id}/cancel")
        cancel_response.raise_for_status()
        cancelled = _wait_terminal(client, cancelled_id, args.timeout)
        running = _wait_terminal(client, running_id, args.timeout)

    report = {
        "running_task_id": running_id,
        "running_status": running.get("status"),
        "cancelled_task_id": cancelled_id,
        "cancelled_status": cancelled.get("status"),
        "cancelled_progress": cancelled.get("progress"),
        "cancelled_result": cancelled.get("result"),
    }
    print(json.dumps(report, ensure_ascii=False))
    if running.get("status") != "completed":
        raise SystemExit("the lane owner did not complete")
    if cancelled.get("status") != "cancelled":
        raise SystemExit("the queued task was not cancelled")
    if cancelled.get("result") is not None:
        raise SystemExit("cancelled task published a partial result")


if __name__ == "__main__":
    main()
