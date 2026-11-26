"""Real HTTP gate for durable, exact match-paper jobs.

The target backend must use its real SQL task store, BGE-M3, Qdrant and
reranker.  This script does not patch imports, replace transports, or create a
scratch retrieval implementation.
"""
from __future__ import annotations

import argparse
import json
import time

import httpx


TERMINAL = {"completed", "failed", "cancelled", "timeout"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-query-chunks", type=int, default=10)
    parser.add_argument("--submit-deadline", type=float, default=2.0)
    parser.add_argument("--completion-timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    payload = {
        "input_type": "paper_id",
        "paper_id": args.paper_id,
        "top_k": args.top_k,
        "max_query_chunks": args.max_query_chunks,
    }
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        started = time.perf_counter()
        response = client.post("/api/match-doc/jobs", json=payload)
        submit_seconds = time.perf_counter() - started
        response.raise_for_status()
        receipt = response.json()
        task_id = int(receipt["task_id"])
        if submit_seconds > args.submit_deadline:
            raise SystemExit(
                f"submission took {submit_seconds:.3f}s, above {args.submit_deadline:.3f}s"
            )

        deadline = time.monotonic() + args.completion_timeout
        history: list[dict] = []
        last_signature = None
        task = None
        while time.monotonic() < deadline:
            # Owner-scoped job route, not generic /api/tasks: that is the surface a
            # real client has, and polling the generic one would not exercise the
            # ownership check at all.
            status_response = client.get(f"/api/match-doc/jobs/{task_id}")
            status_response.raise_for_status()
            task = status_response.json()
            signature = (task.get("status"), task.get("progress"), task.get("total"))
            if signature != last_signature:
                history.append({
                    "status": signature[0],
                    "progress": signature[1],
                    "total": signature[2],
                })
                last_signature = signature
            if task.get("status") in TERMINAL:
                break
            time.sleep(args.poll_interval)
        else:
            raise SystemExit(f"task {task_id} did not finish before timeout")

    report = {
        "task_id": task_id,
        "submit_seconds": submit_seconds,
        "history": history,
        "status": task.get("status") if task else None,
        "error": task.get("error") if task else None,
        "result_metrics": ((task or {}).get("result") or {}).get("metrics"),
        "paper_ids": [
            row.get("paper_id")
            for row in (((task or {}).get("result") or {}).get("papers") or [])
        ],
    }
    print(json.dumps(report, ensure_ascii=False))
    if task is None or task.get("status") != "completed":
        raise SystemExit(f"match-paper task did not complete successfully: {report}")
    result = task.get("result") or {}
    if result.get("input_type") != "paper_id":
        raise SystemExit("completed task is missing the exact match response")
    if not isinstance(result.get("papers"), list):
        raise SystemExit("completed task result has no paper list")
    if not history or history[0].get("status") not in {"pending", "running"}:
        raise SystemExit("task did not expose a pending/running state")


if __name__ == "__main__":
    main()
