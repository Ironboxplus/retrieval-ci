#!/usr/bin/env python
"""Mirror the external ICML 2026 corpus run into the Web /tasks UI.

The ICML 2026 run currently has pieces launched outside the normal observable
pipeline: CDP download on Arc, tar sync to the 4090 box, marker tmux jobs, then
SQL/Qdrant ingest. This watcher is deliberately small: it observes durable
artifacts on the 4090 host and writes progress into the existing TaskStore
tables so the React Tasks page can render the run as a pipeline-shaped task.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.pipeline.store import TaskStore
from core.db.database import get_db_session
from core.db.models import Chunk, Paper, Task, TaskSubtask, TaskSubtaskStatus


TASK_TYPE = "icml2026_external_pipeline"
WATCHER_STATE_KEY = "icml2026_watcher"

STAGES = (
    ("download_sync", "Sync PDFs to 4090"),
    ("extract_md", "Convert PDFs to Markdown"),
    ("store_sql", "Store papers in SQL"),
    ("chunk_sql", "Chunk/index SQL rows"),
    ("qdrant_upsert", "Qdrant upsert"),
)


@dataclass(frozen=True)
class StageView:
    key: str
    name: str
    status: str
    done: int
    total: int
    message: str


def count_files(root: Path, suffix: str) -> int:
    if not root.exists():
        return 0
    wanted = suffix.lower()
    count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not name.endswith(wanted):
            continue
        if name.endswith(".write_in_progress.pdf"):
            continue
        count += 1
    return count


def proc_contains(*needles: str) -> bool:
    proc = Path("/proc")
    if not proc.exists():
        return False
    lowered = [n.lower() for n in needles if n]
    for cmdline in proc.glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore").lower()
        except OSError:
            continue
        if all(n in text for n in lowered):
            return True
    return False


def db_counts() -> dict[str, int]:
    with get_db_session() as session:
        papers = (
            session.query(func.count(Paper.paper_id))
            .filter(Paper.venue == "ICML", Paper.year == 2026)
            .scalar()
            or 0
        )
        chunked_papers = (
            session.query(func.count(func.distinct(Paper.paper_id)))
            .join(Chunk, Paper.doc_id == Chunk.doc_id)
            .filter(Paper.venue == "ICML", Paper.year == 2026)
            .scalar()
            or 0
        )
        chunks = (
            session.query(func.count(Chunk.id))
            .join(Paper, Paper.doc_id == Chunk.doc_id)
            .filter(Paper.venue == "ICML", Paper.year == 2026)
            .scalar()
            or 0
        )
    return {
        "sql_papers": int(papers),
        "chunked_papers": int(chunked_papers),
        "sql_chunks": int(chunks),
    }


def qdrant_count(args: argparse.Namespace) -> int | None:
    if not args.qdrant_url:
        return None
    body = {
        "filter": {
            "must": [
                {"key": "venue", "match": {"value": "ICML"}},
                {"key": "year", "match": {"value": 2026}},
                {"key": "source_type", "match": {"value": "paper"}},
            ]
        },
        "exact": True,
    }
    url = f"{args.qdrant_url.rstrip('/')}/collections/{args.qdrant_collection}/points/count"
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=args.qdrant_timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return int(payload.get("result", {}).get("count", 0))
    except Exception:
        return None


def status_by_count(done: int, total: int, *, started: bool = False) -> str:
    if total > 0 and done >= total:
        return "completed"
    if started or done > 0:
        return "running"
    return "pending"


def collect(args: argparse.Namespace) -> tuple[dict[str, Any], list[StageView]]:
    pdf_dir = Path(args.pdf_dir)
    md_dir = Path(args.md_dir)
    total = max(int(args.total), 1)
    record_total = max(int(args.record_total), total)

    pdfs = count_files(pdf_dir, ".pdf")
    mds = count_files(md_dir, ".md")
    counts = db_counts()
    qdrant_points = qdrant_count(args)

    marker_running = proc_contains("marker") or proc_contains("run_marker_kss")
    sync_running = proc_contains("tar", "icml")
    sync_done = Path(args.sync_done_path).exists()

    q_total = max(counts["sql_chunks"], qdrant_points or 0, 1)
    q_done = qdrant_points if qdrant_points is not None else 0
    q_message = (
        f"{q_done}/{q_total} points; sql_chunks={counts['sql_chunks']}"
        if qdrant_points is not None
        else f"Qdrant count unavailable; sql_chunks={counts['sql_chunks']}"
    )

    stages = [
        StageView(
            "download_sync",
            "Sync PDFs to 4090",
            status_by_count(pdfs, total, started=sync_running or sync_done),
            min(pdfs, total),
            total,
            f"4090_synced_pdfs={pdfs}/{total}; manifest_records={record_total}; sync_running={sync_running}; sync_done={sync_done}",
        ),
        StageView(
            "extract_md",
            "Convert PDFs to Markdown",
            status_by_count(mds, total, started=marker_running),
            min(mds, total),
            total,
            f"markdown={mds}/{total}; marker_running={marker_running}; remote_pdfs={pdfs}",
        ),
        StageView(
            "store_sql",
            "Store papers in SQL",
            status_by_count(counts["sql_papers"], total),
            min(counts["sql_papers"], total),
            total,
            f"sql_papers={counts['sql_papers']}/{total}",
        ),
        StageView(
            "chunk_sql",
            "Chunk/index SQL rows",
            status_by_count(counts["chunked_papers"], total, started=counts["sql_papers"] > 0),
            min(counts["chunked_papers"], total),
            total,
            f"chunked_papers={counts['chunked_papers']}/{total}; sql_chunks={counts['sql_chunks']}",
        ),
        StageView(
            "qdrant_upsert",
            "Qdrant upsert",
            status_by_count(q_done, q_total, started=counts["sql_chunks"] > 0),
            min(q_done, q_total),
            q_total,
            q_message,
        ),
    ]
    snapshot = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "manifest_records": record_total,
        "pdfs": pdfs,
        "markdown": mds,
        "sync_running": sync_running,
        "sync_done": sync_done,
        "marker_running": marker_running,
        "qdrant_points": qdrant_points,
        **counts,
        "stages": {s.key: {"status": s.status, "done": s.done, "total": s.total} for s in stages},
    }
    return snapshot, stages


def task_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "venue": "ICML",
        "year": 2026,
        "source": WATCHER_STATE_KEY,
        "host": socket.gethostname(),
        "total_papers": args.total,
        "manifest_records": args.record_total,
        "pdf_dir": str(Path(args.pdf_dir)),
        "md_dir": str(Path(args.md_dir)),
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.qdrant_collection,
    }


def select_canonical_task_id(rows: list[Task]) -> int | None:
    """Return the oldest task for this durable corpus run.

    Task status is intentionally ignored: a completed task remains the stable
    UI identity for the run and may need its counters reconciled later.
    """
    matching = [
        int(row.id)
        for row in rows
        if (row.params_json or {}).get("source") == WATCHER_STATE_KEY
    ]
    return min(matching) if matching else None


def find_existing_task() -> int | None:
    with get_db_session() as session:
        rows = (
            session.query(Task)
            .filter(Task.task_type == TASK_TYPE)
            .order_by(Task.id.asc())
            .all()
        )
        return select_canonical_task_id(rows)


def ensure_task(args: argparse.Namespace, store: TaskStore) -> int:
    task_id = find_existing_task()
    created = task_id is None
    if task_id is None:
        task_id = store.create_task(task_type=TASK_TYPE, params=task_params(args), total=100)
        store.emit_event(task_id, None, "ICML 2026 external pipeline watcher attached", event_type="watcher.start")

    for order, (key, name) in enumerate(STAGES):
        store.ensure_subtask(
            task_id,
            key,
            subtask_type="pipeline_stage",
            name=name,
            order=order,
            total=args.total,
        )
    if created:
        store.set_task_status(task_id, "running")
    return task_id


def set_subtask(task_id: int, stage: StageView) -> None:
    with get_db_session() as session:
        sub = (
            session.query(TaskSubtask)
            .filter(TaskSubtask.task_id == task_id, TaskSubtask.subtask_key == stage.key)
            .first()
        )
        if sub is None:
            return
        now = datetime.now()
        previous_status = sub.status.value if sub.status else None
        sub.name = stage.name
        sub.status = TaskSubtaskStatus(stage.status)
        sub.progress = int(stage.done)
        sub.total = int(stage.total)
        sub.message = stage.message
        sub.stage = stage.key
        if stage.status == "running" and sub.started_at is None:
            sub.started_at = now
        if stage.status == "completed" and sub.completed_at is None:
            sub.completed_at = now
        if previous_status != stage.status and stage.status != "completed":
            sub.completed_at = None


def previous_snapshot(task_id: int) -> dict[str, Any]:
    with get_db_session() as session:
        task = session.get(Task, task_id)
        if task is None:
            return {}
        return dict((task.params_json or {}).get("state", {}).get(WATCHER_STATE_KEY) or {})


def save_snapshot(task_id: int, snapshot: dict[str, Any]) -> None:
    with get_db_session() as session:
        task = session.get(Task, task_id)
        if task is None:
            return
        params = dict(task.params_json or {})
        state = dict(params.get("state") or {})
        state[WATCHER_STATE_KEY] = snapshot
        params["state"] = state
        task.params_json = params
        task.last_heartbeat = datetime.now()


def maybe_emit_events(
    store: TaskStore,
    task_id: int,
    prev: dict[str, Any],
    stages: list[StageView],
    event_step: int,
) -> None:
    previous_stages = prev.get("stages") or {}
    for stage in stages:
        old = previous_stages.get(stage.key) or {}
        old_done = int(old.get("done") or 0)
        old_status = old.get("status")
        moved = abs(stage.done - old_done) >= event_step
        status_changed = old_status != stage.status
        completed = stage.status == "completed" and old_status != "completed"
        if status_changed or moved or completed:
            store.emit_event(
                task_id,
                stage.key,
                f"{stage.name}: {stage.done}/{stage.total}; {stage.message}",
                event_type="stage.progress",
                stage=stage.key,
            )


def all_stages_completed(stages: list[StageView]) -> bool:
    return bool(stages) and all(stage.status == "completed" for stage in stages)


def update_task(
    task_id: int,
    store: TaskStore,
    snapshot: dict[str, Any],
    stages: list[StageView],
    event_step: int,
) -> bool:
    prev = previous_snapshot(task_id)
    completed = 0
    pct_sum = 0.0
    for stage in stages:
        set_subtask(task_id, stage)
        if stage.status == "completed":
            completed += 1
        if stage.total > 0:
            pct_sum += min(stage.done / stage.total, 1.0)
    parent_progress = int(round((pct_sum / max(len(stages), 1)) * 100))
    store.set_task_progress(task_id, parent_progress, 100)
    maybe_emit_events(store, task_id, prev, stages, event_step)
    save_snapshot(task_id, snapshot)
    finished = completed == len(STAGES)
    if finished:
        store.set_task_status(task_id, "completed")
    else:
        store.set_task_status(task_id, "running")
    return finished


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=6492, help="downloadable/main PDF total used for pipeline progress")
    parser.add_argument("--record-total", type=int, default=6614, help="raw ICML 2026 manifest record count for reference")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--event-step", type=int, default=25)
    parser.add_argument(
        "--pdf-dir",
        default="data/papers/ccf_a_main_conference_corpus/pdf/icml/2026",
    )
    parser.add_argument(
        "--md-dir",
        default="data/papers/ccf_a_main_conference_corpus/markdown/ICML/2026",
    )
    parser.add_argument("--sync-done-path", default="/tmp/icml2026_pdf_sync_done")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant-collection", default="papers")
    parser.add_argument("--qdrant-timeout-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = TaskStore()
    while True:
        snapshot, stages = collect(args)
        if args.dry_run:
            print(json.dumps({"snapshot": snapshot, "stages": [s.__dict__ for s in stages]}, indent=2))
        else:
            task_id = ensure_task(args, store)
            completed = update_task(task_id, store, snapshot, stages, args.event_step)
            print(f"task_id={task_id} {snapshot['ts']} pdf={snapshot['pdfs']} md={snapshot['markdown']} sql={snapshot['sql_papers']} chunks={snapshot['sql_chunks']} qdrant={snapshot['qdrant_points']}", flush=True)

        if args.once or (not args.dry_run and completed):
            return 0
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
