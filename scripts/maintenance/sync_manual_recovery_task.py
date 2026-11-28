"""Keep a manually resumed vector-ingest task visible in the WebUI.

The normal orchestrator writes task heartbeats itself. This small bridge is only
for a recovery run that was launched outside the orchestrator after its retry
budget was exhausted. It reads the durable ingest checkpoint and writes the
same task's heartbeat/progress event, without changing vector data.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--vectors-dir", type=Path, required=True)
    p.add_argument("--shard-end", type=int, required=True)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--db", type=Path, default=Path("data/db/retrieval.db"))
    args = p.parse_args()

    done_path = args.vectors_dir / "ingest_done.txt"
    progress_path = args.vectors_dir / "ingest_done.progress.json"
    db = sqlite3.connect(args.db)
    previous = None
    try:
        while True:
            done = set(done_path.read_text(encoding="utf-8").splitlines()) if done_path.exists() else set()
            progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
            checkpoint = "done=" + str(len(done))
            if progress:
                name, rows = sorted(progress.items())[-1]
                checkpoint = f"{name}:{rows}"
            now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
            params_row = db.execute("select params_json from tasks where id=?", (args.task_id,)).fetchone()
            if params_row is None:
                return
            params = json.loads(params_row[0] or "{}")
            state = params.setdefault("state", {})
            state.update({
                "manual_recovery_run_id": args.run_id,
                "manual_recovery_stage": "ingest",
                "manual_recovery_checkpoint": checkpoint,
            })
            params["state"] = state
            db.execute(
                "update tasks set status='running', progress=4, total=6, error_message=NULL, "
                "worker_id='manual-recovery-ingest', last_heartbeat=?, params_json=? "
                "where id=? and status='running'",
                (now, json.dumps(params), args.task_id),
            )
            if checkpoint != previous:
                db.execute(
                    "insert into task_events(task_id,subtask_id,event_type,level,stage,message,payload_json,created_at) "
                    "values(?,?,?,?,?,?,?,?)",
                    (args.task_id, None, "stage.log", "info", "ingest",
                     f"manual recovery checkpoint {checkpoint}",
                     json.dumps({"run_id": args.run_id}), now),
                )
                db.commit()
                previous = checkpoint
            time.sleep(max(args.interval, 1.0))
    finally:
        db.close()


if __name__ == "__main__":
    main()
