#!/usr/bin/env python
"""
Monitor the formal CCF-A main-conference corpus pipeline until it converges or stalls.

The monitor is intentionally read-only. It samples:
1. Manifest-level download / extract / index counters
2. Pipeline state and latest stage checkpoints
3. Relevant long-running wrapper / rebuild processes
4. Lightweight database counts for already indexed artifacts

Snapshots are written to both a JSONL history log and a current-state JSON file.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil
except Exception:  # pragma: no cover - monitoring still works without process inspection
    psutil = None


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "papers" / "ccf_a_main_conference_corpus" / "manifests" / "paper_manifest.json"
DEFAULT_PIPELINE_STATE_PATH = REPO_ROOT / ".omx" / "state" / "ccf_a_main_corpus_pipeline_state.json"
DEFAULT_SNAPSHOT_LOG_PATH = REPO_ROOT / ".omx" / "logs" / "ccf_a_main_corpus_pipeline_monitor.jsonl"
DEFAULT_CURRENT_STATE_PATH = REPO_ROOT / ".omx" / "state" / "ccf_a_main_corpus_pipeline_monitor_state.json"
DEFAULT_EXTRACT_LOG_PATH = REPO_ROOT / ".omx" / "logs" / "ccf_a_main_corpus_extract.err.log"
DEFAULT_INDEX_LOG_PATH = REPO_ROOT / ".omx" / "logs" / "ccf_a_main_corpus_index.err.log"
DEFAULT_TAIL_REPAIR_LOG_PATH = REPO_ROOT / ".omx" / "logs" / "ccf_a_main_corpus_tail_repair.log"
DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "db" / "retrieval.db"

CHECKPOINT_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+\|\s+INFO\s+\|\s+Saving manifest checkpoint for "
    r"(?P<stage>\w+) \((?P<processed>\d+)/(?P<total>\d+)\)"
)


@dataclass(slots=True)
class ManifestStats:
    total: int
    downloaded: int
    download_pending: int
    download_fail: int
    extracted_ok: int
    extract_fail: int
    extract_pending: int
    indexed_ok: int
    index_fail: int
    index_pending: int


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor the formal CCF-A main-conference pipeline until completion or stall.",
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--pipeline-state-path", type=Path, default=DEFAULT_PIPELINE_STATE_PATH)
    parser.add_argument("--snapshot-log-path", type=Path, default=DEFAULT_SNAPSHOT_LOG_PATH)
    parser.add_argument("--current-state-path", type=Path, default=DEFAULT_CURRENT_STATE_PATH)
    parser.add_argument("--extract-log-path", type=Path, default=DEFAULT_EXTRACT_LOG_PATH)
    parser.add_argument("--index-log-path", type=Path, default=DEFAULT_INDEX_LOG_PATH)
    parser.add_argument("--tail-repair-log-path", type=Path, default=DEFAULT_TAIL_REPAIR_LOG_PATH)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--sample-interval-seconds", type=int, default=300)
    parser.add_argument("--max-idle-samples", type=int, default=12)
    parser.add_argument(
        "--single-snapshot",
        action="store_true",
        help="Write one snapshot and exit immediately.",
    )
    return parser


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def compute_manifest_stats(records: Iterable[dict[str, Any]]) -> ManifestStats:
    total = 0
    downloaded = 0
    download_pending = 0
    download_fail = 0
    extracted_ok = 0
    extract_fail = 0
    indexed_ok = 0
    index_fail = 0

    for record in records:
        total += 1
        has_pdf = bool(record.get("pdf_path"))
        has_markdown = bool(record.get("markdown_path"))
        has_index = bool(record.get("indexed_doc_id"))
        has_download_error = bool(record.get("download_error"))
        has_extract_error = bool(record.get("extract_error"))
        has_index_error = bool(record.get("index_error"))

        if has_pdf and not has_download_error:
            downloaded += 1
        elif has_download_error:
            download_fail += 1
        else:
            download_pending += 1

        if has_markdown and not has_extract_error:
            extracted_ok += 1
        elif has_extract_error:
            extract_fail += 1

        if has_index and not has_index_error:
            indexed_ok += 1
        elif has_index_error:
            index_fail += 1

    extract_pending = max(downloaded - extracted_ok - extract_fail, 0)
    index_pending = max(extracted_ok - indexed_ok - index_fail, 0)
    return ManifestStats(
        total=total,
        downloaded=downloaded,
        download_pending=download_pending,
        download_fail=download_fail,
        extracted_ok=extracted_ok,
        extract_fail=extract_fail,
        extract_pending=extract_pending,
        indexed_ok=indexed_ok,
        index_fail=index_fail,
        index_pending=index_pending,
    )


def query_database_counts(database_path: Path) -> dict[str, int]:
    if not database_path.exists():
        return {}

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        counts: dict[str, int] = {}
        for table_name in ("documents", "chunks", "papers"):
            try:
                counts[table_name] = int(cursor.execute(f"select count(*) from {table_name}").fetchone()[0])
            except sqlite3.DatabaseError:
                counts[table_name] = -1
        return counts


def list_relevant_processes() -> dict[str, list[dict[str, Any]]]:
    process_groups = {
        "extract": [],
        "index": [],
        "download": [],
        "repair_urls": [],
        "extract_and_index_wrapper": [],
        "tail_repair_wrapper": [],
        "monitor": [],
    }
    if psutil is None:
        return process_groups

    current_pid = psutil.Process().pid
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            cmdline_parts = process.info.get("cmdline") or []
            command_line = " ".join(cmdline_parts)
        except (psutil.Error, OSError):
            continue

        if not command_line:
            continue

        if process.info["pid"] == current_pid and "monitor_ccf_a_main_corpus_pipeline.py" in command_line:
            process_groups["monitor"].append(
                {
                    "pid": process.info["pid"],
                    "parent_pid": process.info.get("ppid"),
                    "command_line": command_line,
                }
            )
            continue

        if "rebuild_ccf_a_main_corpus.py" in command_line:
            stage_name = None
            for index, part in enumerate(cmdline_parts):
                if part == "--stage" and index + 1 < len(cmdline_parts):
                    stage_name = cmdline_parts[index + 1]
                    break
            if stage_name in process_groups:
                process_groups[stage_name].append(
                    {
                        "pid": process.info["pid"],
                        "parent_pid": process.info.get("ppid"),
                        "command_line": command_line,
                    }
                )
            continue

        if "run_ccf_a_main_corpus_extract_and_index.ps1" in command_line:
            process_groups["extract_and_index_wrapper"].append(
                {
                    "pid": process.info["pid"],
                    "parent_pid": process.info.get("ppid"),
                    "command_line": command_line,
                }
            )
            continue

        if "run_ccf_a_main_corpus_tail_repair.ps1" in command_line:
            process_groups["tail_repair_wrapper"].append(
                {
                    "pid": process.info["pid"],
                    "parent_pid": process.info.get("ppid"),
                    "command_line": command_line,
                }
            )

    return process_groups


def read_pipeline_state(path: Path) -> dict[str, Any]:
    state = load_json(path, default={})
    return state if isinstance(state, dict) else {}


def read_last_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    recent_lines = deque(maxlen=300)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            recent_lines.append(line.rstrip("\n"))

    for line in reversed(recent_lines):
        match = CHECKPOINT_PATTERN.search(line)
        if match:
            return {
                "line": line,
                "timestamp": match.group("timestamp"),
                "stage": match.group("stage"),
                "processed": int(match.group("processed")),
                "total": int(match.group("total")),
            }
    return None


def read_last_nonempty_line(path: Path) -> str | None:
    if not path.exists():
        return None
    recent_lines = deque(maxlen=200)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                recent_lines.append(stripped)
    return recent_lines[-1] if recent_lines else None


def build_snapshot(
    *,
    manifest_path: Path,
    pipeline_state_path: Path,
    extract_log_path: Path,
    index_log_path: Path,
    tail_repair_log_path: Path,
    database_path: Path,
) -> dict[str, Any]:
    manifest_records = load_json(manifest_path, default=[])
    if not isinstance(manifest_records, list):
        raise RuntimeError(f"Manifest payload is not a list: {manifest_path}")

    manifest_stats = compute_manifest_stats(manifest_records)
    pipeline_state = read_pipeline_state(pipeline_state_path)
    process_groups = list_relevant_processes()
    database_counts = query_database_counts(database_path)

    active_stages = [
        stage_name
        for stage_name in ("repair_urls", "download", "extract", "index")
        if process_groups.get(stage_name)
    ]
    wrappers_active = {
        "extract_and_index": bool(process_groups.get("extract_and_index_wrapper")),
        "tail_repair": bool(process_groups.get("tail_repair_wrapper")),
    }

    no_active_stage_process = not active_stages and not wrappers_active["extract_and_index"] and not wrappers_active["tail_repair"]
    success_converged = (
        manifest_stats.download_pending == 0
        and manifest_stats.download_fail == 0
        and manifest_stats.extract_pending == 0
        and manifest_stats.extract_fail == 0
        and manifest_stats.index_pending == 0
        and manifest_stats.index_fail == 0
        and no_active_stage_process
    )

    pending_work = (
        manifest_stats.download_pending > 0
        or manifest_stats.extract_pending > 0
        or manifest_stats.index_pending > 0
    )
    failures_present = (
        manifest_stats.download_fail > 0
        or manifest_stats.extract_fail > 0
        or manifest_stats.index_fail > 0
    )

    if success_converged:
        overall_status = "completed_successfully"
    elif active_stages or wrappers_active["extract_and_index"] or wrappers_active["tail_repair"]:
        overall_status = "running"
    elif pending_work or failures_present:
        overall_status = "waiting_or_stalled"
    else:
        overall_status = "idle"

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall_status,
        "manifest": asdict(manifest_stats),
        "database": database_counts,
        "pipeline_state": pipeline_state,
        "active_stages": active_stages,
        "wrappers_active": wrappers_active,
        "processes": process_groups,
        "last_extract_checkpoint": read_last_checkpoint(extract_log_path),
        "last_index_checkpoint": read_last_checkpoint(index_log_path),
        "last_tail_repair_line": read_last_nonempty_line(tail_repair_log_path),
    }


def write_snapshot(snapshot: dict[str, Any], *, snapshot_log_path: Path, current_state_path: Path) -> None:
    ensure_parent_directory(snapshot_log_path)
    ensure_parent_directory(current_state_path)
    with snapshot_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    current_state_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def progress_key(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    manifest = snapshot["manifest"]
    return (
        manifest["downloaded"],
        manifest["download_pending"],
        manifest["download_fail"],
        manifest["extracted_ok"],
        manifest["extract_fail"],
        manifest["indexed_ok"],
        manifest["index_fail"],
        tuple(snapshot.get("active_stages", [])),
        snapshot.get("pipeline_state", {}).get("stage"),
    )


def main() -> int:
    args = build_argument_parser().parse_args()

    previous_key: tuple[Any, ...] | None = None
    idle_samples = 0

    while True:
        snapshot = build_snapshot(
            manifest_path=args.manifest_path,
            pipeline_state_path=args.pipeline_state_path,
            extract_log_path=args.extract_log_path,
            index_log_path=args.index_log_path,
            tail_repair_log_path=args.tail_repair_log_path,
            database_path=args.database_path,
        )
        write_snapshot(
            snapshot,
            snapshot_log_path=args.snapshot_log_path,
            current_state_path=args.current_state_path,
        )

        current_key = progress_key(snapshot)
        if current_key == previous_key:
            idle_samples += 1
        else:
            idle_samples = 0
        previous_key = current_key

        if args.single_snapshot:
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            return 0

        if snapshot["overall_status"] == "completed_successfully":
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            return 0

        if (
            snapshot["overall_status"] == "waiting_or_stalled"
            and idle_samples >= args.max_idle_samples
        ):
            snapshot["overall_status"] = "stalled"
            snapshot["idle_samples"] = idle_samples
            write_snapshot(
                snapshot,
                snapshot_log_path=args.snapshot_log_path,
                current_state_path=args.current_state_path,
            )
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            return 1

        time.sleep(max(args.sample_interval_seconds, 1))


if __name__ == "__main__":
    raise SystemExit(main())
