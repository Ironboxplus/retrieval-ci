#!/usr/bin/env python
"""Unified CLI entry for retrival project scripts.

Usage:
    python scripts/retrival.py <group> <command> [args...]

Groups:
    pipeline   Paper collection / extraction / preindex (MD -> SQL)
    collect    Conference-specific collect-into-manifest helpers
    sync       Manifest <-> filesystem (SMB) path reconcilers
    smoke      Local smoke probes (single-PDF marker, etc.)
    embed      Remote BGE-M3 embed pipeline (SQL -> jsonl -> kss GPU -> parquet -> Qdrant)
    qdrant     Qdrant ops (health, recreate, cleanup, payload migration)
    db         SQLite migrations
    verify     API contract / data source verification
    ops        Operational (cleanup zombies, monitor, watch progress)

Examples:
    python scripts/retrival.py                        # list all groups
    python scripts/retrival.py pipeline               # list pipeline commands
    python scripts/retrival.py embed dump --min-chunk-id 7791515 --start-shard 67
    python scripts/retrival.py embed ship up
    python scripts/retrival.py qdrant health
    python scripts/retrival.py qdrant segments
    python scripts/retrival.py verify api-contracts

Args after the command are passed through verbatim to the underlying script.
See scripts/README.md for the full catalog and archived scripts.
"""
from __future__ import annotations
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Per-command spec: (runner, *cmd_parts)
#   runner: "py" -> use venv python or sys.executable
#           "sh" -> bash
#   cmd_parts: script path (and optional fixed leading args)
COMMANDS: dict[tuple[str, str], tuple[str, ...]] = {
    # ─── pipeline: paper collection / extraction / preindex ───
    ("pipeline", "preindex-md"):       ("py", "scripts/preindex_md_to_sql_only.py"),
    ("pipeline", "preindex-corpus"):   ("py", "scripts/preindex_existing_extracted_corpus_papers.py"),
    ("pipeline", "rebuild-corpus"):    ("py", "scripts/rebuild_ccf_a_main_corpus.py"),
    ("pipeline", "run-2025"):          ("py", "scripts/run_2025_pipeline.py"),
    ("pipeline", "download-subset"):   ("py", "scripts/download_subset.py"),
    ("pipeline", "download-arxiv"):    ("py", "scripts/download_via_arxiv.py"),
    ("pipeline", "download-missing"):  ("py", "scripts/download_missing_pdfs.py"),

    # ─── collect: per-conference manifest-merge helpers ───
    ("collect", "year-slice"):         ("py", "scripts/collect/collect_2026_to_manifest.py"),
    ("collect", "acm"):                ("py", "scripts/collect/collect_and_download_acm.py"),
    ("collect", "emnlp"):              ("py", "scripts/collect/collect_and_download_emnlp.py"),

    # ─── sync: manifest <-> filesystem reconcilers ───
    ("sync", "manifest"):              ("py", "scripts/sync/sync_manifest_pdf_md_paths.py"),
    ("sync", "iclr-manifest"):         ("py", "scripts/sync/sync_iclr_manifest_from_smb.py"),
    ("sync", "iclr-md"):               ("py", "scripts/sync/sync_iclr_md_smb.py"),
    ("sync", "new-conf-md"):           ("py", "scripts/sync/sync_new_conf_md_smb.py"),

    # ─── smoke: local single-PDF / single-component probes ───
    ("smoke", "marker"):               ("py", "scripts/smoke/smoke_marker.py"),
    ("smoke", "real-pdf-download"):    ("py", "scripts/smoke/smoke_download_real_pdf.py"),

    # ─── embed: remote BGE-M3 embed pipeline ───
    ("embed", "dump"):                 ("py", "scripts/remote_embed/dump_chunks.py"),
    ("embed", "worker"):               ("py", "scripts/remote_embed/embed_worker.py"),
    ("embed", "ingest"):               ("py", "scripts/remote_embed/local_ingest.py"),
    ("embed", "ship"):                 ("sh", "scripts/remote_embed/ship.sh"),
    ("embed", "run-kss"):              ("sh", "scripts/remote_embed/run_embed_kss.sh"),

    # ─── qdrant: collection / payload / ops ───
    ("qdrant", "health"):              ("py", "tools/qdrant_ops.py", "health"),
    ("qdrant", "segments"):            ("py", "tools/qdrant_ops.py", "segments"),
    ("qdrant", "progress"):            ("py", "tools/qdrant_ops.py", "progress"),
    ("qdrant", "search"):              ("py", "tools/qdrant_ops.py", "search"),
    ("qdrant", "start"):               ("py", "tools/qdrant_ops.py", "start"),
    ("qdrant", "stop"):                ("py", "tools/qdrant_ops.py", "stop"),
    ("qdrant", "recreate"):            ("py", "scripts/recreate_qdrant_collection.py"),
    ("qdrant", "cleanup-orphans"):     ("py", "scripts/cleanup_qdrant_orphans.py"),
    ("qdrant", "migrate-payload"):     ("py", "scripts/migrate_qdrant_payload.py"),

    # ─── db: SQLite migrations (one-shot, applied; kept for rollback ref) ───
    ("db", "migrate-papers-pk"):       ("py", "scripts/migrate_papers_pk.py"),

    # ─── verify: API contracts and data source checks ───
    ("verify", "live-apis"):           ("py", "scripts/live_verify_all_apis.py"),
    ("verify", "api-contracts"):       ("py", "scripts/verify_live_api_contracts.py"),
    ("verify", "2025-sources"):        ("py", "scripts/verify_2025_sources.py"),

    # ─── ops: maintenance / monitoring ───
    ("ops", "cleanup-zombies"):        ("py", "scripts/cleanup_zombie_tasks.py"),
    ("ops", "monitor-corpus"):         ("py", "scripts/monitor_ccf_a_main_corpus_pipeline.py"),
    ("ops", "watch-progress"):         ("sh", "scripts/watch_progress.sh"),
}

GROUP_DESC = {
    "pipeline": "Paper collection / extraction / preindex (MD -> SQL)",
    "collect":  "Conference-specific collect-into-manifest helpers",
    "sync":     "Manifest <-> filesystem (SMB) path reconcilers",
    "smoke":    "Local smoke probes (single-PDF marker, etc.)",
    "embed":    "Remote BGE-M3 embed pipeline (SQL -> jsonl -> kss GPU -> parquet -> Qdrant)",
    "qdrant":   "Qdrant ops (health, recreate, cleanup, payload migration)",
    "db":       "SQLite migrations (one-shot, applied; kept for rollback reference)",
    "verify":   "API contract / data source verification",
    "ops":      "Operational (cleanup zombies, monitor, watch progress)",
}


def venv_python() -> str:
    """Return the project .venv python on Windows / Linux, falling back to sys.executable."""
    for candidate in (ROOT / ".venv/Scripts/python.exe", ROOT / ".venv/bin/python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def print_top_help() -> None:
    print(__doc__)
    print("Available groups:")
    for g, desc in GROUP_DESC.items():
        print(f"  {g:<10} {desc}")
    print()
    print("Run `python scripts/retrival.py <group>` to list commands in that group.")


def print_group_help(group: str) -> None:
    print(f"Group: {group}  -  {GROUP_DESC.get(group, '')}")
    print()
    print("Commands:")
    for (g, cmd), spec in COMMANDS.items():
        if g != group:
            continue
        path = spec[1]
        extra = " ".join(spec[2:])
        suffix = f" [pre-args: {extra}]" if extra else ""
        print(f"  {cmd:<22} -> {path}{suffix}")
    print()
    print(f"Run `python scripts/retrival.py {group} <command> --help` to pass --help through.")


def dispatch(group: str, cmd: str, passthrough: list[str]) -> int:
    key = (group, cmd)
    spec = COMMANDS.get(key)
    if spec is None:
        print(f"error: unknown command '{group} {cmd}'", file=sys.stderr)
        print_group_help(group)
        return 2
    runner, script, *prepend = spec
    script_abs = ROOT / script
    if not script_abs.exists():
        print(f"error: script not found: {script_abs}", file=sys.stderr)
        return 2

    if runner == "py":
        argv = [venv_python(), str(script_abs), *prepend, *passthrough]
    elif runner == "sh":
        argv = ["bash", str(script_abs), *prepend, *passthrough]
    else:
        print(f"error: unknown runner '{runner}' for {key}", file=sys.stderr)
        return 2

    # Run from project root so relative paths in scripts resolve consistently.
    print(f"[retrival] -> {' '.join(shlex.quote(a) for a in argv)}", file=sys.stderr)
    return subprocess.call(argv, cwd=str(ROOT))


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print_top_help()
        return 0
    group = argv[0]
    if group not in GROUP_DESC:
        print(f"error: unknown group '{group}'", file=sys.stderr)
        print_top_help()
        return 2
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print_group_help(group)
        return 0
    cmd = argv[1]
    passthrough = argv[2:]
    return dispatch(group, cmd, passthrough)


if __name__ == "__main__":
    sys.exit(main())
