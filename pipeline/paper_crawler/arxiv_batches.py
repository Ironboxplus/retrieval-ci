"""Batch handoff between the downloading host and the ingest host.

The HTML is fetched on the host that holds the proxy pool; the ingest task runs
where production SQL and Qdrant live. A batch (``arxiv_corpus`` key
``YYYYMMPP``) crosses over as one bundle:

  bundle   arxiv_<key>.tar = rows.jsonl (every eligible harvest row of the
           batch, whatever its status) + html/<yymm>/<id>.html.gz (status ok)
  push     downloading host: once no row of a batch is pending or retryable,
           the downloader bundles it and copies it to <inbox>/arxiv_<key>.tar
           (written as .part, then renamed, so a reader never sees half a file)
  progress every push round also copies <inbox>/progress.json: per-key status
           counts of the batches still downloading, so the ingest host's Tasks
           panel shows live download progress before any bundle exists
  receive  ingest host: the arXiv download stage imports the rows into its own
           harvest DB and unpacks the HTML (``import_bundle``)

The daily update crosses the same way from the GitHub Actions job, through a 115
volume instead of ssh. Per run the job writes, in this order:

  inbox/arxiv_delta_<UTC>.tar         the rows it changed and their HTML (``write_rows_bundle``),
                                      none when nothing changed
  inbox/arxiv_delta_<UTC>.tar.sha256  written before the bundle, so it is there when the
                                      bundle is: ``<sha256>  <name>``
  inbox/heartbeat.json                last, on every run: {run_id, started_at, finished_at,
                                      bundles: [names written this run], counts...}

The ingest host's delta download stage waits for a heartbeat it has not consumed, takes every
bundle it has not taken (verified against its .sha256), freezes the run, and only then records
the heartbeat as consumed. ``volume_imports`` keeps each bundle's outcome (imported, or
rejected with its reason); ``volume_heartbeats`` the consumed runs.

The harvest DB on the ingest host therefore holds exactly the batches it has
received, which is what the extract stage and the stage probes read.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import sqlite3
import threading
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pipeline.paper_crawler.arxiv_corpus import LEGACY_SLICE, batch_of, slice_label, slice_sql
from pipeline.paper_crawler.arxiv_html import ELIGIBLE_SQL, HTML_CUTOFF, HarvestStore, html_path
from pipeline.paper_crawler.inbox_transfer import (  # noqa: F401 - re-exported for callers and tests
    SSH_OPTIONS,
    TRANSFER_TIMEOUT_S,
    Executable,
    InboxTransfer,
    _default_exe,
)

PUSH_SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_pushes (
    batch_key INTEGER PRIMARY KEY,
    n_rows INTEGER NOT NULL,
    n_html INTEGER NOT NULL,
    bundle_bytes INTEGER NOT NULL,
    pushed_at TEXT NOT NULL
);
"""
ROWS_MEMBER = "rows.jsonl"
PROGRESS_FILE = "progress.json"
HEARTBEAT_FILE = "heartbeat.json"
DELTA_BUNDLE_RE = re.compile(r"^arxiv_delta_\d{8}T\d{6}\.tar$")
# A "lean" row is what the Actions job's seed holds for a paper the ingest host already has:
# status and version facts, the NOT NULL text columns blank. OAI enumerate rewrites them, so a
# row stays lean only if the run never listed that paper again.
LEAN_TEXT_COLUMNS = ("categories", "title", "authors", "abstract")
# The ingest host's ledger of what it took from the volume, in its harvest DB: the outcome per
# bundle (imported_at is when it was recorded), and the Actions runs whose heartbeat it consumed.
VOLUME_SCHEMA = """
CREATE TABLE IF NOT EXISTS volume_imports (
    name TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    imported_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'imported',
    reason TEXT
);
CREATE TABLE IF NOT EXISTS volume_heartbeats (
    run_id TEXT PRIMARY KEY,
    finished_at TEXT NOT NULL,
    consumed_at TEXT NOT NULL
);
"""


def bundle_name(key: int) -> str:
    return f"arxiv_{key}.tar"


def delta_bundle_name(when: datetime) -> str:
    """The package of one daily run: a UTC timestamp, so names sort by time."""
    return f"arxiv_delta_{when.astimezone(timezone.utc):%Y%m%dT%H%M%S}.tar"


def inbox_dir(harvest_root: Path) -> Path:
    return harvest_root / "inbox"


def _batch_rows(db: sqlite3.Connection, key: int) -> list[dict]:
    where, params = slice_sql(key)
    cur = db.execute(f"SELECT * FROM papers WHERE {ELIGIBLE_SQL} AND {where} ORDER BY arxiv_id",
                     [HTML_CUTOFF.isoformat(), *params])
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur]


def unresolved(db: sqlite3.Connection, key: int, max_attempts: int) -> int:
    where, params = slice_sql(key)
    return db.execute(
        f"SELECT COUNT(*) FROM papers WHERE {ELIGIBLE_SQL} AND {where} "
        "AND (status='pending' OR (status='error' AND attempts < ?))",
        [HTML_CUTOFF.isoformat(), *params, max_attempts]).fetchone()[0]


def status_counts(db: sqlite3.Connection, key: int, max_attempts: int) -> dict[str, int]:
    """{status: n} plus error_retryable — the shape the download probe reads."""
    where, params = slice_sql(key)
    counts: dict[str, int] = {}
    for status, retryable, n in db.execute(
            f"SELECT status, attempts < ? AS retryable, COUNT(*) FROM papers "
            f"WHERE {ELIGIBLE_SQL} AND {where} GROUP BY status, retryable",
            [max_attempts, HTML_CUTOFF.isoformat(), *params]):
        counts[status] = counts.get(status, 0) + n
        if status == "error" and retryable:
            counts["error_retryable"] = counts.get("error_retryable", 0) + n
    return counts


def read_progress(inbox: Path) -> dict[str, dict[str, int]]:
    try:
        return json.loads((inbox / PROGRESS_FILE).read_text(encoding="utf-8")).get("batches", {})
    except (OSError, ValueError):
        return {}


def is_lean_row(row: dict) -> bool:
    return all(row.get(column) == "" for column in LEAN_TEXT_COLUMNS)


def write_rows_bundle(rows: list[dict], harvest_root: Path, out: Path, *, html_ids: Optional[set[str]] = None,
                      label: str = "bundle") -> tuple[int, int]:
    """Write harvest ``rows`` (dicts of ``papers`` columns) and their HTML as one bundle at
    ``out``; returns (rows, html files).

    ``html_ids`` None: every row with status ok carries its page, as a batch must. Otherwise
    only the pages of those ids travel, and each must exist: an ok row not named is a row
    from an earlier run whose page the receiving host already holds.

    A lean row is written without ``LEAN_TEXT_COLUMNS``: those are blank placeholders of the
    seed, and the receiving host owns the real values (``import_bundle`` updates only the
    columns a row carries)."""
    rows = [{k: v for k, v in r.items() if k not in LEAN_TEXT_COLUMNS} if is_lean_row(r) else r for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    n_html = 0
    try:
        with tarfile.open(tmp, "w") as tar:
            payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
            info = tarfile.TarInfo(ROWS_MEMBER)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            for r in rows:
                if (r["status"] != "ok") if html_ids is None else (r["arxiv_id"] not in html_ids):
                    continue
                src = html_path(harvest_root, r["arxiv_id"])
                if not src.is_file():
                    raise FileNotFoundError(f"{label}: the page of {r['arxiv_id']} should travel but {src} is missing")
                tar.add(src, arcname=src.relative_to(harvest_root).as_posix())
                n_html += 1
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return len(rows), n_html


def write_bundle(db: sqlite3.Connection, harvest_root: Path, key: int, out: Path) -> tuple[int, int]:
    """Write one batch bundle; returns (rows, html files)."""
    rows = _batch_rows(db, key)
    if not rows:
        raise ValueError(f"batch {key} has no eligible rows")
    return write_rows_bundle(rows, harvest_root, out, label=f"batch {key}")


def _read_rows(tar: tarfile.TarFile) -> list[dict]:
    member = tar.getmember(ROWS_MEMBER)
    return [json.loads(line) for line in tar.extractfile(member).read().decode("utf-8").splitlines() if line]


def bundle_rows(bundle: Path) -> list[dict]:
    """The harvest rows a bundle carries, without importing anything."""
    with tarfile.open(bundle, "r") as tar:
        return _read_rows(tar)


# The columns of a bundle row that the sending host owns: the OAI metadata it enumerated and
# the HTML download state it produced. The rest of a ``papers`` row is the ingest host's own:
# convert_* (its HTML -> Markdown converter) and pdf_* (a PDF fallback whose files never
# travel in a bundle), so an import leaves them alone.
SENDER_COLUMNS = (
    "datestamp", "categories", "primary_category", "title", "authors", "abstract", "comments",
    "license", "latest_version", "latest_version_date", "first_version_date", "latest_size",
    "withdrawn", "content_version", "content_version_date",
    "status", "attempts", "html_version", "raw_bytes", "gz_bytes", "fetched_at", "error",
)

# A row the lean seed started has no page sizes until its page is fetched again, so a NULL from
# the sender never erases the size the ingest host recorded when it fetched the page.
_KEEP_WHEN_UNKNOWN = ("raw_bytes", "gz_bytes")

# As HarvestStore.upsert: a new content version makes the stored PDF the PDF of an older paper.
_CONTENT_CHANGED = ("COALESCE(papers.content_version, papers.latest_version)"
                    " != COALESCE(excluded.content_version, excluded.latest_version)")


def import_bundle(store: HarvestStore, harvest_root: Path, bundle: Path) -> dict[str, int]:
    """Receive one bundle into this host's harvest DB + html tree. Idempotent: a row is
    upserted by arxiv_id with the sender's columns only (SENDER_COLUMNS) and files are
    overwritten. A row without ``LEAN_TEXT_COLUMNS`` (see ``write_rows_bundle``) only updates a
    paper this host has; for an unknown one it cannot be inserted without its metadata, so it is
    skipped and counted in ``skipped_lean``."""
    have = {row[1] for row in store.db.execute("PRAGMA table_info(papers)")}
    counts = {"rows": 0, "html": 0}
    with tarfile.open(bundle, "r") as tar:
        rows = _read_rows(tar)
        for m in tar.getmembers():
            if m.name == ROWS_MEMBER:
                continue
            if not (m.isfile() and m.name.startswith("html/") and ".." not in m.name.split("/")):
                raise ValueError(f"{bundle.name}: unexpected member {m.name!r}")
            dest = harvest_root / m.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(tar.extractfile(m).read())
            os.replace(tmp, dest)
            counts["html"] += 1
    for r in rows:
        cols = ["arxiv_id", *(c for c in SENDER_COLUMNS if c in r and c in have)]
        values = [r[c] for c in cols]
        absent = [c for c in LEAN_TEXT_COLUMNS if c not in r]
        if absent:
            held = store.db.execute(f"SELECT {','.join(absent)} FROM papers WHERE arxiv_id=?",
                                    (r["arxiv_id"],)).fetchone()
            if held is None:
                counts["skipped_lean"] = counts.get("skipped_lean", 0) + 1
                continue
        update = [f"{c}=COALESCE(excluded.{c}, papers.{c})" if c in _KEEP_WHEN_UNKNOWN else f"{c}=excluded.{c}"
                  for c in cols[1:]]
        update += [f"pdf_status=CASE WHEN {_CONTENT_CHANGED} THEN NULL ELSE papers.pdf_status END",
                   f"pdf_attempts=CASE WHEN {_CONTENT_CHANGED} THEN 0 ELSE papers.pdf_attempts END"]
        if absent:
            # NOT NULL is checked before the conflict resolves: the held values only satisfy it
            # and are not in the UPDATE set.
            cols, values = [*cols, *absent], [*values, *held]
        store.db.execute(
            f"INSERT INTO papers({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
            f"ON CONFLICT(arxiv_id) DO UPDATE SET {', '.join(update)}",
            values)
        counts["rows"] += 1
    store.db.commit()
    return counts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_file_name(bundle: str) -> str:
    return bundle + ".sha256"


def checksum_text(bundle: Path) -> str:
    """The content of ``<bundle>.sha256``: ``<sha256>  <name>``, as sha256sum prints it."""
    return f"{file_sha256(bundle)}  {bundle.name}\n"


def parse_checksum(text: bytes | str) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    token = text.split()[0].lower() if text.split() else ""
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError(f"no sha256 in {text[:80]!r}")
    return token


def volume_imports(db: sqlite3.Connection) -> dict[str, str]:
    """{bundle name: imported_at} of the volume bundles this harvest DB has taken in."""
    return dict(db.execute("SELECT name, imported_at FROM volume_imports WHERE status='imported'"))


def volume_rejections(db: sqlite3.Connection) -> dict[str, str]:
    """{bundle name: reason} of the volume bundles found damaged; later runs skip them."""
    return dict(db.execute("SELECT name, reason FROM volume_imports WHERE status='rejected'"))


def record_volume_import(db: sqlite3.Connection, name: str, size: int) -> None:
    db.execute("INSERT OR REPLACE INTO volume_imports(name, size, imported_at, status, reason) "
               "VALUES(?,?,?,'imported',NULL)", (name, size, datetime.now(timezone.utc).isoformat()))
    db.commit()


def record_volume_rejection(db: sqlite3.Connection, name: str, size: int, reason: str) -> None:
    db.execute("INSERT OR REPLACE INTO volume_imports(name, size, imported_at, status, reason) "
               "VALUES(?,?,?,'rejected',?)", (name, size, datetime.now(timezone.utc).isoformat(), reason))
    db.commit()


def last_consumed_heartbeat(db: sqlite3.Connection) -> Optional[tuple[str, str]]:
    """(run_id, finished_at) of the newest heartbeat a completed stage consumed, or None."""
    return db.execute(
        "SELECT run_id, finished_at FROM volume_heartbeats ORDER BY finished_at DESC LIMIT 1").fetchone()


def heartbeat_is_new(db: sqlite3.Connection, beat: dict) -> bool:
    """A heartbeat is new when its run was not consumed and it finished after the last one that was."""
    if db.execute("SELECT 1 FROM volume_heartbeats WHERE run_id=?", (beat["run_id"],)).fetchone():
        return False
    last = last_consumed_heartbeat(db)
    return last is None or beat["finished_at"] > last[1]


def record_heartbeat_consumed(db: sqlite3.Connection, run_id: str, finished_at: str) -> None:
    db.execute("INSERT OR REPLACE INTO volume_heartbeats(run_id, finished_at, consumed_at) VALUES(?,?,?)",
               (run_id, finished_at, datetime.now(timezone.utc).isoformat()))
    db.commit()


class BatchPusher(InboxTransfer):
    """Downloader-side push: bundles every fully resolved batch in scope and
    copies it to the ingest host's inbox. Runs its SQLite work on its own
    connections, so it can sit in a thread beside the downloader's loop."""

    def __init__(self, harvest_db: Path, harvest_root: Path, *, ssh_target: str, remote_inbox: str,
                 scope: Optional[tuple[str, list[object]]], max_attempts: int,
                 log: Callable[[str], None] = print,
                 ssh: Executable | None = None, scp: Executable | None = None) -> None:
        super().__init__(ssh_target=ssh_target, remote_inbox=remote_inbox, ssh=ssh, scp=scp)
        self.harvest_db = harvest_db
        self.harvest_root = harvest_root
        self.scope = scope
        self.max_attempts = max_attempts
        self.log = log
        # The periodic round and the final one run in worker threads and can
        # overlap; two rounds pushing one key would write the same outbox file.
        self._round_lock = threading.Lock()
        db = sqlite3.connect(harvest_db, timeout=120)
        db.executescript(PUSH_SCHEMA)
        db.close()

    def _scan(self) -> tuple[list[int], dict[str, dict[str, int]]]:
        """(keys ready to push, progress of keys not yet pushed)."""
        db = sqlite3.connect(f"file:{self.harvest_db}?mode=ro", uri=True, timeout=120)
        try:
            extra, params = self.scope if self.scope else ("1=1", [])
            ids = [r[0] for r in db.execute(
                f"SELECT arxiv_id FROM papers WHERE {ELIGIBLE_SQL} AND {extra}",
                [HTML_CUTOFF.isoformat(), *params])]
            keys = sorted({k for k in (batch_of(a) for a in ids if "/" not in a) if k}, reverse=True)
            pushed = {r[0] for r in db.execute("SELECT batch_key FROM batch_pushes")}
            ready, progress = [], {}
            for k in keys:
                if k in pushed:
                    continue
                counts = status_counts(db, k, self.max_attempts)
                progress[str(k)] = counts
                if counts.get("pending", 0) + counts.get("error_retryable", 0) == 0:
                    ready.append(k)
            return ready, progress
        finally:
            db.close()

    def _push_progress(self, progress: dict[str, dict[str, int]]) -> None:
        local = self.harvest_root / "outbox" / PROGRESS_FILE
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(),
                                     "batches": progress}), encoding="utf-8")
        self.send(local, PROGRESS_FILE)

    def push(self, key: int) -> dict[str, int]:
        local = self.harvest_root / "outbox" / bundle_name(key)
        db = sqlite3.connect(f"file:{self.harvest_db}?mode=ro", uri=True, timeout=120)
        try:
            n_rows, n_html = write_bundle(db, self.harvest_root, key, local)
        finally:
            db.close()
        size = local.stat().st_size
        self.send(local, bundle_name(key))
        db = sqlite3.connect(self.harvest_db, timeout=120)
        try:
            db.execute("INSERT OR REPLACE INTO batch_pushes VALUES(?,?,?,?,?)",
                       (key, n_rows, n_html, size, datetime.now(timezone.utc).isoformat()))
            db.commit()
        finally:
            db.close()
        local.unlink()
        return {"rows": n_rows, "html": n_html, "bytes": size}

    def round_sync(self) -> int:
        with self._round_lock:
            return self._round_locked()

    def _round_locked(self) -> int:
        ready, progress = self._scan()
        try:
            self._push_progress(progress)
        except Exception as exc:  # progress is advisory; bundles still go
            self.log(f"push progress failed: {type(exc).__name__}: {str(exc)[:300]}")
        pushed = 0
        for key in ready:
            try:
                info = self.push(key)
            except Exception as exc:  # the next round tries again
                self.log(f"push {slice_label(key)} failed: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            pushed += 1
            self.log(f"push {slice_label(key)} ({key}): {info['rows']:,} rows, {info['html']:,} html, "
                     f"{info['bytes'] / 1e6:.0f} MB -> {self.ssh_target}")
        return pushed

    async def round(self) -> int:
        return await asyncio.to_thread(self.round_sync)


__all__ = ["BatchPusher", "DELTA_BUNDLE_RE", "HEARTBEAT_FILE", "LEAN_TEXT_COLUMNS", "LEGACY_SLICE", "VOLUME_SCHEMA",
           "bundle_name", "bundle_rows", "checksum_file_name", "checksum_text", "delta_bundle_name", "file_sha256",
           "heartbeat_is_new", "import_bundle", "inbox_dir", "is_lean_row", "last_consumed_heartbeat", "parse_checksum",
           "read_progress", "record_heartbeat_consumed", "record_volume_import", "record_volume_rejection",
           "status_counts", "unresolved", "volume_imports", "volume_rejections", "write_bundle", "write_rows_bundle"]
