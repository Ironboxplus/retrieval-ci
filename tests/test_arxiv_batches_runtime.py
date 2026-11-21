"""arXiv batch handoff, on real pages, real SQLite and a real ssh transport.

A batch leaves the downloading host as a bundle and enters the ingest host's
harvest DB through the pipeline's download (receive) stage. Nothing is faked:
the HTML is a real harvested page, both harvest DBs are real SQLite files in
temporary directories, and the push test copies to a scratch directory on the
real ingest host over the same ssh the downloader uses (skipped when that host
is unreachable).
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from app.services.pipeline.arxiv_stages import (
    ArxivStageContext,
    arxiv_download_probe,
    discover_batch_keys,
    receive_bundle,
    validate_slice_key,
)
from pipeline.paper_crawler.arxiv_batches import (
    PROGRESS_FILE,
    BatchPusher,
    _default_exe,
    bundle_name,
    import_bundle,
    inbox_dir,
    write_bundle,
)
from pipeline.paper_crawler.arxiv_corpus import batch_of, batch_range, slice_label, slice_sql
from pipeline.paper_crawler.arxiv_html import ArxivRawRecord, HarvestStore, html_path

REPO = Path(__file__).resolve().parents[1]
MIXTRAL = "2401.04088"  # batch 2024-01 #1
LOCAL_HTML = REPO / "data" / "arxiv" / "cs_html" / "html" / "2401" / f"{MIXTRAL}.html.gz"
INGEST_HOST = "4090-88"


def _record(arxiv_id: str) -> ArxivRawRecord:
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp="2024-01-09", categories="cs.LG", title=f"t {arxiv_id}",
        authors="A B", abstract="x", comments=None, license=None, latest_version="v1",
        latest_version_date="2024-01-08T18:47:34+00:00", first_version_date="2024-01-08T18:47:34+00:00",
        latest_size="100kb", content_version="v1", content_version_date="2024-01-08T18:47:34+00:00",
        withdrawn=False)


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    """A downloading host's harvest root with batch 2024-01 #1: one page with
    HTML, one without, one still pending in the neighbouring batch #0."""
    if not LOCAL_HTML.exists():
        pytest.skip("no harvested copy of the Mixtral page")
    root = tmp_path / "source"
    store = HarvestStore(root / "arxiv_cs.sqlite3")
    store.upsert([_record(MIXTRAL), _record("2401.07999"), _record("2401.00001")])
    store.mark(MIXTRAL, "ok", html_version="v1", raw_bytes=1, gz_bytes=1)
    store.mark("2401.07999", "no_html", error="HTTP 404")
    store.close()
    dest = html_path(root, MIXTRAL)
    dest.parent.mkdir(parents=True)
    shutil.copy2(LOCAL_HTML, dest)
    return root


def test_batch_keys():
    assert batch_of("2609.12345") == 20260903
    assert batch_range(20260903) == (12000, 15999)
    assert slice_label(20260903) == "2026-09 #3"
    assert batch_of("cs/0101001") is None
    validate_slice_key(20260903)
    validate_slice_key(202609)
    for bad in (20261303, 20260925, 20231203 - 100):
        with pytest.raises(ValueError):
            validate_slice_key(bad)
    where, params = slice_sql(20240101)
    assert params == ["2401.", 4000, 7999]


def test_bundle_round_trip_and_receive_stage(source: Path, tmp_path: Path):
    import sqlite3

    key = batch_of(MIXTRAL)
    assert key == 20240101
    bundle = tmp_path / "out" / bundle_name(key)
    db = sqlite3.connect(source / "arxiv_cs.sqlite3")
    assert write_bundle(db, source, key, bundle) == (2, 1)  # 2401.00001 is batch #0
    db.close()

    ingest_root = tmp_path / "ingest"
    ctx = ArxivStageContext(venue="arXiv", year=key, store=None, task_id=0,
                            corpus_root=tmp_path / "corpus", harvest_root=ingest_root)
    # Before the bundle: the probe reads the downloading host's progress file.
    inbox_dir(ingest_root).mkdir(parents=True)
    (inbox_dir(ingest_root) / PROGRESS_FILE).write_text(json.dumps(
        {"batches": {str(key): {"ok": 1, "pending": 1}}}), encoding="utf-8")
    before = arxiv_download_probe(ctx)
    assert (before.done, before.total) == (1, 2) and before.message.startswith("downloading")
    # Fully resolved on the downloading host is still not received here: the
    # stage must not be skipped before the bundle is imported.
    (inbox_dir(ingest_root) / PROGRESS_FILE).write_text(json.dumps(
        {"batches": {str(key): {"ok": 1, "no_html": 1}}}), encoding="utf-8")
    resolved = arxiv_download_probe(ctx)
    assert (resolved.done, resolved.total) == (1, 2) and not resolved.is_complete
    assert discover_batch_keys(ingest_root, [202401]) == [key]

    shutil.copy2(bundle, ctx.bundle)
    lines: list[str] = []
    counts = receive_bundle(ctx, lambda m, level="info": lines.append(m), poll_s=0.01)
    assert counts == {"rows": 2, "html": 1}
    after = arxiv_download_probe(ctx)
    assert (after.done, after.total) == (2, 2) and after.message.startswith("received")
    assert html_path(ingest_root, MIXTRAL).read_bytes() == LOCAL_HTML.read_bytes()
    assert ctx.ok_ids() == [MIXTRAL]
    got = sqlite3.connect(ingest_root / "arxiv_cs.sqlite3").execute(
        "SELECT arxiv_id, status, title FROM papers ORDER BY arxiv_id").fetchall()
    assert got == [(MIXTRAL, "ok", f"t {MIXTRAL}"), ("2401.07999", "no_html", "t 2401.07999")]

    # Receiving the same bundle again changes nothing.
    store = HarvestStore(ingest_root / "arxiv_cs.sqlite3")
    assert import_bundle(store, ingest_root, ctx.bundle) == {"rows": 2, "html": 1}
    assert store.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
    store.close()


def _host_up() -> bool:
    try:
        proc = subprocess.run([_default_exe("ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                               INGEST_HOST, "true"], capture_output=True, timeout=30)
        return proc.returncode == 0
    except Exception:
        return False


def test_pusher_hands_resolved_batches_to_the_ingest_host(source: Path):
    if not _host_up():
        pytest.skip(f"ssh {INGEST_HOST} unavailable")
    ssh = _default_exe("ssh")
    remote = f"/tmp/arxiv-push-test-{uuid.uuid4().hex[:8]}/inbox"
    logs: list[str] = []
    pusher = BatchPusher(source / "arxiv_cs.sqlite3", source, ssh_target=INGEST_HOST,
                         remote_inbox=remote, scope=None, max_attempts=3, log=logs.append)
    try:
        # #1 is resolved (ok + no_html); #0 still has a pending paper.
        assert pusher.round_sync() == 1, logs
        listing = subprocess.run([ssh, INGEST_HOST, f"ls {remote}"], capture_output=True, text=True).stdout.split()
        assert sorted(listing) == sorted([bundle_name(20240101), PROGRESS_FILE])
        progress = json.loads(subprocess.run([ssh, INGEST_HOST, f"cat {remote}/{PROGRESS_FILE}"],
                                             capture_output=True, text=True).stdout)["batches"]
        assert progress == {"20240100": {"pending": 1}, "20240101": {"ok": 1, "no_html": 1}}
        remote_size = int(subprocess.run([ssh, INGEST_HOST, f"stat -c %s {remote}/{bundle_name(20240101)}"],
                                         capture_output=True, text=True).stdout)
        # A pushed batch is not pushed again.
        assert pusher.round_sync() == 0
        import sqlite3
        row = sqlite3.connect(source / "arxiv_cs.sqlite3").execute(
            "SELECT batch_key, n_rows, n_html, bundle_bytes FROM batch_pushes").fetchall()
        assert row == [(20240101, 2, 1, remote_size)]
        assert not (source / "outbox" / bundle_name(20240101)).exists()
    finally:
        subprocess.run([ssh, INGEST_HOST, f"rm -rf {remote.rsplit('/', 1)[0]}"], capture_output=True)


def test_dump_upper_bound_keeps_tasks_apart(tmp_path: Path):
    """dump_chunks --max-chunk-id: a task's chunk range excludes chunks written
    after its preindex (the next task's), on a real SQLite file."""
    import sqlite3
    import sys

    db_path = tmp_path / "retrieval.db"
    db = sqlite3.connect(db_path)
    db.executescript("""
        CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT, source TEXT, source_type TEXT,
                                metadata_json TEXT);
        CREATE TABLE papers (paper_id TEXT, doc_id INTEGER, venue TEXT, year INTEGER, venue_tier TEXT,
                             track_type TEXT, arxiv_id TEXT, doi TEXT, citation_count INTEGER,
                             withdrawn INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id INTEGER, content TEXT, level INTEGER,
                             start_idx INTEGER, end_idx INTEGER, metadata_json TEXT);
    """)
    db.execute("INSERT INTO documents VALUES (1, 't', 's', 'paper', '{}')")
    db.executemany("INSERT INTO chunks VALUES (?, 1, ?, 2, ?, ?, '{}')",
                   [(i, f"chunk {i}", i * 10, i * 10 + 9) for i in range(1, 11)])
    db.commit()
    db.close()
    out = tmp_path / "chunks"
    proc = subprocess.run([sys.executable, "scripts/remote_embed/dump_chunks.py", "--db", str(db_path),
                           "--out", str(out), "--min-chunk-id", "4", "--max-chunk-id", "7",
                           "--start-shard", "0"], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    with gzip.open(out / "chunks_0000.jsonl.gz", "rt", encoding="utf-8") as fh:
        contents = [json.loads(line)["content"] for line in fh]
    assert contents == ["chunk 4", "chunk 5", "chunk 6", "chunk 7"]


def test_next_free_shard_skips_past_partial_dumps(tmp_path: Path):
    """A dump killed midway leaves chunks_0056..0060 and a later task wrote
    0061..0066; a re-dump must start at 0067, not at a file-count position."""
    from app.services.pipeline.stages import _next_free_shard

    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    chunks.mkdir()
    vectors.mkdir()
    assert _next_free_shard(chunks, vectors) == 0
    for i in [*range(0, 3), *range(56, 67)]:
        (chunks / f"chunks_{i:04d}.jsonl.gz").write_bytes(b"x")
    (chunks / "done.txt").write_text("x")
    (vectors / "vectors_0070.parquet").write_bytes(b"x")
    assert _next_free_shard(chunks) == 67
    assert _next_free_shard(chunks, vectors) == 71
    assert _next_free_shard(tmp_path / "missing") == 0
