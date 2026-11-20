"""The arXiv daily update received from a 115 volume, through a real OpenList node.

The GitHub Actions job leaves, in the Cfs volume ``/cfs-arxiv``, ``inbox/arxiv_delta_<UTC>.tar``
bundles with a ``.sha256`` file beside each, and writes ``inbox/heartbeat.json`` last, on every
run (``run_id``, ``finished_at``, ``bundles``). With ``options.source = "volume"`` the delta
download stage waits for a heartbeat it has not consumed yet, imports every bundle it has not
taken, freezes the run as the local stage does, and only then records the heartbeat as consumed.
Bundles here are written by the real ``write_rows_bundle`` from a real harvest DB, the volume is
a Local storage on a real node, and the importer is the real ``import_bundle``.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.pipeline import arxiv_schedule, arxiv_stages
from app.services.pipeline.arxiv_stages import (
    ArxivStageContext,
    make_arxiv_delta_download_stage,
    receive_volume_bundles,
)
from app.services.pipeline.service import INGEST_PIPELINE
from app.services.pipeline.store import TaskStore
from pipeline.openlist_reader import VolumeReader
from pipeline.paper_crawler.arxiv_batches import (
    HEARTBEAT_FILE,
    LEAN_TEXT_COLUMNS,
    bundle_rows,
    delta_bundle_name,
    file_sha256,
    import_bundle,
    inbox_dir,
    last_consumed_heartbeat,
    volume_imports,
    volume_rejections,
    write_rows_bundle,
)
from pipeline.paper_crawler.arxiv_corpus import DELTA_KEY
from pipeline.paper_crawler.arxiv_html import ArxivRawRecord, HarvestStore, html_path
from tests.openlist_node import start_node
from tests.sqlite_task_db import make_session_factory

REPO = Path(__file__).resolve().parents[1]
VOLUME = "/cfs-arxiv"
FIRST, SECOND, THIRD = ("arxiv_delta_20261113T010000.tar", "arxiv_delta_20261114T010000.tar",
                        "arxiv_delta_20261115T010000.tar")


def raw(arxiv_id: str, *, version="v1", date="2026-04-02T00:00:00+00:00") -> ArxivRawRecord:
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp="2026-04-03", categories="cs.LG", title=f"title {arxiv_id}",
        authors="A B", abstract="abstract", comments=None, license=None, latest_version=version,
        latest_version_date=date, first_version_date="2026-04-01T00:00:00+00:00", latest_size="10kb",
        content_version=version, content_version_date=date, withdrawn=False)


@pytest.fixture(scope="module")
def node(tmp_path_factory):
    node = start_node(tmp_path_factory)
    yield node
    node.stop()


@pytest.fixture(scope="module")
def reader_env(node):
    return {"OPENLIST_URL": node.url, "OPENLIST_READER_USER": "retrival-reader",
            "OPENLIST_READER_PASS_FILE": str(node.add_reader())}


@pytest.fixture()
def volume(node, tmp_path):
    mounts = {s["mount_path"]: s for s in node.api("GET", "/api/admin/storage/list?page=1&per_page=100")["content"]}
    if VOLUME in mounts:
        node.api("POST", f"/api/admin/storage/delete?id={mounts[VOLUME]['id']}")
    root = node.mount_local(VOLUME, tmp_path / "volume")
    (root / "inbox").mkdir()
    return root


@pytest.fixture()
def reader(node, reader_env):
    return VolumeReader(reader_env["OPENLIST_URL"], reader_env["OPENLIST_READER_USER"],
                        password_file=Path(reader_env["OPENLIST_READER_PASS_FILE"]))


@pytest.fixture()
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


def ago(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def make_package(workdir: Path, name: str, *, ok=(), no_html=(), dest: Path) -> Path:
    """What the Actions job does at its end: a harvest DB of its own, the rows it changed
    written by ``write_rows_bundle``, the bundle and its .sha256 copied into the inbox."""
    root = workdir / name.removesuffix(".tar")
    harvest = HarvestStore(root / "arxiv_cs.sqlite3")
    harvest.upsert([raw(a) for a in (*ok, *no_html)])
    for arxiv_id in ok:
        harvest.mark(arxiv_id, "ok", html_version="v1", raw_bytes=10, gz_bytes=5)
        page = html_path(root, arxiv_id)
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_bytes(gzip.compress(f"<html>{arxiv_id}</html>".encode()))
    for arxiv_id in no_html:
        harvest.mark(arxiv_id, "no_html", error="HTTP 404")
    rows = _rows(harvest)
    harvest.close()
    out = dest / name
    write_rows_bundle(rows, root, out)
    (dest / (name + ".sha256")).write_text(f"{file_sha256(out)}  {name}\n", encoding="utf-8")
    return out


def put_heartbeat(volume: Path, run_id: str, finished_at: str, bundles: list[str], *, started_at: str | None = None) -> None:
    """The job's last write of a run, bundles or none."""
    beat = {"run_id": run_id, "finished_at": finished_at, "started_at": started_at or ago(30), "bundles": bundles}
    (volume / "inbox" / HEARTBEAT_FILE).write_text(json.dumps(beat), encoding="utf-8")


def deliver(workdir: Path, volume: Path, run_id: str, finished_at: str, bundles: dict[str, dict]) -> None:
    """One whole run of the job: its bundles, then the heartbeat naming them."""
    for name, kw in bundles.items():
        make_package(workdir, name, dest=volume / "inbox", **kw)
    put_heartbeat(volume, run_id, finished_at, list(bundles))


def new_ctx(tmp_path: Path, store, **options) -> ArxivStageContext:
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "arXiv", "year": 0})
    return ArxivStageContext(
        venue="arXiv", year=DELTA_KEY, store=store, task_id=task_id, project_root=REPO,
        corpus_root=tmp_path / "corpus", harvest_root=tmp_path / "harvest",
        options={"source": "volume", "volume": VOLUME, "receive_poll_seconds": 0.05, **options})


def imported_ids(ctx) -> list[str]:
    db = sqlite3.connect(ctx.harvest_db)
    try:
        return [r[0] for r in db.execute("SELECT arxiv_id FROM papers ORDER BY arxiv_id")]
    finally:
        db.close()


def ledger(ctx) -> tuple[dict, dict, tuple | None]:
    db = sqlite3.connect(ctx.harvest_db)
    try:
        return volume_imports(db), volume_rejections(db), last_consumed_heartbeat(db)
    finally:
        db.close()


def quiet(message, level="info"):
    pass


def lines_of(fn):
    lines: list[str] = []
    result = fn(lambda message, level="info": lines.append(message))
    return result, lines


class Counting(VolumeReader):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fetched: list[str] = []

    def fetch(self, path, dest, **kw):
        self.fetched.append(path.rsplit("/", 1)[1])
        return super().fetch(path, dest, **kw)


def counting(reader_env) -> Counting:
    return Counting(reader_env["OPENLIST_URL"], reader_env["OPENLIST_READER_USER"],
                    password_file=Path(reader_env["OPENLIST_READER_PASS_FILE"]))


def _rows(harvest: HarvestStore) -> list[dict]:
    cur = harvest.db.execute("SELECT * FROM papers ORDER BY arxiv_id")
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur]


# ----------------------------------------------------------------- package format
def test_rows_bundle_carries_the_given_rows_and_only_the_html_asked_for(tmp_path):
    root = tmp_path / "h"
    harvest = HarvestStore(root / "arxiv_cs.sqlite3")
    harvest.upsert([raw("2601.00001"), raw("2601.00002"), raw("2601.00003")])
    for arxiv_id in ("2601.00001", "2601.00002"):
        harvest.mark(arxiv_id, "ok", html_version="v1")
    page = html_path(root, "2601.00001")
    page.parent.mkdir(parents=True)
    page.write_bytes(gzip.compress(b"<html/>"))
    rows = _rows(harvest)
    harvest.close()

    # Strict by default: an ok row whose page is missing is an error, as for a batch.
    with pytest.raises(FileNotFoundError, match="2601.00002"):
        write_rows_bundle(rows, root, tmp_path / "strict.tar")
    assert not (tmp_path / "strict.tar").exists() and not list(tmp_path.glob("*.part"))
    # The Actions job names the pages it fetched in this run; an ok row from an earlier
    # day (page kept by the ingest host) travels as a row only.
    assert write_rows_bundle(rows, root, tmp_path / "ok.tar", html_ids={"2601.00001"}) == (3, 1)
    with pytest.raises(FileNotFoundError, match="2601.00003"):
        write_rows_bundle(rows, root, tmp_path / "bad.tar", html_ids={"2601.00003"})


def _lean_row(db: sqlite3.Connection, arxiv_id: str) -> None:
    """A row as a lean seed leaves one the ingest host already holds: status and version
    facts only, the NOT NULL text columns blank."""
    db.execute(
        "INSERT INTO papers(arxiv_id, datestamp, categories, primary_category, title, authors, abstract,"
        " latest_version, latest_version_date, first_version_date, status, attempts, html_version,"
        " fetched_at, content_version, content_version_date)"
        " VALUES(?, '2026-04-03', '', 'cs.LG', '', '', '', 'v1', '2026-04-02T00:00:00+00:00',"
        " '2026-04-01T00:00:00+00:00', 'ok', 1, 'v1', '2026-04-02T01:00:00+00:00', 'v1',"
        " '2026-04-02T00:00:00+00:00')", (arxiv_id,))
    db.commit()


def _row_of(db: sqlite3.Connection, arxiv_id: str) -> dict:
    cur = db.execute("SELECT * FROM papers WHERE arxiv_id=?", (arxiv_id,))
    names = [d[0] for d in cur.description]
    return dict(zip(names, cur.fetchone()))


def test_a_lean_row_travels_without_its_blank_text_columns(tmp_path):
    """The blank title/abstract/authors/categories of a lean row are placeholders of the seed; the
    receiving host owns the real ones, so the row is written without those keys, not refused."""
    root = tmp_path / "h"
    harvest = HarvestStore(root / "arxiv_cs.sqlite3")
    harvest.upsert([raw("2601.00002")])
    _lean_row(harvest.db, "2601.00001")
    rows = _rows(harvest)
    harvest.close()

    assert write_rows_bundle(rows, root, tmp_path / "out.tar", html_ids=set()) == (2, 0)
    lean, full = bundle_rows(tmp_path / "out.tar")
    assert lean["arxiv_id"] == "2601.00001" and not set(LEAN_TEXT_COLUMNS) & set(lean)
    assert (lean["status"], lean["html_version"], lean["fetched_at"], lean["attempts"]) == (
        "ok", "v1", "2026-04-02T01:00:00+00:00", 1)
    assert full["arxiv_id"] == "2601.00002" and all(full[c] for c in LEAN_TEXT_COLUMNS)
    assert not list(tmp_path.glob("*.part"))


def test_a_lean_row_updates_a_paper_the_ingest_host_has_and_keeps_its_text(tmp_path):
    ingest = HarvestStore(tmp_path / "ingest" / "arxiv_cs.sqlite3")
    ingest.upsert([raw("2601.00001")])
    ingest.mark("2601.00001", "no_html", error="HTTP 404")
    root = tmp_path / "h"
    harvest = HarvestStore(root / "arxiv_cs.sqlite3")
    _lean_row(harvest.db, "2601.00001")
    rows = _rows(harvest)
    harvest.close()
    write_rows_bundle(rows, root, tmp_path / "out.tar", html_ids=set())

    assert import_bundle(ingest, tmp_path / "ingest", tmp_path / "out.tar") == {"rows": 1, "html": 0}

    after = _row_of(ingest.db, "2601.00001")
    assert (after["title"], after["authors"], after["abstract"], after["categories"]) == (
        "title 2601.00001", "A B", "abstract", "cs.LG")
    assert (after["status"], after["html_version"], after["fetched_at"], after["error"]) == (
        "ok", "v1", "2026-04-02T01:00:00+00:00", None)


def test_a_lean_row_of_a_paper_the_ingest_host_lacks_is_skipped_and_counted(tmp_path):
    ingest = HarvestStore(tmp_path / "ingest" / "arxiv_cs.sqlite3")
    root = tmp_path / "h"
    harvest = HarvestStore(root / "arxiv_cs.sqlite3")
    _lean_row(harvest.db, "2601.00001")
    rows = _rows(harvest)
    harvest.close()
    write_rows_bundle(rows, root, tmp_path / "out.tar", html_ids=set())

    counts = import_bundle(ingest, tmp_path / "ingest", tmp_path / "out.tar")

    assert counts == {"rows": 0, "html": 0, "skipped_lean": 1}
    assert ingest.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0


def test_a_full_row_still_overwrites_the_text_columns_beside_a_lean_one(tmp_path):
    ingest = HarvestStore(tmp_path / "ingest" / "arxiv_cs.sqlite3")
    ingest.upsert([raw("2601.00001"), raw("2601.00002")])
    root = tmp_path / "h"
    harvest = HarvestStore(root / "arxiv_cs.sqlite3")
    harvest.upsert([replace(raw("2601.00002"), title="new title", abstract="new abstract")])
    _lean_row(harvest.db, "2601.00001")
    rows = _rows(harvest)
    harvest.close()
    write_rows_bundle(rows, root, tmp_path / "out.tar", html_ids=set())

    assert import_bundle(ingest, tmp_path / "ingest", tmp_path / "out.tar") == {"rows": 2, "html": 0}

    assert _row_of(ingest.db, "2601.00002")["title"] == "new title"
    assert _row_of(ingest.db, "2601.00002")["abstract"] == "new abstract"
    assert _row_of(ingest.db, "2601.00001")["title"] == "title 2601.00001"


def test_a_lean_row_the_run_enumerated_again_may_travel_with_its_real_metadata(tmp_path):
    root = tmp_path / "h"
    harvest = HarvestStore(root / "arxiv_cs.sqlite3")
    _lean_row(harvest.db, "2601.00001")
    harvest.upsert([raw("2601.00001")])  # enumerate lists it again: the upsert rewrites its metadata
    rows = _rows(harvest)
    harvest.close()
    assert rows[0]["title"] == "title 2601.00001" and rows[0]["status"] == "ok" and rows[0]["html_version"] == "v1"

    assert write_rows_bundle(rows, root, tmp_path / "out.tar", html_ids=set()) == (1, 0)


def test_delta_bundle_names_are_utc_timestamps_that_sort_by_time():
    early = delta_bundle_name(datetime(2026, 11, 13, 1, 2, 3, tzinfo=timezone.utc))
    late = delta_bundle_name(datetime(2026, 11, 13, 17, 0, 0, tzinfo=timezone.utc))
    assert early == "arxiv_delta_20261113T010203.tar" and early < late


# ------------------------------------------------------------- waiting for the run
def test_bundles_import_in_name_order_and_a_heartbeat_is_consumed_only_by_the_stage(
        node, volume, reader_env, store, tmp_path):
    work = tmp_path / "work"
    deliver(work, volume, "run-1", ago(60), {
        FIRST: {"ok": ("2601.00001",), "no_html": ("2601.00002",)}, SECOND: {"ok": ("2601.00003",)}})
    # What the importer must not take: a batch bundle, an unfinished copy, the checksum files.
    (volume / "inbox" / "arxiv_20260100.tar").write_bytes(b"not ours")
    (volume / "inbox" / "arxiv_delta_20261116T010000.tar.part").write_bytes(b"half")
    spy = counting(reader_env)
    ctx = new_ctx(tmp_path, store)

    delivery, lines = lines_of(lambda sink: receive_volume_bundles(ctx, sink, spy))

    fetched = [n for n in spy.fetched]
    assert fetched == [FIRST, SECOND]  # the .sha256 files are read, not fetched as bundles
    assert delivery.imported == [FIRST, SECOND] and delivery.heartbeat["run_id"] == "run-1"
    assert imported_ids(ctx) == ["2601.00001", "2601.00002", "2601.00003"]
    assert html_path(ctx.harvest_root, "2601.00001").is_file() and html_path(ctx.harvest_root, "2601.00003").is_file()
    assert not html_path(ctx.harvest_root, "2601.00002").exists()  # no_html: a row, no page
    imported, rejected, consumed = ledger(ctx)
    assert sorted(imported) == [FIRST, SECOND] and rejected == {} and all(imported.values())
    assert consumed is None  # receiving is not consuming: the stage records it after the freeze
    assert not list(inbox_dir(ctx.harvest_root).glob("*"))  # the local copies are not kept


def test_the_stage_waits_for_the_days_run_not_for_a_bundle(node, volume, reader, store, tmp_path):
    """A bundle without its run's heartbeat is a run still going: nothing is imported, the stage waits."""
    factory = make_session_factory(tmp_path / "app.db")
    make_package(tmp_path / "work", FIRST, ok=("2601.00001",), dest=volume / "inbox")
    ctx = new_ctx(tmp_path, store, receive_deadline_hours=0.05)
    spec = make_arxiv_delta_download_stage(ctx, session_factory=factory, reader=reader)
    before = spec.probe(ctx)
    assert (before.done, before.total) == (0, 0) and VOLUME in before.message and not spec.gate(ctx).ok

    def finish_run():
        time.sleep(0.8)
        put_heartbeat(volume, "run-1", ago(1), [FIRST])

    thread = threading.Thread(target=finish_run)
    thread.start()
    lines: list[str] = []
    started = time.monotonic()
    spec.run(ctx, lambda message, level="info": lines.append(message))
    thread.join()

    assert time.monotonic() - started >= 0.7  # it did not take the bundle on sight
    assert any("waiting for" in line and f"{VOLUME}/inbox" in line for line in lines), lines
    assert json.loads(ctx.delta_file.read_text(encoding="utf-8"))["ids"] == ["2601.00001"]


def test_a_heartbeat_naming_a_bundle_not_yet_listed_is_not_taken_yet(node, volume, reader, store, tmp_path):
    put_heartbeat(volume, "run-1", ago(1), [FIRST])  # the heartbeat is visible before the bundle it names
    ctx = new_ctx(tmp_path, store, receive_deadline_hours=0.05)

    def bundle_appears():
        time.sleep(0.6)
        make_package(tmp_path / "work", FIRST, ok=("2601.00001",), dest=volume / "inbox")

    thread = threading.Thread(target=bundle_appears)
    thread.start()
    delivery, _ = lines_of(lambda sink: receive_volume_bundles(ctx, sink, reader))
    thread.join()
    assert delivery.imported == [FIRST] and imported_ids(ctx) == ["2601.00001"]


def test_an_empty_delta_completes_normally_and_the_heartbeat_is_consumed(node, volume, reader, store, tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    put_heartbeat(volume, "run-1", ago(60), [])  # the run changed nothing: no bundle at all
    ctx = new_ctx(tmp_path, store, receive_deadline_hours=0.0003)
    spec = make_arxiv_delta_download_stage(ctx, session_factory=factory, reader=reader)

    spec.run(ctx, quiet)

    frozen = json.loads(ctx.delta_file.read_text(encoding="utf-8"))
    assert frozen["ids"] == [] and frozen["counts"]["selected"] == 0 and spec.gate(ctx).ok
    assert ledger(ctx)[2][0] == "run-1"
    probe = spec.probe(ctx)
    assert (probe.done, probe.total) == (1, 1)


def test_a_consumed_heartbeat_makes_the_next_task_wait_for_a_newer_one(node, volume, reader, store, tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    deliver(tmp_path / "work", volume, "run-1", ago(60), {FIRST: {"ok": ("2601.00001",)}})
    first = new_ctx(tmp_path, store)
    make_arxiv_delta_download_stage(first, session_factory=factory, reader=reader).run(first, quiet)
    assert ledger(first)[2][0] == "run-1"

    later = new_ctx(tmp_path, store, receive_deadline_hours=0.0003)
    spec = make_arxiv_delta_download_stage(later, session_factory=factory, reader=reader)
    with pytest.raises(RuntimeError) as failed:
        spec.run(later, quiet)
    message = str(failed.value)
    assert f"{VOLUME}/inbox" in message  # where it looked
    assert "run-1" in message  # the last heartbeat on the volume ...
    assert FIRST in message  # ... and the last bundle imported
    assert "GitHub" in message

    deliver(tmp_path / "work", volume, "run-2", ago(1), {SECOND: {"ok": ("2601.00002",)}})
    again = new_ctx(tmp_path, store)
    make_arxiv_delta_download_stage(again, session_factory=factory, reader=reader).run(again, quiet)
    frozen = json.loads(again.delta_file.read_text(encoding="utf-8"))
    assert frozen["ids"] == ["2601.00001", "2601.00002"]  # the harvest DB holds both, SQL neither
    assert ledger(again)[2][0] == "run-2"


def test_nothing_on_the_volume_fails_with_the_path(node, volume, reader, store, tmp_path):
    ctx = new_ctx(tmp_path, store, receive_deadline_hours=0.0003)
    with pytest.raises(RuntimeError, match=r"/cfs-arxiv/inbox.*no heartbeat.*no bundle has been imported yet"):
        receive_volume_bundles(ctx, quiet, reader)


def test_a_missing_inbox_is_an_empty_one(node, volume, reader, store, tmp_path):
    (volume / "inbox").rmdir()
    ctx = new_ctx(tmp_path, store, receive_deadline_hours=0.0003)
    with pytest.raises(RuntimeError, match="no new run"):
        receive_volume_bundles(ctx, quiet, reader)


def test_papers_imported_by_a_task_whose_freeze_failed_are_picked_up_by_the_next_freeze(
        node, volume, reader, store, tmp_path):
    """freeze_delta compares the harvest DB with SQL, so a later task needs no new bundle for them."""
    factory = make_session_factory(tmp_path / "app.db")
    deliver(tmp_path / "work", volume, "run-1", ago(60), {FIRST: {"ok": ("2601.00001", "2601.00002")}})

    def broken_sql():
        raise sqlite3.OperationalError("database is locked")

    failing = new_ctx(tmp_path, store)
    with pytest.raises(sqlite3.OperationalError):
        make_arxiv_delta_download_stage(failing, session_factory=broken_sql, reader=reader).run(failing, quiet)
    imported, _, consumed = ledger(failing)
    assert sorted(imported) == [FIRST] and consumed is None  # imported, but the run was not completed
    assert not failing.delta_file.exists()

    # A new task (another task id, nothing new on the volume) must not wait for a bundle and
    # must freeze exactly the papers the first one imported.
    retry = new_ctx(tmp_path, store, receive_deadline_hours=0.0003)
    make_arxiv_delta_download_stage(retry, session_factory=factory, reader=reader).run(retry, quiet)

    assert json.loads(retry.delta_file.read_text(encoding="utf-8"))["ids"] == ["2601.00001", "2601.00002"]
    assert ledger(retry)[2][0] == "run-1"


def test_a_resumed_stage_does_not_wait_for_another_run(node, volume, reader, store, tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    deliver(tmp_path / "work", volume, "run-1", ago(60), {FIRST: {"ok": ("2601.00001",)}})
    ctx = new_ctx(tmp_path, store, receive_deadline_hours=0.0003)

    def broken_sql():
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):  # the process dies after the import, before the freeze
        make_arxiv_delta_download_stage(ctx, session_factory=broken_sql, reader=reader).run(ctx, quiet)
    first_start = ctx.state_get("delta_started")
    assert first_start and ctx.state_get("volume_imported") == 1

    # The same task again: the bundle is recorded, the heartbeat not consumed, so it goes straight on.
    make_arxiv_delta_download_stage(ctx, session_factory=factory, reader=reader).run(ctx, quiet)

    assert json.loads(ctx.delta_file.read_text(encoding="utf-8"))["ids"] == ["2601.00001"]
    assert ctx.state_get("delta_started") == first_start


def test_the_stage_imports_freezes_and_counts_this_runs_rows(node, volume, reader, store, tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    deliver(tmp_path / "work", volume, "run-1", ago(60), {
        FIRST: {"ok": ("2601.00001", "2601.00003"), "no_html": ("2601.00002",)}})
    ctx = new_ctx(tmp_path, store)
    spec = make_arxiv_delta_download_stage(ctx, session_factory=factory, reader=reader)
    assert spec.key == "download" and spec.resource_key == "" and not spec.gate(ctx).ok

    lines: list[str] = []
    spec.run(ctx, lambda message, level="info": lines.append(message))

    frozen = json.loads(ctx.delta_file.read_text(encoding="utf-8"))
    assert frozen["ids"] == ["2601.00001", "2601.00003"] and frozen["task_id"] == ctx.task_id
    # The rows were fetched by the Action before this stage began: still "this run's".
    assert frozen["counts"]["html_missing"] == 1 and frozen["html_missing"] == ["2601.00002"]
    assert spec.gate(ctx).ok
    after = spec.probe(ctx)
    assert (after.done, after.total) == (2, 2)  # one bundle imported, then the freeze
    assert any("frozen:" in line for line in lines)


# ----------------------------------------------------------------- bad bundles
def _tamper(path: Path, how: str) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2] if how == "truncate" else data[:2000] + bytes([data[2000] ^ 0xFF]) + data[2001:])


@pytest.mark.parametrize("how", ["truncate", "alter"])
def test_a_corrupt_bundle_is_refetched_once_then_rejected_and_the_stage_fails(
        node, volume, reader_env, store, tmp_path, how):
    factory = make_session_factory(tmp_path / "app.db")
    deliver(tmp_path / "work", volume, "run-1", ago(60), {FIRST: {"ok": ("2601.00001",)}})
    _tamper(volume / "inbox" / FIRST, how)  # the .sha256 beside it still names the original bytes
    spy = counting(reader_env)
    ctx = new_ctx(tmp_path, store)

    with pytest.raises(RuntimeError, match=rf"{FIRST} rejected.*sha256") as failed:
        make_arxiv_delta_download_stage(ctx, session_factory=factory, reader=spy).run(ctx, quiet)

    assert spy.fetched == [FIRST, FIRST]  # once more, then given up
    imported, rejected, consumed = ledger(ctx)
    assert imported == {} and list(rejected) == [FIRST] and "sha256" in rejected[FIRST] and consumed is None
    assert imported_ids(ctx) == [] and not list(inbox_dir(ctx.harvest_root).glob("*"))

    # The same day again: the rejected bundle is skipped (nothing else is new), the run completes,
    # and the stage still says what it left out.
    retry = new_ctx(tmp_path, store, receive_deadline_hours=0.0003)
    spy.fetched.clear()
    lines: list[str] = []
    make_arxiv_delta_download_stage(retry, session_factory=factory, reader=spy).run(
        retry, lambda message, level="info": lines.append(message))
    assert spy.fetched == [] and any(FIRST in line and "rejected" in line for line in lines), lines
    assert ledger(retry)[2][0] == "run-1"

    # The next day's good bundle comes through, and the rejected one is still listed.
    deliver(tmp_path / "work", volume, "run-2", ago(1), {SECOND: {"ok": ("2601.00002",)}})
    day2 = new_ctx(tmp_path, store)
    lines.clear()
    make_arxiv_delta_download_stage(day2, session_factory=factory, reader=spy).run(
        day2, lambda message, level="info": lines.append(message))
    assert spy.fetched == [SECOND] and imported_ids(day2) == ["2601.00002"]
    assert any(FIRST in line and "rejected" in line for line in lines), lines


def test_a_bundle_damaged_in_transit_is_taken_on_the_refetch(node, volume, reader_env, store, tmp_path):
    deliver(tmp_path / "work", volume, "run-1", ago(60), {FIRST: {"ok": ("2601.00001",)}})

    class DamagedOnce(Counting):
        def fetch(self, path, dest, **kw):
            written = super().fetch(path, dest, **kw)
            if len(self.fetched) == 1:
                Path(dest).write_bytes(Path(dest).read_bytes()[:-512])  # a bad first copy
            return written

    spy = DamagedOnce(reader_env["OPENLIST_URL"], reader_env["OPENLIST_READER_USER"],
                      password_file=Path(reader_env["OPENLIST_READER_PASS_FILE"]))
    ctx = new_ctx(tmp_path, store)
    delivery, _ = lines_of(lambda sink: receive_volume_bundles(ctx, sink, spy))
    assert delivery.imported == [FIRST] and spy.fetched == [FIRST, FIRST]
    assert ledger(ctx)[1] == {}


def test_a_bundle_whose_bytes_match_its_checksum_but_is_not_a_tar_is_rejected(
        node, volume, reader, store, tmp_path):
    import hashlib

    garbage = b"this is not a tar archive" * 50
    (volume / "inbox" / FIRST).write_bytes(garbage)
    (volume / "inbox" / (FIRST + ".sha256")).write_text(f"{hashlib.sha256(garbage).hexdigest()}  {FIRST}\n", encoding="utf-8")
    put_heartbeat(volume, "run-1", ago(1), [FIRST])
    ctx = new_ctx(tmp_path, store)
    with pytest.raises(RuntimeError, match=rf"{FIRST} rejected.*not a readable bundle"):
        receive_volume_bundles(ctx, quiet, reader)
    assert list(ledger(ctx)[1]) == [FIRST]


def test_a_bundle_without_its_checksum_file_is_rejected(node, volume, reader, store, tmp_path):
    make_package(tmp_path / "work", FIRST, ok=("2601.00001",), dest=volume / "inbox")
    (volume / "inbox" / (FIRST + ".sha256")).unlink()
    put_heartbeat(volume, "run-1", ago(1), [FIRST])
    ctx = new_ctx(tmp_path, store)
    with pytest.raises(RuntimeError, match=rf"{FIRST} rejected.*no .*sha256"):
        receive_volume_bundles(ctx, quiet, reader)
    assert list(ledger(ctx)[1]) == [FIRST]


def test_a_failed_fetch_is_not_a_rejection(node, volume, store, tmp_path, reader_env):
    """The node being unreachable says nothing about the bundle: the stage fails, nothing is recorded."""
    make_package(tmp_path / "work", FIRST, ok=("2601.00001",), dest=volume / "inbox")
    put_heartbeat(volume, "run-1", ago(1), [FIRST])

    class Unreachable(VolumeReader):
        def fetch(self, path, dest, **kw):
            from pipeline.openlist_reader import VolumeError
            raise VolumeError("GET: connection refused")

    down = Unreachable(reader_env["OPENLIST_URL"], reader_env["OPENLIST_READER_USER"],
                       password_file=Path(reader_env["OPENLIST_READER_PASS_FILE"]))
    ctx = new_ctx(tmp_path, store)
    with pytest.raises(RuntimeError, match="connection refused"):
        receive_volume_bundles(ctx, quiet, down)
    assert ledger(ctx)[:2] == ({}, {})


# ------------------------------------------------------- this host never downloads
LOCAL_REFUSAL = "downloads run on cfscan into the 115 volume; this host only receives"


def test_the_delta_receives_from_the_volume_by_default_and_refuses_the_local_source(tmp_path, store):
    """No options.source means the volume; 'local' is refused when the context is built, so the
    enumerate-and-download stage is never chosen on this host."""
    task_id = store.create_task(task_type=INGEST_PIPELINE, params={"venue": "arXiv", "year": 0})

    def build(**options):
        return ArxivStageContext(venue="arXiv", year=DELTA_KEY, store=store, task_id=task_id, project_root=REPO,
                                 corpus_root=tmp_path / "corpus", harvest_root=tmp_path / "harvest", options=options)

    ctx = build(no_html_retry_days=0)
    assert ctx.source == "volume"
    spec = make_arxiv_delta_download_stage(ctx, session_factory=lambda: None)
    assert spec.resource_key == "" and spec.title == "Receive arXiv delta HTML (volume)"
    with pytest.raises(ValueError, match=LOCAL_REFUSAL):
        build(source="local")


def test_starting_a_local_delta_is_refused_before_a_task_exists(tmp_path, store):
    from app.services.pipeline.service import PipelineRunner

    runner = PipelineRunner(store=store)
    with pytest.raises(ValueError, match=LOCAL_REFUSAL):
        runner.start_arxiv_delta(options={"source": "local"})
    assert store.list_running(INGEST_PIPELINE) == []
    assert store.get_task(1) is None  # no task row was written


def test_the_source_option_is_validated_and_the_schedule_carries_it(tmp_path, store):
    with pytest.raises(ValueError, match="options.source must be 'volume'"):
        new_ctx(tmp_path, store, source="ftp")
    with pytest.raises(ValueError, match=LOCAL_REFUSAL):
        new_ctx(tmp_path, store, source="local")
    assert new_ctx(tmp_path, store).source == "volume"
    default = arxiv_schedule.ArxivDeltaParams()
    assert (default.source, default.receive_deadline_hours, default.volume) == ("volume", 6, "/cfs-arxiv")
    assert arxiv_schedule.ArxivDeltaParams(source="volume", receive_deadline_hours=1.5).source == "volume"
    with pytest.raises(ValueError):
        arxiv_schedule.ArxivDeltaParams(source="local")
    for bad in ({"source": "ftp"}, {"receive_deadline_hours": 0}, {"receive_deadline_hours": 49}):
        with pytest.raises(ValueError):
            arxiv_schedule.ArxivDeltaParams(**bad)
    # What a schedule hands the pipeline round-trips into the context unchanged.
    ctx = new_ctx(tmp_path, store, **{k: v for k, v in arxiv_schedule.ArxivDeltaParams(
        source="volume").model_dump().items() if k != "volume"})
    assert (ctx.source, ctx.volume, ctx.receive_deadline_hours) == ("volume", VOLUME, 6)
