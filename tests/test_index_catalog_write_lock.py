"""The catalog report must not hold SQLite's write lock for long, nor show readers a half-replaced scope.

``IndexCatalogRepository.report_files`` wrote a whole scope (25 s for 100k new rows) in one transaction,
the second long writer next to the journal bulk create (tests/test_journal_bulk_write_lock.py). Real
repository on a real WAL file; a second real connection asks for the write lock all the time
(``tests/sqlite_lock_probe.py``), and a third reads the live inventory the way search does.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from app.services.index_catalog.repository import IndexCatalogRepository
from core.db.models import IndexCatalog
from tests.sqlite_lock_probe import WriteLockProbe, make_wal_session_factory

SCOPE = dict(owner_user_id="owner", host="h", project="C:/proj/catalog", root_dir="C:/proj/catalog")


def _files(indexes, tag: str) -> list[dict]:
    return [{"rel_path": f"src/{i % 300}/f{i}.py", "kind": "code", "language": "python", "chunk_count": 3,
             "content_hash": f"{tag}{i}", "size_bytes": 1000, "metadata": {}} for i in indexes]


def _live_paths(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path), timeout=60)
    try:
        return {row[0] for row in conn.execute(
            "SELECT rel_path FROM index_catalog WHERE deleted_at IS NULL")}
    finally:
        conn.close()


def test_a_big_first_report_gives_other_writers_the_lock_between_batches(tmp_path: Path):
    """30,000 new rows: before the fix one transaction held the lock for the whole insert (~0.25 ms/row)."""
    db = tmp_path / "catalog.db"
    factory = make_wal_session_factory(db)
    repository = IndexCatalogRepository(session_factory=factory)

    with WriteLockProbe(db) as probe:
        started = time.monotonic()
        upserted, removed = repository.report_files(files=_files(range(30_000), "a"), **SCOPE)
        took = time.monotonic() - started

    assert (upserted, removed) == (30_000, 0)
    with factory() as session:
        assert session.query(IndexCatalog).filter(IndexCatalog.deleted_at.is_(None)).count() == 30_000
    assert probe.longest_wait < 2.0, (
        f"another writer waited {probe.longest_wait:.2f}s for the write lock while the report ran for {took:.2f}s")
    assert len(probe.waits) > 5, "the second writer must have been able to run during the report"


def test_a_replace_report_never_hides_a_file_the_new_snapshot_keeps(tmp_path: Path):
    """The old snapshot is 0..9,999, the new one 5,000..14,999. Readers (search, inventory) may see the old
    set, the new set, or the old plus part of the new, never a set missing a file that is in both, and the
    stale files go last: at no instant is anything outside old-or-new visible."""
    db = tmp_path / "catalog.db"
    factory = make_wal_session_factory(db)
    repository = IndexCatalogRepository(session_factory=factory)
    repository.report_files(files=_files(range(0, 10_000), "a"), **SCOPE)
    old = _live_paths(db)
    new_files = _files(range(5_000, 15_000), "a")          # the kept half has the same hashes: unchanged
    new = {f["rel_path"] for f in new_files}
    kept = old & new
    assert len(kept) == 5_000 and len(old | new) == 15_000

    observations: list[set[str]] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            observations.append(_live_paths(db))

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        upserted, removed = repository.report_files(files=new_files, **SCOPE)
    finally:
        stop.set()
        thread.join(30)

    assert (upserted, removed) == (10_000, 5_000)
    assert _live_paths(db) == new
    assert len(observations) > 3
    for seen in observations:
        assert kept <= seen, f"{len(kept - seen)} files of both snapshots were invisible at some instant"
        assert seen <= old | new


def test_a_replace_report_with_an_unchanged_snapshot_changes_nothing(tmp_path: Path):
    factory = make_wal_session_factory(tmp_path / "catalog.db")
    repository = IndexCatalogRepository(session_factory=factory)
    files = _files(range(2_500), "a")
    repository.report_files(files=files, **SCOPE)

    upserted, removed = repository.report_files(files=files, **SCOPE)

    assert (upserted, removed) == (2_500, 0)
    with factory() as session:
        assert session.query(IndexCatalog).filter(IndexCatalog.deleted_at.is_(None)).count() == 2_500
