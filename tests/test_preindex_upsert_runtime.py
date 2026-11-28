"""The preindex script as an idempotent upsert: new / replaced / unchanged papers.

The real script, real chunker, real SQLite file with the application's tables. The
script's session factory points at that file (the way core.db.database.engine is
pointed at a test engine elsewhere); nothing is faked.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from tests.sqlite_task_db import make_session_factory

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def preindex():
    spec = importlib.util.spec_from_file_location(
        "preindex_upsert_under_test", REPO / "scripts" / "preindex_md_to_sql_only.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


class Env:
    def __init__(self, tmp_path: Path, preindex, factory):
        self.tmp, self.preindex, self.factory = tmp_path, preindex, factory
        self.report_path = tmp_path / "report.json"

    def md(self, name: str, body: str) -> str:
        path = self.tmp / f"{name}.md"
        path.write_text(f"# {name}\n\n## Abstract\n\n{body}\n\n## 1 Introduction\n\n"
                        + ("A sentence that fills the section. " * 40), encoding="utf-8")
        return str(path)

    def record(self, name: str, body: str = "first text", **over) -> dict:
        rec = {
            "paper_id": f"arxiv-2601.{name}", "title": f"Paper {name}", "authors": ["A B"],
            "conference": "arXiv", "year": 2026,
            "pdf_url": f"https://arxiv.org/pdf/2601.{name}",
            "paper_page_url": f"https://arxiv.org/abs/2601.{name}", "source_name": "arxiv_html",
            "markdown_path": self.md(name, body), "arxiv_id": f"2601.{name}",
            "venue_tier": "preprint", "track_type": "preprint",
            "source_version": "v1", "withdrawn": False,
        }
        rec.update(over)
        return rec

    def run(self, records: list[dict], *extra: str, report: Path | None = None) -> dict:
        manifest = self.tmp / "manifest.json"
        manifest.write_text(json.dumps(records), encoding="utf-8")
        args = self.preindex.build_argparser().parse_args(
            ["--manifest-path", str(manifest), "--report-path", str(report or self.report_path), *extra])
        return self.preindex.run(args)

    def sql(self, query: str, **params):
        with self.factory.engine.connect() as conn:
            return conn.execute(text(query), params).fetchall()

    def paper(self, paper_id: str):
        [row] = self.sql("SELECT doc_id, source_version, withdrawn, withdrawn_at FROM papers "
                         "WHERE paper_id=:p", p=paper_id)
        return row

    def n_chunks(self, doc_id: int) -> int:
        return self.sql("SELECT COUNT(*) FROM chunks WHERE doc_id=:d", d=doc_id)[0][0]


@pytest.fixture
def env(tmp_path, preindex, monkeypatch):
    factory = make_session_factory(tmp_path / "app.db")
    monkeypatch.setattr(preindex, "get_db_session", factory)
    monkeypatch.setattr(preindex, "_set_sqlite_pragma_fast", lambda: None)
    return Env(tmp_path, preindex, factory)


def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def test_preindex_retries_a_real_sqlite_writer_lock_without_losing_papers(env):
    import sqlite3
    import threading
    from sqlalchemy import event

    rec = env.record("locked")
    # Force contention past SQLite's per-attempt wait, then release it before
    # the application retry. No fake exception or fake database is involved.
    @event.listens_for(env.factory.engine, "connect")
    def short_timeout(connection, _):
        connection.execute("PRAGMA busy_timeout=20")

    env.factory.engine.dispose()
    lock = sqlite3.connect(str(env.tmp / "app.db"), check_same_thread=False)
    releases = []

    @event.listens_for(env.factory.engine, "before_cursor_execute")
    def hold_at_first_write(_conn, _cursor, statement, _parameters, _context, _many):
        if not releases and statement.lstrip().lower().startswith("insert into documents"):
            lock.execute("BEGIN IMMEDIATE")
            release = threading.Timer(0.2, lock.rollback)
            releases.append(release)
            release.start()
    try:
        result = env.run([rec])
    finally:
        for release in releases:
            release.join()
        lock.close()
    assert releases, "the real writer lock must have overlapped the document insert"
    assert result["failed"] == 0
    assert result["new"] == [rec["paper_id"]]
    assert env.sql("SELECT COUNT(*) FROM papers")[0][0] == 1


def test_new_then_unchanged_then_replaced_with_flags(env):
    a, b = env.record("00001"), env.record("00002", withdrawn=True)

    first = env.run([a, b])
    assert first["counts"]["new"] == 2 and first["failed"] == 0
    assert first["withdrawn_changed"] == [{"paper_id": b["paper_id"], "withdrawn": True}]
    doc_a, version, withdrawn, _ = env.paper(a["paper_id"])
    doc_b, _, withdrawn_b, withdrawn_at = env.paper(b["paper_id"])
    assert (version, withdrawn, bool(withdrawn_b)) == ("v1", 0, True) and withdrawn_at
    assert env.sql("SELECT content_hash FROM documents WHERE id=:d", d=doc_a)[0][0] == digest(a["markdown_path"])
    n_a = env.n_chunks(doc_a)
    assert n_a > 1

    # Same manifest again: nothing is written, the documents are the same rows.
    again = env.run([a, b])
    assert again["counts"]["unchanged"] == 2 and again["indexed"] == 0
    assert (env.paper(a["paper_id"])[0], env.paper(b["paper_id"])[0]) == (doc_a, doc_b)
    assert env.sql("SELECT COUNT(*) FROM documents")[0][0] == 2
    assert env.n_chunks(doc_a) == n_a

    # A new version of A (other text) replaces it; B turns out not withdrawn after all.
    a2 = env.record("00001", body="a rewritten abstract", source_version="v2")
    b2 = env.record("00002", withdrawn=False)
    fresh = env.tmp / "report_v2.json"
    third = env.run([a2, b2], report=fresh)
    assert third["counts"]["replaced"] == 1 and third["counts"]["unchanged"] == 1
    assert third["replaced"] == [a["paper_id"]]
    assert third["superseded"] == [{"doc_id": doc_a, "paper_id": a["paper_id"]}]
    assert third["withdrawn_changed"] == [{"paper_id": b["paper_id"], "withdrawn": False}]
    new_doc_a, version, _, _ = env.paper(a["paper_id"])
    assert new_doc_a != doc_a and version == "v2"
    assert env.n_chunks(new_doc_a) > 1
    # The old document and its chunks stay for cleanup to remove after the vectors.
    assert env.n_chunks(doc_a) == n_a
    assert env.sql("SELECT COUNT(*) FROM papers WHERE paper_id=:p", p=a["paper_id"])[0][0] == 1
    _, _, flag, at = env.paper(b["paper_id"])
    assert (bool(flag), at) == (False, None)

    # The crashed-and-rerun case: the replacement is committed, the report is the same
    # file. A is unchanged now, yet the document it left is still named.
    rerun = env.run([a2, b2], report=fresh)
    assert rerun["superseded"] == [{"doc_id": doc_a, "paper_id": a["paper_id"]}]
    assert rerun["counts"]["unchanged"] == 2
    assert env.paper(a["paper_id"])[0] == new_doc_a


def test_document_without_a_stored_hash_is_compared_by_version_not_rebuilt(env):
    rec = env.record("00003")
    env.run([rec])
    doc, *_ = env.paper(rec["paper_id"])
    with env.factory.engine.begin() as conn:  # a document stored before hashes were written
        conn.execute(text("UPDATE documents SET content_hash=NULL"))

    same_version = env.run([rec], report=env.tmp / "r2.json")
    assert same_version["counts"] == {"new": 0, "replaced": 0, "unchanged": 1,
                                           "superseded": 0, "withdrawn_changed": 0}
    assert env.paper(rec["paper_id"])[0] == doc
    assert env.sql("SELECT content_hash FROM documents WHERE id=:d", d=doc)[0][0] == digest(rec["markdown_path"])

    with env.factory.engine.begin() as conn:
        conn.execute(text("UPDATE documents SET content_hash=NULL"))
    newer = env.record("00003", source_version="v2")
    replaced = env.run([newer], report=env.tmp / "r3.json")
    assert replaced["replaced"] == [rec["paper_id"]]
    assert env.paper(rec["paper_id"])[0] != doc


def test_a_conference_row_without_version_fields_leaves_those_columns_alone(env):
    rec = env.record("00004")
    for key in ("source_version", "withdrawn", "arxiv_id", "venue_tier", "track_type"):
        rec.pop(key)
    rec.update(paper_id="cvpr-2026-1", conference="CVPR")
    report = env.run([rec])
    assert report["counts"]["new"] == 1 and report["withdrawn_changed"] == []
    assert env.paper("cvpr-2026-1")[1:3] == (None, 0)
    again = env.run([rec], report=env.tmp / "again.json")
    assert again["counts"]["unchanged"] == 1


def test_one_failing_paper_does_not_take_its_batch_down(env):
    good1, good2 = env.record("00005"), env.record("00006")
    bad = env.record("00007", title="broken \ud800 title")  # SQLite cannot encode it at INSERT
    report = env.run([good1, bad, good2])
    assert report["failed"] == 1
    assert report["failures_tail"][0]["paper_id"] == bad["paper_id"]
    assert sorted(report["new"]) == sorted([good1["paper_id"], good2["paper_id"]])
    assert env.sql("SELECT COUNT(*) FROM papers")[0][0] == 2
    assert env.sql("SELECT COUNT(*) FROM documents")[0][0] == 2


def test_limit_counts_written_papers_and_duplicates_write_once(env):
    recs = [env.record(f"1000{i}") for i in range(3)]
    report = env.run([recs[0], recs[0], *recs[1:]], "--limit", "2")
    assert report["counts"]["new"] == 2
    assert env.sql("SELECT COUNT(*) FROM papers")[0][0] == 2
