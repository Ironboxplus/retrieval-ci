"""scripts/maintenance/backfill_arxiv_source_version.py on test copies of both databases.

Real SQLite files: the application's tables (migrated, as the backend does at start)
and a harvest database written by HarvestStore; the Qdrant leg runs against scratch
collections on the local server. The real CLI is run as a subprocess.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from core.db.models import Document as DBDocument, Paper as DBPaper, SourceType
from pipeline.paper_crawler.arxiv_html import ArxivRawRecord, HarvestStore
from tests.qdrant_scratch import drop, scratch_names
from tests.sqlite_task_db import make_session_factory

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "maintenance" / "backfill_arxiv_source_version.py"


def raw(arxiv_id: str, *, latest="v1", content=None, withdrawn=False) -> ArxivRawRecord:
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp="2026-04-03", categories="cs.LG", title=arxiv_id, authors="A B",
        abstract="a", comments=None, license=None, latest_version=latest,
        latest_version_date="2026-04-02T00:00:00+00:00", first_version_date="2026-04-01T00:00:00+00:00",
        latest_size="0kb" if withdrawn else "10kb", content_version=content or latest,
        content_version_date="2026-04-02T00:00:00+00:00", withdrawn=withdrawn)


@pytest.fixture()
def dbs(tmp_path):
    factory = make_session_factory(tmp_path / "app.db")
    harvest = HarvestStore(tmp_path / "arxiv_cs.sqlite3")
    harvest.upsert([raw("2601.00001"), raw("2601.00002", latest="v3", content="v2"),
                    raw("2601.00003", withdrawn=True), raw("2601.00004", withdrawn=True),
                    raw("2601.00005")])
    harvest.mark("2601.00001", "ok", html_version="v1")
    harvest.mark("2601.00002", "ok", html_version="v2")  # downloaded version, not the newest
    harvest.mark("2601.00003", "ok", html_version="v1")
    harvest.mark("2601.00004", "ok", html_version="v1")
    harvest.close()
    with factory() as db:
        rows = [("arxiv-2601.00001", None, None, False), ("arxiv-2601.00002", None, None, False),
                ("arxiv-2601.00003", None, None, False),
                ("arxiv-2601.00004", "2601.00004", "v1", True),  # already filled, already withdrawn
                ("arxiv-2601.00005", None, None, True),          # SQL says withdrawn, harvest says not
                ("arxiv-2601.09999", None, None, False),         # not in the harvest at all
                ("cvpr-2026-1", None, None, False)]              # not an arXiv paper
        for pid, arxiv_id, version, withdrawn in rows:
            doc = DBDocument(title=pid, source="s", source_type=SourceType.paper)
            db.add(doc)
            db.flush()
            db.add(DBPaper(paper_id=pid, doc_id=doc.id, title=pid, arxiv_id=arxiv_id,
                           source_version=version, withdrawn=withdrawn))
    return tmp_path, factory


def run(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(tmp_path / "app.db"),
         "--harvest-db", str(tmp_path / "arxiv_cs.sqlite3"), *extra],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    return done, (json.loads(done.stdout) if done.returncode == 0 else {})


def papers(tmp_path: Path) -> dict[str, tuple]:
    conn = sqlite3.connect(tmp_path / "app.db")
    try:
        return {r[0]: r[1:] for r in conn.execute(
            "SELECT paper_id, source_version, withdrawn, withdrawn_at IS NOT NULL FROM papers")}
    finally:
        conn.close()


def test_dry_run_reports_and_writes_nothing(dbs):
    tmp_path, _ = dbs
    before = papers(tmp_path)
    done, report = run(tmp_path)
    assert done.returncode == 0, done.stderr[-800:]
    assert report["mode"] == "dry-run"
    assert report["arxiv_papers"] == 6 and report["in_harvest"] == 5 and report["not_in_harvest"] == 1
    assert report["source_version_to_set"] == 4  # 1, 2, 3, 5; 4 is filled
    assert (report["withdrawn_to_set"], report["withdrawn_to_clear"]) == (1, 1)
    assert papers(tmp_path) == before


def test_apply_fills_versions_and_flags_once(dbs):
    tmp_path, _ = dbs
    done, report = run(tmp_path, "--apply")
    assert done.returncode == 0, done.stderr[-800:]
    now = papers(tmp_path)
    assert now["arxiv-2601.00001"] == ("v1", 0, 0)
    assert now["arxiv-2601.00002"] == ("v2", 0, 0)          # the downloaded version
    assert now["arxiv-2601.00003"] == ("v1", 1, 1)          # newly withdrawn, stamped
    assert now["arxiv-2601.00004"] == ("v1", 1, 0)          # untouched
    assert now["arxiv-2601.00005"] == ("v1", 0, 0)          # revived
    assert now["arxiv-2601.09999"] == (None, 0, 0)          # not in the harvest: stays NULL
    assert now["cvpr-2026-1"] == (None, 0, 0)

    again, second = run(tmp_path, "--apply")
    assert again.returncode == 0
    assert (second["source_version_to_set"], second["withdrawn_to_set"], second["withdrawn_to_clear"]) == (0, 0, 0)
    assert papers(tmp_path) == now


def test_a_value_the_pipeline_wrote_is_never_overwritten(dbs):
    tmp_path, factory = dbs
    with factory.engine.begin() as conn:
        conn.execute(text("UPDATE papers SET source_version='v9' WHERE paper_id='arxiv-2601.00001'"))
    run(tmp_path, "--apply")
    assert papers(tmp_path)["arxiv-2601.00001"][0] == "v9"


def test_cli_refuses_a_database_that_has_not_been_migrated(tmp_path):
    legacy = sqlite3.connect(tmp_path / "app.db")
    legacy.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY, arxiv_id TEXT)")
    legacy.commit()
    legacy.close()
    HarvestStore(tmp_path / "arxiv_cs.sqlite3").close()
    done, _ = run(tmp_path)
    assert done.returncode == 2 and "init_db migrates" in done.stderr
    done, _ = run(tmp_path, "--sync-qdrant")
    assert done.returncode == 2 and "--sync-qdrant needs --apply" in done.stderr


def test_sync_qdrant_marks_withdrawn_papers_in_every_collection(dbs, monkeypatch):
    qdrant_client = pytest.importorskip("qdrant_client")
    from qdrant_client.http import models as qmodels

    from app.models.schemas import Chunk
    from core.storage.corpus_store import paper_store

    tmp_path, _ = dbs
    try:
        client = qdrant_client.QdrantClient(url="http://127.0.0.1:6333", timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant: {exc}")
    names = scratch_names("backfill")
    for name in names.values():
        client.create_collection(
            name, vectors_config={"dense": qmodels.VectorParams(size=8, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()})
    try:
        for n, pid in enumerate(("arxiv-2601.00003", "arxiv-2601.00001"), start=1):
            paper_store(names["arxiv"]).add_chunks(
                [Chunk(id=f"{n}", doc_id=str(n), content=f"text {n}", level=2, start_idx=0, end_idx=5,
                       embedding=[0.5] * 8, metadata={"paper_id": pid, "venue": "arXiv", "year": 2026,
                                                      "venue_tier": "preprint"})], source_type="paper")

        # The real function, on the migrated test copy, with the scratch client and names.
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            import backfill_arxiv_source_version as backfill
        finally:
            sys.path.pop(0)
        conn = sqlite3.connect(tmp_path / "app.db")
        try:
            backfill.apply_backfill(conn, backfill.plan_backfill(conn, sqlite3.connect(tmp_path / "arxiv_cs.sqlite3")))
            assert backfill.sync_qdrant(conn, client, [names["papers"], names["arxiv"]]) == 2  # 00003, 00004
        finally:
            conn.close()

        flagged = client.scroll(names["arxiv"], scroll_filter=qmodels.Filter(must=[
            qmodels.FieldCondition(key="withdrawn", match=qmodels.MatchValue(value=True))]),
            limit=10, with_payload=True)[0]
        assert [p.payload["paper_id"] for p in flagged] == ["arxiv-2601.00003"]
    finally:
        drop(client, names.values())
