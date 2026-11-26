"""Cleanup stage (stage 6): what a re-indexed or withdrawn paper leaves behind.

Real SQLite file with the application's tables (foreign keys on, so the chunk
cascade is the database's own), real local Qdrant with scratch collections.
No fakes: the stage is handed the scratch client the way production hands it
the deployment's.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.models.schemas import Chunk  # noqa: E402
from app.services.pipeline.context import StageContext  # noqa: E402
from app.services.pipeline.stages import (  # noqa: E402
    CleanupPlan,
    apply_cleanup,
    cleanup_gate,
    cleanup_probe,
    cleanup_state_path,
    load_cleanup_plan,
    make_cleanup_stage,
    preindex_report_path,
)
from app.services.pipeline.store import TaskStore  # noqa: E402
from core.db.models import Chunk as DBChunk, Document as DBDocument, Paper as DBPaper, SourceType  # noqa: E402
from core.storage import corpus_layout  # noqa: E402
from core.storage.corpus_store import paper_store  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402
from tests.sqlite_task_db import make_session_factory  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
DIM = 8


@pytest.fixture()
def qdrant(monkeypatch, tmp_path):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    names = scratch_names("cleanup")
    monkeypatch.setattr(corpus_layout, "QDRANT_PAPERS_COLLECTION", names["papers"])
    monkeypatch.setattr(corpus_layout, "QDRANT_ARXIV_COLLECTION", names["arxiv"])
    monkeypatch.setattr(corpus_layout, "QDRANT_REVIEWS_COLLECTION", names["reviews"])
    monkeypatch.setenv("QDRANT_CORPUS_LAYOUT_FILE", str(tmp_path / "corpus_layout.json"))
    for name in names.values():
        client.delete_collection(name)
        client.create_collection(
            name,
            vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
    client.names = names
    yield client
    drop(client, names.values())


class World:
    """One SQLite file plus the scratch collections, with helpers to seed both."""

    def __init__(self, tmp_path: Path, client):
        self.tmp, self.client = tmp_path, client
        self.factory = make_session_factory(tmp_path / "app.db")
        self.collections = [client.names["papers"], client.names["arxiv"]]
        self.state = tmp_path / "state.json"

    def document(self, paper_id: str, n_chunks: int, *, points: bool, link: bool = True) -> int:
        """A document with n chunks (and as many points in the arxiv collection)."""
        with self.factory() as db:
            doc = DBDocument(title=paper_id, source="s", source_type=SourceType.paper)
            db.add(doc)
            db.flush()
            for i in range(n_chunks):
                db.add(DBChunk(doc_id=doc.id, chunk_index=i, content=f"c{i}", level=2, start_idx=0, end_idx=2,
                               metadata_json={"paper_id": paper_id}))
            if link:
                existing = db.get(DBPaper, paper_id)
                if existing is None:
                    db.add(DBPaper(paper_id=paper_id, doc_id=doc.id, title=paper_id))
                else:
                    existing.doc_id = doc.id
            doc_id = doc.id
        if points:
            paper_store(self.client.names["arxiv"]).add_chunks(
                [Chunk(id=f"{doc_id}-{i}", doc_id=str(doc_id), content=f"text {doc_id} {i}", level=2,
                       start_idx=0, end_idx=6, embedding=[0.5] * DIM,
                       metadata={"paper_id": paper_id, "venue": "arXiv", "year": 2026,
                                 "venue_tier": "preprint"})
                 for i in range(n_chunks)], source_type="paper")
        return doc_id

    def points(self, doc_id: int) -> int:
        flt = qmodels.Filter(must=[qmodels.FieldCondition(
            key="doc_id", match=qmodels.MatchValue(value=str(doc_id)))])
        return sum(self.client.count(name, count_filter=flt, exact=True).count for name in self.collections)

    def withdrawn_points(self) -> int:
        flt = qmodels.Filter(must=[qmodels.FieldCondition(key="withdrawn", match=qmodels.MatchValue(value=True))])
        return sum(self.client.count(name, count_filter=flt, exact=True).count for name in self.collections)

    def sql_count(self, model, **where) -> int:
        with self.factory() as db:
            return db.query(model).filter_by(**where).count()

    def apply(self, plan: CleanupPlan, state: Path | None = None):
        return apply_cleanup(plan, session_factory=self.factory, client=self.client,
                             collections=self.collections, state_path=state or self.state)


@pytest.fixture()
def world(tmp_path, qdrant):
    return World(tmp_path, qdrant)


def plan(superseded=None, withdrawn=None, report=None) -> CleanupPlan:
    return CleanupPlan(superseded=superseded or {}, withdrawn=withdrawn or {}, report=report or {})


def test_replaced_document_loses_vectors_then_rows_and_the_new_one_stays(world):
    old = world.document("arxiv-2601.1", 3, points=True)
    new = world.document("arxiv-2601.1", 2, points=True)  # the paper moved to the new document
    other = world.document("arxiv-2601.2", 2, points=True)
    assert (world.points(old), world.points(new)) == (3, 2)
    assert world.sql_count(DBChunk, doc_id=old) == 3

    outcome = world.apply(plan({old: "arxiv-2601.1"}))

    assert outcome == {"superseded_deleted": 1, "superseded_kept": 0}
    assert (world.points(old), world.points(new), world.points(other)) == (0, 2, 2)
    assert world.sql_count(DBDocument, id=old) == 0
    assert world.sql_count(DBChunk, doc_id=old) == 0  # ON DELETE CASCADE
    assert world.sql_count(DBChunk, doc_id=new) == 2 and world.sql_count(DBChunk, doc_id=other) == 2
    with world.factory() as db:
        assert db.get(DBPaper, "arxiv-2601.1").doc_id == new
    assert json.loads(world.state.read_text())["superseded_done"] == [old]


def test_cleanup_twice_gives_the_same_end_state(world):
    old = world.document("arxiv-2601.3", 3, points=True)
    new = world.document("arxiv-2601.3", 2, points=True)
    p = plan({old: "arxiv-2601.3"})
    world.apply(p)
    again = world.apply(p)  # same state file: nothing left to do
    assert again == {"superseded_deleted": 0, "superseded_kept": 0}
    fresh = world.apply(p, state=world.tmp / "fresh.json")  # a lost state file: still harmless
    assert fresh["superseded_kept"] == 0
    assert (world.points(old), world.points(new)) == (0, 2)
    assert world.sql_count(DBDocument, id=old) == 0 and world.sql_count(DBDocument, id=new) == 1


def test_a_document_the_paper_still_points_at_is_kept(world):
    # The report names the document, but the preindex commit that would have moved the
    # paper off it never happened (or was rolled back): deleting it would lose a live paper.
    doc = world.document("arxiv-2601.4", 3, points=True)
    outcome = world.apply(plan({doc: "arxiv-2601.4"}))
    assert outcome == {"superseded_deleted": 0, "superseded_kept": 1}
    assert world.points(doc) == 3 and world.sql_count(DBDocument, id=doc) == 1
    assert json.loads(world.state.read_text())["superseded_kept"] == [doc]


def test_old_vectors_are_never_the_last_copy(world):
    old = world.document("arxiv-2601.5", 3, points=True)
    new = world.document("arxiv-2601.5", 2, points=False)  # embed/ingest did not finish
    with pytest.raises(RuntimeError, match="has no points"):
        world.apply(plan({old: "arxiv-2601.5"}))
    assert world.points(old) == 3 and world.sql_count(DBDocument, id=old) == 1
    assert world.sql_count(DBChunk, doc_id=old) == 3

    paper_store(world.client.names["arxiv"]).add_chunks(
        [Chunk(id=f"{new}-{i}", doc_id=str(new), content=f"t {new} {i}", level=2, start_idx=0, end_idx=1,
               embedding=[0.5] * DIM, metadata={"paper_id": "arxiv-2601.5", "venue": "arXiv", "year": 2026,
                                                "venue_tier": "preprint"}) for i in range(2)],
        source_type="paper")
    assert world.apply(plan({old: "arxiv-2601.5"}))["superseded_deleted"] == 1
    assert world.points(old) == 0 and world.points(new) == 2


def test_partial_replacement_never_deletes_the_old_complete_document(world):
    old = world.document("arxiv-2601.partial", 3, points=True)
    new = world.document("arxiv-2601.partial", 2, points=True)
    # SQL expects a third chunk that ingest has not written yet.
    with world.factory() as db:
        db.add(DBChunk(doc_id=new, chunk_index=2, content="missing", level=2,
                       start_idx=2, end_idx=3))
    with pytest.raises(RuntimeError, match="incomplete"):
        world.apply(plan({old: "arxiv-2601.partial"}))
    assert world.points(old) == 3
    assert world.sql_count(DBDocument, id=old) == 1


def test_retried_cleanup_does_not_restore_an_outdated_withdrawn_flag(world):
    pid = "arxiv-2601.revised"
    doc = world.document(pid, 2, points=True)
    with world.factory() as db:
        db.get(DBPaper, pid).withdrawn = False
    outdated = CleanupPlan(superseded={}, withdrawn={pid: True},
                          report={"withdrawn_changed": [{"paper_id": pid, "withdrawn": True}]})
    world.apply(outdated)
    with world.factory() as db:
        assert db.get(DBPaper, pid).withdrawn is False
    assert world.withdrawn_points() == 0
    assert world.points(doc) == 2

def test_a_crash_midway_resumes_without_redoing_finished_documents(world):
    old1 = world.document("arxiv-2601.6", 2, points=True)
    new1 = world.document("arxiv-2601.6", 2, points=True)
    old2 = world.document("arxiv-2601.7", 2, points=True)
    new2 = world.document("arxiv-2601.7", 2, points=False)
    p = plan({old1: "arxiv-2601.6", old2: "arxiv-2601.7"})
    with pytest.raises(RuntimeError):
        world.apply(p)
    assert json.loads(world.state.read_text())["superseded_done"] == [old1]
    assert (world.points(old1), world.points(old2)) == (0, 2)

    paper_store(world.client.names["arxiv"]).add_chunks(
        [Chunk(id=f"{new2}-{i}", doc_id=str(new2), content=f"t {new2} {i}", level=2, start_idx=0, end_idx=1,
               embedding=[0.5] * DIM, metadata={"paper_id": "arxiv-2601.7", "venue": "arXiv", "year": 2026,
                                                "venue_tier": "preprint"}) for i in range(2)],
        source_type="paper")
    assert world.apply(p) == {"superseded_deleted": 1, "superseded_kept": 0}  # only old2 is left to do
    assert (world.points(old2), world.points(new1), world.points(new2)) == (0, 2, 2)
    assert sorted(json.loads(world.state.read_text())["superseded_done"]) == sorted([old1, old2])


def test_withdrawn_flag_round_trips_between_sql_and_every_collection(world):
    world.document("arxiv-2601.8", 2, points=True)
    world.document("arxiv-2601.9", 2, points=True)
    flagged = plan(withdrawn={"arxiv-2601.8": True})
    world.apply(flagged)
    with world.factory() as db:
        row = db.get(DBPaper, "arxiv-2601.8")
        assert row.withdrawn is True and row.withdrawn_at
        assert db.get(DBPaper, "arxiv-2601.9").withdrawn is False
    assert world.withdrawn_points() == 2  # only that paper's two points

    world.apply(flagged)  # same plan again: unchanged
    assert world.withdrawn_points() == 2

    world.apply(plan(withdrawn={"arxiv-2601.8": False}), state=world.tmp / "revive.json")
    with world.factory() as db:
        row = db.get(DBPaper, "arxiv-2601.8")
        assert row.withdrawn is False and row.withdrawn_at is None
    assert world.withdrawn_points() == 0


# ------------------------------------------------------------------ the stage
@pytest.fixture()
def stage_world(world):
    store = TaskStore(world.factory)
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = StageContext(venue="arXiv", year=0, store=store, task_id=task_id,
                       project_root=world.tmp, corpus_root=world.tmp / "corpus")
    return world, store, ctx


def write_report(ctx: StageContext, report: dict) -> None:
    path = preindex_report_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_stage_reads_the_report_probes_runs_and_writes_the_counts(stage_world):
    world, store, ctx = stage_world
    old = world.document("arxiv-2601.10", 3, points=True)
    world.document("arxiv-2601.10", 2, points=True)
    world.document("arxiv-2601.11", 2, points=True)
    with world.factory() as db:
        db.get(DBPaper, "arxiv-2601.11").withdrawn = True
    write_report(ctx, {
        "new": ["arxiv-2601.11"], "replaced": ["arxiv-2601.10"], "unchanged": ["a", "b"], "failed": 1,
        "superseded": [{"doc_id": old, "paper_id": "arxiv-2601.10"}],
        "withdrawn_changed": [{"paper_id": "arxiv-2601.11", "withdrawn": True}],
    })
    spec = make_cleanup_stage(ctx, session_factory=world.factory,
                              qdrant=lambda: (world.client, world.collections))
    assert spec.resource_key == "qdrant"
    assert load_cleanup_plan(ctx).superseded == {old: "arxiv-2601.10"}
    assert (cleanup_probe(ctx).done, cleanup_probe(ctx).total) == (0, 2)
    assert not cleanup_gate(ctx).ok

    lines: list[str] = []
    spec.run(ctx, lambda m, level="info": lines.append(m))

    assert (cleanup_probe(ctx).done, cleanup_probe(ctx).total) == (2, 2) and cleanup_gate(ctx).ok
    assert world.points(old) == 0 and world.withdrawn_points() == 2
    assert store.get_task(ctx.task_id)["result"] == {
        "new": 1, "replaced": 1, "unchanged": 2, "withdrawn": 1, "revived": 0, "preindex_failed": 1,
        "superseded_deleted": 1, "superseded_kept": 0}
    assert cleanup_state_path(ctx).exists()

    spec.run(ctx, lambda *a, **k: None)  # a second run changes nothing
    assert world.points(old) == 0 and world.withdrawn_points() == 2


def test_stage_without_a_report_has_nothing_to_do(stage_world):
    world, store, ctx = stage_world
    spec = make_cleanup_stage(ctx, session_factory=world.factory,
                              qdrant=lambda: (world.client, world.collections))
    assert (cleanup_probe(ctx).done, cleanup_probe(ctx).total) == (0, 0) and cleanup_gate(ctx).ok
    spec.run(ctx, lambda *a, **k: None)
    assert store.get_task(ctx.task_id)["result"]["superseded_deleted"] == 0


def test_withdrawn_only_papers_of_a_delta_run_are_synced_without_any_preindex_report(world):
    """A run whose papers changed only their withdrawn flag has no manifest, so the preindex
    report lists nothing; the frozen selection is the source (arxiv_stages.freeze_delta)."""
    from app.services.pipeline.arxiv_stages import ArxivStageContext

    store = TaskStore(world.factory)
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = ArxivStageContext(venue="arXiv", year=0, store=store, task_id=task_id, project_root=world.tmp,
                            corpus_root=world.tmp / "corpus", harvest_root=world.tmp / "harvest")
    for n in range(1, 5):
        world.document(f"arxiv-2601.2000{n}", 2, points=True)
    with world.factory() as db:  # one is already flagged in SQL (stale Qdrant), one gets revived
        row = db.get(DBPaper, "arxiv-2601.20003")
        row.withdrawn = True
        row.withdrawn_at = "2026-01-01T00:00:00+00:00"
    ctx.delta_file.parent.mkdir(parents=True)
    ctx.delta_file.write_text(json.dumps({
        "ids": [], "html_missing": [],
        "withdrawn_only": {"arxiv-2601.20001": True, "arxiv-2601.20002": True,
                           "arxiv-2601.20003": False, "arxiv-2601.20004": True},
        "counts": {"selected": 0, "withdrawn_only": 4, "html_missing": 0, "download_errors": 0}}),
        encoding="utf-8")
    assert not preindex_report_path(ctx).exists()

    spec = make_cleanup_stage(ctx, session_factory=world.factory,
                              qdrant=lambda: (world.client, world.collections))
    assert (cleanup_probe(ctx).done, cleanup_probe(ctx).total) == (0, 1)
    spec.run(ctx, lambda *a, **k: None)

    with world.factory() as db:
        flags = {p.paper_id: (p.withdrawn, p.withdrawn_at is not None) for p in db.query(DBPaper)}
    assert flags == {"arxiv-2601.20001": (True, True), "arxiv-2601.20002": (True, True),
                     "arxiv-2601.20003": (False, False), "arxiv-2601.20004": (True, True)}
    assert world.withdrawn_points() == 6  # three papers x two points; the revived one carries none
    result = store.get_task(task_id)["result"]
    assert (result["withdrawn"], result["revived"], result["new"], result["replaced"]) == (3, 1, 0, 0)
    assert result["withdrawn_only"] == 4 and result["superseded_deleted"] == 0
    assert cleanup_gate(ctx).ok

    spec.run(ctx, lambda *a, **k: None)  # re-driven: same end state
    assert world.withdrawn_points() == 6


def test_stage_removes_the_task_embed_directory_once_ingest_is_complete(stage_world):
    from app.services.pipeline.stages import _embed_task_dir

    world, store, ctx = stage_world
    task_dir = _embed_task_dir(ctx)
    vectors = task_dir / "vectors"
    vectors.mkdir(parents=True)
    (vectors / "vectors_0000.parquet").write_bytes(b"x")
    ctx.state_set(chunk_id_before=0, chunk_id_after=5, shard_start=0, shard_end=1)
    spec = make_cleanup_stage(ctx, session_factory=world.factory,
                              qdrant=lambda: (world.client, world.collections))

    spec.run(ctx, lambda *a, **k: None)  # the ingest has not recorded its shard: files stay for a re-drive
    assert (vectors / "vectors_0000.parquet").exists()

    (vectors / "ingest_done.txt").write_text("vectors_0000.parquet\n", encoding="utf-8")
    spec.run(ctx, lambda *a, **k: None)
    assert not task_dir.exists()
