"""CodeAnalysisStore: what changed since the watermark, and the standing duplicate pairs.

Real SQLite through the real catalog repository; no Qdrant needed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.services.code_kb.analysis_store import CodeAnalysisStore
from core.code_kb.duplicate_rollup import ChunkRef, DuplicateChunkPair, DuplicateFilePair
from core.db.models import IndexCatalog
from tests.catalog_sqlite import CatalogDb

OWNER = "owner"
PROJECT = "/repo"
T0 = datetime(2026, 5, 1, 12, 0, 0)


@pytest.fixture()
def db():
    return CatalogDb()


@pytest.fixture()
def store(db):
    return CodeAnalysisStore(session_factory=db.session)


def _row(db, path, *, project=PROJECT, changed=None, deleted=None, kind="code", owner=OWNER):
    with db.session() as session:
        session.add(IndexCatalog(
            owner_user_id=owner, host="h", project=project, root_dir=project, rel_path=path,
            kind=kind, chunk_count=1, content_changed_at=changed, deleted_at=deleted))


def _pair(a, b, shared=2, score=0.97):
    chunk = ChunkRef(point_id=1, path=a, start_line=1, end_line=10, label="f")
    other = ChunkRef(point_id=2, path=b, start_line=11, end_line=20, label="g")
    return DuplicateFilePair(
        path_a=a, path_b=b, shared_chunks=shared, max_score=score, mean_score=score,
        examples=(DuplicateChunkPair(a=chunk, b=other, score=score),))


# ------------------------------------------------------------------ changed files

def test_first_analysis_covers_the_live_files_with_a_recorded_change(db, store):
    _row(db, "a.py", changed=T0)
    _row(db, "b.py", changed=T0 + timedelta(minutes=1))
    _row(db, "old.py", changed=None)  # indexed before the column existed
    _row(db, "gone.py", changed=T0, deleted=T0)
    changes = store.changed_files(OWNER, [PROJECT], since=None)
    assert sorted(changes.changed) == ["a.py", "b.py"]
    # No baseline means no stored pairs: a removal has nothing to clean up.
    assert changes.removed == ()
    assert changes.newest == T0 + timedelta(minutes=1)


def test_only_changes_after_the_watermark_are_returned(db, store):
    _row(db, "before.py", changed=T0)
    _row(db, "after.py", changed=T0 + timedelta(minutes=5))
    _row(db, "deleted.py", changed=T0, deleted=T0 + timedelta(minutes=6))
    _row(db, "long-gone.py", changed=T0, deleted=T0 - timedelta(days=30))
    changes = store.changed_files(OWNER, [PROJECT], since=T0)
    assert changes.changed == ("after.py",)
    assert changes.removed == ("deleted.py",)
    assert changes.newest == T0 + timedelta(minutes=6)


def test_a_file_changed_and_then_deleted_is_only_removed(db, store):
    _row(db, "x.py", changed=T0 + timedelta(minutes=1), deleted=T0 + timedelta(minutes=2))
    changes = store.changed_files(OWNER, [PROJECT], since=T0)
    assert changes.changed == () and changes.removed == ("x.py",)


def test_nothing_after_the_watermark_is_no_change(db, store):
    _row(db, "a.py", changed=T0)
    changes = store.changed_files(OWNER, [PROJECT], since=T0)
    assert changes.changed == () and changes.removed == () and changes.newest is None


def test_scope_spellings_owner_and_kind_are_respected(db, store):
    _row(db, "a.py", project="E:/work/repo", changed=T0)
    _row(db, "b.py", project="E:/Work/Repo", changed=T0)  # legacy spelling of the same scope
    _row(db, "other-scope.py", project="E:/work/other", changed=T0)
    _row(db, "other-owner.py", project="E:/work/repo", changed=T0, owner="someone-else")
    _row(db, "notes.md", project="E:/work/repo", changed=T0, kind="doc")
    changes = store.changed_files(OWNER, ["E:/work/repo", "E:/Work/Repo"], since=None)
    assert sorted(changes.changed) == ["a.py", "b.py"]


# ------------------------------------------------------------------ pairs and state

def test_applying_a_run_replaces_every_pair_that_mentions_a_touched_file(store):
    store.apply_run(OWNER, PROJECT, touched_paths=["a.py", "b.py", "c.py"],
                    pairs=[_pair("a.py", "b.py"), _pair("a.py", "c.py"), _pair("b.py", "c.py")],
                    watermark=T0, summary=None, now=T0)
    assert {(p.path_a, p.path_b) for p in store.all_pairs(OWNER, PROJECT)} == {
        ("a.py", "b.py"), ("a.py", "c.py"), ("b.py", "c.py")}

    # c.py changed: its pairs are recomputed, the pair that does not mention it stays.
    store.apply_run(OWNER, PROJECT, touched_paths=["c.py"], pairs=[_pair("a.py", "c.py", 5)],
                    watermark=T0 + timedelta(minutes=1), summary=None,
                    now=T0 + timedelta(minutes=1))
    pairs = {(p.path_a, p.path_b): p for p in store.all_pairs(OWNER, PROJECT)}
    assert set(pairs) == {("a.py", "b.py"), ("a.py", "c.py")}
    assert pairs[("a.py", "c.py")].shared_chunks == 5


def test_a_pair_that_survives_keeps_its_first_seen_time(store):
    store.apply_run(OWNER, PROJECT, touched_paths=["a.py"], pairs=[_pair("a.py", "b.py")],
                    watermark=T0, summary=None, now=T0)
    store.apply_run(OWNER, PROJECT, touched_paths=["a.py"], pairs=[_pair("a.py", "b.py", 3)],
                    watermark=T0, summary=None, now=T0 + timedelta(days=1))
    (pair,) = store.all_pairs(OWNER, PROJECT)
    assert pair.first_seen_at == T0 and pair.shared_chunks == 3


def test_pairs_touching_finds_either_side_and_ignores_other_scopes(store):
    store.apply_run(OWNER, PROJECT, touched_paths=["a.py", "z.py"],
                    pairs=[_pair("a.py", "b.py"), _pair("c.py", "z.py")],
                    watermark=T0, summary=None, now=T0)
    store.apply_run(OWNER, "/other", touched_paths=["a.py"], pairs=[_pair("a.py", "q.py")],
                    watermark=T0, summary=None, now=T0)
    assert set(store.pairs_touching(OWNER, PROJECT, ["b.py"])) == {("a.py", "b.py")}
    assert set(store.pairs_touching(OWNER, PROJECT, ["c.py", "a.py"])) == {
        ("a.py", "b.py"), ("c.py", "z.py")}
    assert store.pairs_touching("someone-else", PROJECT, ["a.py"]) == {}


def test_many_touched_paths_fit_in_one_call(store):
    paths = [f"f{i}.py" for i in range(2500)]
    store.apply_run(OWNER, PROJECT, touched_paths=paths[:3],
                    pairs=[_pair("f0.py", "f1.py")], watermark=T0, summary=None, now=T0)
    assert set(store.pairs_touching(OWNER, PROJECT, paths)) == {("f0.py", "f1.py")}
    store.apply_run(OWNER, PROJECT, touched_paths=paths, pairs=[], watermark=T0,
                    summary=None, now=T0)
    assert store.all_pairs(OWNER, PROJECT) == []


def test_a_run_with_a_summary_bumps_the_version_and_one_without_keeps_it(store):
    assert store.get_state(OWNER, PROJECT) is None
    first = store.apply_run(OWNER, PROJECT, touched_paths=["a.py"], pairs=[],
                            watermark=T0, summary={"text": "found 1"}, now=T0)
    assert first == 1
    quiet = store.apply_run(OWNER, PROJECT, touched_paths=["a.py"], pairs=[],
                            watermark=T0 + timedelta(minutes=1), summary=None,
                            now=T0 + timedelta(minutes=1))
    assert quiet == 1
    state = store.get_state(OWNER, PROJECT)
    assert state.version == 1 and state.delivered_version == 0
    assert state.summary == {"text": "found 1"}
    assert state.watermark == T0 + timedelta(minutes=1)
    assert state.last_run_at == T0 + timedelta(minutes=1)
    assert store.apply_run(OWNER, PROJECT, touched_paths=[], pairs=[], watermark=T0,
                           summary={"text": "found 2"}, now=T0) == 2


def test_the_watermark_never_moves_backwards(store):
    store.apply_run(OWNER, PROJECT, touched_paths=[], pairs=[],
                    watermark=T0 + timedelta(hours=1), summary=None, now=T0)
    store.apply_run(OWNER, PROJECT, touched_paths=[], pairs=[], watermark=T0,
                    summary=None, now=T0)
    assert store.get_state(OWNER, PROJECT).watermark == T0 + timedelta(hours=1)


def test_examples_survive_the_round_trip(store):
    store.apply_run(OWNER, PROJECT, touched_paths=["a.py"], pairs=[_pair("a.py", "b.py")],
                    watermark=T0, summary=None, now=T0)
    (pair,) = store.all_pairs(OWNER, PROJECT)
    assert pair.examples == [{
        "a": {"path": "a.py", "start_line": 1, "end_line": 10, "label": "f"},
        "b": {"path": "b.py", "start_line": 11, "end_line": 20, "label": "g"},
        "score": 0.97}]


def test_deleting_a_scope_clears_its_analysis(db, store):
    store.apply_run(OWNER, PROJECT, touched_paths=["a.py"], pairs=[_pair("a.py", "b.py")],
                    watermark=T0, summary={"text": "x"}, now=T0)
    store.apply_run(OWNER, "/other", touched_paths=["a.py"], pairs=[_pair("a.py", "b.py")],
                    watermark=T0, summary={"text": "x"}, now=T0)
    db.service.delete_project(owner_user_id=OWNER, project=PROJECT, kind="code")
    assert store.get_state(OWNER, PROJECT) is None
    assert store.all_pairs(OWNER, PROJECT) == []
    assert store.get_state(OWNER, "/other") is not None
    assert len(store.all_pairs(OWNER, "/other")) == 1
