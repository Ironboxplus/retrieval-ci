"""TDD: registry — assemble the ordered ingest pipeline from stage factories.

build_ingest_pipeline(ctx) must return the 6 stages in canonical order, each a
StageSpec, generic for ANY (venue, year). No real I/O — just construction.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.database import Base
from core.db import models  # noqa: F401
from app.services.pipeline.spec import StageSpec
from app.services.pipeline.store import TaskStore
from app.services.pipeline.context import StageContext
from app.services.pipeline.registry import build_ingest_pipeline, STAGE_KEYS
from app.services.pipeline import stages as st


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        s = Local()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return TaskStore(factory)


@pytest.fixture
def ctx(store, tmp_path):
    task_id = store.create_task(task_type="ingest_pipeline")
    return StageContext(venue="AAAI", year=2026, store=store, task_id=task_id, corpus_root=tmp_path)


def test_stage_keys_constant():
    assert STAGE_KEYS == ["download", "extract", "preindex", "embed", "ingest", "cleanup"]


def test_build_returns_six_specs_in_order(ctx):
    specs = build_ingest_pipeline(ctx)
    assert len(specs) == 6
    assert [s.key for s in specs] == STAGE_KEYS
    for s in specs:
        assert isinstance(s, StageSpec)


def test_each_stage_has_callbacks(ctx):
    specs = build_ingest_pipeline(ctx)
    by_key = {s.key: s for s in specs}
    # run + probe always present
    for s in specs:
        assert callable(s.run)
        assert callable(s.probe)
    # extract & embed have GPU preflight; ingest has Qdrant preflight
    assert by_key["extract"].preflight is not None
    assert by_key["embed"].preflight is not None
    assert by_key["ingest"].preflight is not None
    # download has no preflight
    assert by_key["download"].preflight is None
    # every stage has a gate
    for s in specs:
        assert s.gate is not None


def test_pipeline_is_generic_for_any_venue(store, tmp_path):
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = StageContext(venue="USENIX Security", year=2025, store=store, task_id=task_id, corpus_root=tmp_path)
    specs = build_ingest_pipeline(ctx)
    assert [s.key for s in specs] == STAGE_KEYS


# ============================================= W106 execution_mode branch
def test_default_mode_is_local_extract_embed(ctx):
    """No execution_mode → local extract/embed. The retired kss/SMB remote
    topology runs only when a caller asks for it by name."""
    by_key = {s.key: s for s in build_ingest_pipeline(ctx)}
    assert by_key["extract"].resource_key == "gpu-local"
    assert by_key["embed"].resource_key == "embed-service"   # a client of the vLLM service, no card


def test_remote_mode_still_selectable_by_name(store, tmp_path):
    task_id = store.create_task(task_type="ingest_pipeline")
    remote = StageContext(venue="CVPR", year=2026, store=store, task_id=task_id,
                          corpus_root=tmp_path, execution_mode="remote")
    by_key = {s.key: s for s in build_ingest_pipeline(remote)}
    assert by_key["extract"].resource_key == f"gpu-{remote.ssh_target}"
    assert by_key["embed"].resource_key == f"gpu-{remote.ssh_target}"


def test_local_mode_uses_local_extract_embed(store, tmp_path):
    task_id = store.create_task(task_type="ingest_pipeline")
    ctx = StageContext(
        venue="CVPR", year=2026, store=store, task_id=task_id,
        corpus_root=tmp_path, execution_mode="local",
    )
    specs = build_ingest_pipeline(ctx)
    # same canonical order + keys
    assert [s.key for s in specs] == STAGE_KEYS
    by_key = {s.key: s for s in specs}
    # extract + embed swapped to the local variants: extract holds the card, embed the service
    assert by_key["extract"].resource_key == "gpu-local"
    assert by_key["embed"].resource_key == "embed-service"
    assert by_key["extract"].preflight is st.extract_local_preflight
    # embed's own checks sit behind the nothing-to-embed short-circuit (stages.embeds_only_new_chunks)
    assert by_key["embed"].preflight is not None
    # download / preindex / ingest / cleanup unchanged (local in both modes)
    assert by_key["download"].resource_key == "net"
    assert by_key["preindex"].resource_key == "sql"
    assert by_key["ingest"].resource_key == "qdrant"
    assert by_key["cleanup"].resource_key == "qdrant"


def test_local_vs_remote_pick_different_makers(store, tmp_path):
    """Sanity: remote ctx → remote extract preflight; local ctx → local one."""
    tid_r = store.create_task(task_type="ingest_pipeline")
    remote = StageContext(venue="CVPR", year=2026, store=store, task_id=tid_r, corpus_root=tmp_path,
                          execution_mode="remote")
    tid_l = store.create_task(task_type="ingest_pipeline")
    local = StageContext(venue="CVPR", year=2026, store=store, task_id=tid_l,
                         corpus_root=tmp_path, execution_mode="local")
    r = {s.key: s for s in build_ingest_pipeline(remote)}
    l = {s.key: s for s in build_ingest_pipeline(local)}
    assert r["extract"].preflight is st.extract_preflight
    assert l["extract"].preflight is st.extract_local_preflight
