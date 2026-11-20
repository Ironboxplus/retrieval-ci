"""Marker (torch / surya) is retired: every way into it raises RetiredInferencePath.

User decision 2026-11-14: "Marker 报错，Paddle 保留". The Marker worker, extractor and scripts stay in the
tree but raise the moment they are used; the PaddleOCR-VL worker registered through
PIPELINE_LOCAL_EXTRACTOR is the extractor. Nothing here loads a model: each call must raise before it
would, and the local extract stage must not fall back to anything when no extractor is registered.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from tests.reembed_support import new_store

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_extractor(monkeypatch):
    monkeypatch.delenv("PIPELINE_LOCAL_EXTRACTOR", raising=False)


def _blocked_heavy_imports(monkeypatch):
    for heavy in ("torch", "marker", "surya"):
        monkeypatch.setitem(sys.modules, heavy, None)               # loading a model would fail differently


def test_the_marker_extractor_raises_when_constructed(monkeypatch):
    from core.inference_retired import RetiredInferencePath
    from pipeline.paper_crawler.converters.extractors import MarkerExtractor

    _blocked_heavy_imports(monkeypatch)
    with pytest.raises(RetiredInferencePath, match="MarkerExtractor"):
        MarkerExtractor()
    with pytest.raises(RetiredInferencePath):
        MarkerExtractor(extra_config={"disable_ocr": True})


def test_the_extractor_factory_and_the_fallback_chain_cannot_build_marker():
    from core.inference_retired import RetiredInferencePath
    from pipeline.paper_crawler.converters.extractors import (
        FallbackExtractor, MarkerExtractor, PyMuPDFExtractor, create_extractor,
    )

    with pytest.raises(RetiredInferencePath):
        create_extractor("marker")
    with pytest.raises(RetiredInferencePath):                       # the chain is never built with a Marker primary
        FallbackExtractor(primary=MarkerExtractor(), fallback=PyMuPDFExtractor())


def test_the_extractor_that_stays_is_still_constructible():
    from pipeline.paper_crawler.converters.extractors import PyMuPDFExtractor, create_extractor

    assert isinstance(create_extractor("pymupdf"), PyMuPDFExtractor)


@pytest.mark.parametrize("module, needle", [
    ("scripts.remote_marker.marker_worker", "marker_worker"),
    ("scripts.upload.marker_extract_book", "marker_extract_book"),
    ("scripts.smoke.smoke_marker", "smoke_marker"),
])
def test_marker_scripts_raise_in_main_before_loading_anything(module, needle, monkeypatch):
    from core.inference_retired import RetiredInferencePath

    _blocked_heavy_imports(monkeypatch)
    monkeypatch.setattr(sys, "argv", [needle])                      # no valid arguments: it must raise before parsing
    main = importlib.import_module(module).main

    with pytest.raises(RetiredInferencePath, match=needle):
        main()


@pytest.mark.parametrize("script, arguments", [
    ("scripts/remote_marker/marker_worker.py", ["--input", "in", "--output", "out"]),
    ("scripts/upload/marker_extract_book.py", ["--pdf", "book.pdf", "--out", "book.md"]),
    ("scripts/smoke/smoke_marker.py", []),
])
def test_running_a_marker_script_exits_with_the_exception(script, arguments):
    done = subprocess.run([sys.executable, str(REPO / script), *arguments], cwd=REPO, capture_output=True,
                          text=True, timeout=180, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert done.returncode != 0
    assert "RetiredInferencePath" in done.stderr and "is retired" in done.stderr


# ----------------------------------------------------------- the local extract stage has no Marker default
@pytest.fixture()
def ctx(tmp_path):
    store, _ = new_store()
    task_id = store.create_task(task_type="ingest_pipeline")
    c = StageContext(venue="CVPR", year=2026, store=store, task_id=task_id, corpus_root=tmp_path / "corpus",
                     project_root=tmp_path / "proj")
    c.pdf_dir.mkdir(parents=True, exist_ok=True)
    (c.pdf_dir / "cvpr-2026-a.pdf").write_bytes(b"%PDF-1.4")
    c.state_set(gpu_ids="0")
    return c


def test_without_a_registered_extractor_the_local_command_is_refused_not_defaulted(ctx):
    from core.inference_retired import RetiredInferencePath

    with pytest.raises(RetiredInferencePath, match="PIPELINE_LOCAL_EXTRACTOR"):
        st.build_extract_local_cmd(ctx, gpu=0, rank=0, world_size=1)


def test_without_a_registered_extractor_the_floor_and_the_preflight_are_refused(ctx):
    from core.inference_retired import RetiredInferencePath

    with pytest.raises(RetiredInferencePath, match="PIPELINE_LOCAL_EXTRACTOR"):
        st.extract_local_min_free_mb()
    with pytest.raises(RetiredInferencePath, match="PIPELINE_LOCAL_EXTRACTOR"):
        st.extract_local_preflight(ctx)


def test_without_a_registered_extractor_the_stage_launches_nothing(ctx, monkeypatch):
    from core.inference_retired import RetiredInferencePath

    launched = []
    monkeypatch.setattr(st, "run_local", lambda *a, **k: launched.append(a))
    spec = st.make_extract_stage_local(ctx)                          # building the stage is harmless

    with pytest.raises(RetiredInferencePath, match="PIPELINE_LOCAL_EXTRACTOR"):
        spec.run(ctx, lambda *a, **k: None)
    assert launched == []


def test_the_stage_title_does_not_claim_marker(ctx):
    title = st.make_extract_stage_local(ctx).title
    assert "marker" not in title.lower()


def test_a_registered_extractor_still_builds_its_own_command(ctx, tmp_path, monkeypatch):
    import json

    block = tmp_path / "extractor.json"
    block.write_text(json.dumps({"python": "/venv/bin/python", "worker": "scripts/remote_pvl/pvl_worker.py",
                                 "args": ["--x", "1"], "extract_min_free_mb": 9216}), encoding="utf-8")
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACTOR", str(block))

    cmd = st.build_extract_local_cmd(ctx, gpu=1, rank=0, world_size=2)

    assert cmd[:2] == ["/venv/bin/python", "scripts/remote_pvl/pvl_worker.py"] and cmd[-2:] == ["--x", "1"]
    assert st.extract_local_min_free_mb() == 9216
    monkeypatch.setenv("PIPELINE_LOCAL_MIN_FREE_MB", "1")             # the registered figure is the only floor
    monkeypatch.setenv("PIPELINE_LOCAL_EXTRACT_MIN_FREE_MB", "2")
    assert st.extract_local_min_free_mb() == 9216


def test_an_uploaded_pdf_no_longer_goes_through_marker(tmp_path, monkeypatch):
    """scripts/upload/extract_one.py built MarkerExtractor first, then fell back to PyMuPDF when marker-pdf
    was missing; building it raises now, and the worker exits non-zero (the stage fails, loudly)."""
    from core.inference_retired import RetiredInferencePath

    for name in ("CUDA_DEVICE_ORDER", "TQDM_DISABLE"):
        monkeypatch.setenv(name, os.environ.get(name, "x"))          # the module's import sets them: restore afterwards
    extract_one = importlib.import_module("scripts.upload.extract_one")
    pdf = tmp_path / "upload.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with pytest.raises(RetiredInferencePath, match="MarkerExtractor"):
        extract_one._extract_markdown(pdf, 1)
