"""Pages with no usable article are remembered, not reselected on every run.

Real SQLite files (harvest database via HarvestStore, application tables), the real
converter and CLI as subprocesses, a real LaTeXML page (Mixtral) and a real non-LaTeXML
page. Nothing is faked.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from app.services.pipeline.arxiv_stages import (
    ArxivStageContext,
    arxiv_extract_probe,
    freeze_delta,
    make_arxiv_extract_stage,
)
from pipeline.paper_crawler.arxiv_corpus import (
    DELTA_KEY,
    SlicePaths,
    convert_rows,
    convert_slice,
    rejected_ids,
    rows_by_ids,
)
from pipeline.paper_crawler.arxiv_html import HarvestStore, html_path
from tests.sqlite_task_db import make_session_factory
from tests.test_arxiv_delta_corpus import _rec, mixtral_html  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
GOOD = "2604.00001"
NOT_LATEXML = b"<html><head><title>arXiv</title></head><body><p>Conversion to HTML had a Fatal error.</p></body></html>"
STARTED = "2026-01-01T00:00:00+00:00"


class Harvest:
    """A harvest root with rows whose HTML is already on disk (status ok)."""

    def __init__(self, tmp_path: Path):
        self.root, self.croot = tmp_path / "harvest", tmp_path / "corpus"
        self.db = self.root / "arxiv_cs.sqlite3"

    def add(self, arxiv_id: str, html: bytes | None, *, version="v1") -> None:
        store = HarvestStore(self.db)
        store.upsert([_rec(arxiv_id, latest=version, content=version)])
        store.mark(arxiv_id, "ok", html_version=version)
        store.close()
        self.write_html(arxiv_id, html)

    def write_html(self, arxiv_id: str, html: bytes | None) -> None:
        dest = html_path(self.root, arxiv_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.compress(html) if html is not None else b"not a gzip file")

    def new_version(self, arxiv_id: str, html: bytes, version: str) -> None:
        store = HarvestStore(self.db)
        store.upsert([_rec(arxiv_id, latest=version, content=version)])  # a new content version requeues it
        store.mark(arxiv_id, "ok", html_version=version)
        store.close()
        self.write_html(arxiv_id, html)

    def convert(self, ids: list[str], key: int = DELTA_KEY, run_id: int | None = 1) -> dict:
        paths = SlicePaths(self.croot, key, run_id if key == DELTA_KEY else None)
        lines: list[str] = []
        report = convert_rows(rows_by_ids(self.db, ids), paths, hroot=self.root, workers=1, log=lines.append)
        report["log"] = lines
        return report

    def outcome(self, arxiv_id: str) -> tuple:
        db = sqlite3.connect(self.db)
        try:
            return db.execute("SELECT convert_status, convert_version, convert_error FROM papers "
                              "WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        finally:
            db.close()


def freeze(h: Harvest, tmp_path: Path) -> dict:
    return freeze_delta(h.db, make_session_factory(tmp_path / "app.db"), started=STARTED)


def test_rejection_is_classified_persisted_and_not_converted_again(tmp_path, mixtral_html):
    h = Harvest(tmp_path)
    h.add(GOOD, mixtral_html)
    h.add("2604.00002", NOT_LATEXML)
    h.add("2604.00003", None)  # unreadable file: a crash, not a verdict on the page

    report = h.convert([GOOD, "2604.00002", "2604.00003"])

    assert h.outcome(GOOD) == ("ok", "v1", None)
    status, version, error = h.outcome("2604.00002")
    assert (status, version) == ("rejected", "v1") and "no LaTeXML article" in error
    status, version, error = h.outcome("2604.00003")
    assert (status, version) == ("error", "v1") and error
    assert report["rejected"] == ["2604.00002"] and report["errors"] == ["2604.00003"]
    assert (report["downloaded"], report["convertible"], report["markdown"]) == (3, 2, 1)
    assert set(report["failures"]) == {"2604.00002", "2604.00003"}

    # Again, as a re-driven task would: the verdict stands (nothing re-converted for it),
    # the crash is tried again.
    again = h.convert([GOOD, "2604.00002", "2604.00003"], run_id=2)
    assert "1 to convert" in again["log"][0] and "1 rejected earlier" in again["log"][0]
    assert again["rejected"] == ["2604.00002"] and again["errors"] == ["2604.00003"]
    assert rejected_ids(h.db, [GOOD, "2604.00002", "2604.00003"]) == {"2604.00002"}


def test_freeze_leaves_out_rejected_papers_until_a_new_version(tmp_path, mixtral_html):
    h = Harvest(tmp_path)
    h.add(GOOD, mixtral_html)
    h.add("2604.00002", NOT_LATEXML)
    first = freeze(h, tmp_path)
    assert first["ids"] == [GOOD, "2604.00002"] and first["convert_rejected"] == []
    h.convert(first["ids"])

    second = freeze(h, tmp_path)
    # The converted-and-indexed one is not in SQL here, so it is still new; the rejected one is not.
    assert second["ids"] == [GOOD]
    assert second["convert_rejected"] == ["2604.00002"]
    assert second["counts"]["selected"] == 1 and second["counts"]["convert_rejected"] == 1

    # arXiv publishes v2 (with a good page): selected again, converted, and the old verdict is replaced.
    h.new_version("2604.00002", mixtral_html, "v2")
    third = freeze(h, tmp_path)
    assert third["ids"] == [GOOD, "2604.00002"] and third["convert_rejected"] == []
    h.convert(third["ids"], run_id=3)
    assert h.outcome("2604.00002") == ("ok", "v2", None)
    assert rejected_ids(h.db, ["2604.00002"]) == set()


def test_a_new_version_that_is_rejected_again_is_remembered_for_that_version(tmp_path, mixtral_html):
    h = Harvest(tmp_path)
    h.add("2604.00002", NOT_LATEXML)
    h.convert(["2604.00002"])
    h.new_version("2604.00002", NOT_LATEXML, "v2")
    assert rejected_ids(h.db, ["2604.00002"]) == set()  # the v1 verdict does not carry over
    h.convert(["2604.00002"], run_id=2)
    assert h.outcome("2604.00002")[:2] == ("rejected", "v2")
    assert rejected_ids(h.db, ["2604.00002"]) == {"2604.00002"}


# ------------------------------------------------------------- the extract stage
def delta_ctx(tmp_path: Path, h: Harvest, ids: list[str], task_id: int = 9) -> ArxivStageContext:
    ctx = ArxivStageContext(venue="arXiv", year=DELTA_KEY, store=None, task_id=task_id, project_root=REPO,
                            corpus_root=h.croot, harvest_root=h.root, convert_workers=1)
    ctx.delta_file.parent.mkdir(parents=True, exist_ok=True)
    ctx.delta_file.write_text(json.dumps({"ids": ids, "counts": {"selected": len(ids)}}), encoding="utf-8")
    return ctx


def run_stage(ctx: ArxivStageContext) -> list[str]:
    lines: list[str] = []
    make_arxiv_extract_stage(ctx).run(ctx, lambda m, level="info": lines.append(m))
    return lines


def test_extract_gate_passes_with_rejections_also_when_the_task_is_redriven(tmp_path, mixtral_html):
    h = Harvest(tmp_path)
    ids = [GOOD, "2604.00002", "2604.00003", "2604.00004"]
    h.add(GOOD, mixtral_html)
    for rejected in ids[1:]:
        h.add(rejected, NOT_LATEXML)
    ctx = delta_ctx(tmp_path, h, ids)
    spec = make_arxiv_extract_stage(ctx)

    lines = run_stage(ctx)  # the converter exits 0: 1 of 1 convertible has Markdown

    assert any("3 rejected" in line for line in lines)
    probe = arxiv_extract_probe(ctx)
    assert (probe.done, probe.total) == (1, 1) and "3 rejected" in probe.message
    assert spec.gate(ctx).ok

    redriven = run_stage(ctx)  # the same frozen set after a failure further on
    assert any("3 rejected" in line for line in redriven)
    assert spec.gate(ctx).ok
    assert sorted(ctx.convert_report()["rejected"]) == ids[1:]


def test_extract_fails_when_crashes_exceed_five_percent(tmp_path, mixtral_html):
    h = Harvest(tmp_path)
    h.add(GOOD, mixtral_html)
    h.add("2604.00002", None)  # a crash, not a rejection
    ctx = delta_ctx(tmp_path, h, [GOOD, "2604.00002"])
    spec = make_arxiv_extract_stage(ctx)

    with pytest.raises(RuntimeError, match="command failed"):  # markdown ratio 0.5 < 0.95
        run_stage(ctx)

    probe = arxiv_extract_probe(ctx)
    assert (probe.done, probe.total) == (1, 2)
    gate = spec.gate(ctx)
    assert not gate.ok and "1/2" in gate.reason
    assert ctx.convert_report()["errors"] == ["2604.00002"]


def test_a_month_slice_converts_and_reports_the_same_way(tmp_path, mixtral_html):
    h = Harvest(tmp_path)
    h.add(GOOD, mixtral_html)
    h.add("2604.00002", NOT_LATEXML)
    report = convert_slice(202604, hroot=h.root, croot=h.croot, workers=1, log=lambda _m: None)
    assert (report["downloaded"], report["convertible"], report["markdown"]) == (2, 1, 1)
    assert report["rejected"] == ["2604.00002"] and report["errors"] == []
    assert h.outcome("2604.00002")[0] == "rejected"

    import subprocess

    cli = subprocess.run(
        [sys.executable, "scripts/collect/arxiv_cs_corpus.py", "convert", "--slice", "202604", "--workers", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "ARXIV_HARVEST_ROOT": str(h.root), "ARXIV_CORPUS_ROOT": str(h.croot)})
    assert cli.returncode == 0, cli.stdout[-600:] + cli.stderr[-600:]
