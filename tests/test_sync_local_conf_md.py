"""TDD: sync_local_conf_md.py — scoped manifest backfill for ONE (conf, year)
from the LOCAL filesystem (W106).

The key invariant: it MUST mutate ONLY the target (conf, year) slice. The full
sync_manifest_pdf_md_paths.py NULLs ~89K unrelated Windows paths on the 4090
box; this slice-scoped tool must leave every other record byte-identical.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync" / "sync_local_conf_md.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_local_conf_md", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sync_local_conf_md"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _rec(conf, year, pid, *, pdf_path=None, markdown_path=None):
    return {
        "paper_id": pid, "conference": conf, "year": year,
        "pdf_path": pdf_path, "markdown_path": markdown_path,
    }


def _write_pdf(corpus, conf, year, pid):
    d = corpus / "pdf" / conf.lower() / str(year)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{mod.sanitize_file_component(pid)}.pdf"
    p.write_bytes(b"%PDF-1.4")
    return p


def _write_md(corpus, conf, year, pid):
    d = corpus / "markdown" / conf.lower() / str(year)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{mod.sanitize_file_component(pid)}.md"
    p.write_text("# Title\nbody", encoding="utf-8")
    return p


def test_sync_slice_fills_only_target_slice(tmp_path):
    corpus = tmp_path / "corpus"
    # target slice: CVPR 2026 (2 records); other slices must be untouched
    data = [
        _rec("CVPR", 2026, "cvpr-2026-a"),
        _rec("CVPR", 2026, "cvpr-2026-b"),
        _rec("CVPR", 2025, "cvpr-2025-x", pdf_path="C:/old/win/path.pdf"),  # other year
        _rec("AAAI", 2026, "aaai-2026-y", markdown_path="//win/share/y.md"),  # other venue
    ]
    _write_pdf(corpus, "CVPR", 2026, "cvpr-2026-a")
    _write_md(corpus, "CVPR", 2026, "cvpr-2026-a")
    _write_pdf(corpus, "CVPR", 2026, "cvpr-2026-b")
    # also lay down files for the OTHER slices — they must NOT be picked up
    _write_pdf(corpus, "CVPR", 2025, "cvpr-2025-x")
    _write_md(corpus, "AAAI", 2026, "aaai-2026-y")

    counts = mod.sync_slice(data, "CVPR", 2026, corpus_root=corpus)

    # target slice filled (md dir candidate casing may differ on case-insensitive
    # FS — assert via lowercased path so Windows CVPR==cvpr doesn't fail spuriously)
    assert data[0]["pdf_path"] == (corpus / "pdf" / "cvpr" / "2026" / "cvpr-2026-a.pdf").as_posix()
    assert data[0]["markdown_path"].lower() == (corpus / "markdown" / "cvpr" / "2026" / "cvpr-2026-a.md").as_posix().lower()
    assert data[1]["pdf_path"] == (corpus / "pdf" / "cvpr" / "2026" / "cvpr-2026-b.pdf").as_posix()
    assert data[1]["markdown_path"] is None  # no md file → left as-is, NOT filled

    # OTHER slices byte-identical — the Windows paths are NOT nulled
    assert data[2]["pdf_path"] == "C:/old/win/path.pdf"
    assert data[3]["markdown_path"] == "//win/share/y.md"

    assert counts["slice_records"] == 2
    assert counts["pdf_filled"] == 2 and counts["md_filled"] == 1


def test_sync_slice_never_nulls_missing_in_slice(tmp_path):
    """A target-slice record with no local file keeps its (possibly stale) path —
    never NULLed (purely additive within the slice)."""
    corpus = tmp_path / "corpus"
    data = [_rec("CVPR", 2026, "cvpr-2026-a", pdf_path="C:/stale.pdf", markdown_path="C:/stale.md")]
    counts = mod.sync_slice(data, "CVPR", 2026, corpus_root=corpus)
    assert data[0]["pdf_path"] == "C:/stale.pdf"
    assert data[0]["markdown_path"] == "C:/stale.md"
    assert counts["pdf_filled"] == 0 and counts["md_filled"] == 0
    assert counts["pdf_missing"] == 1 and counts["md_missing"] == 1


def test_sync_slice_case_insensitive_conf_match(tmp_path):
    """conference stored as 'CVPR' but --conf 'cvpr' (lowercased) still matches."""
    corpus = tmp_path / "corpus"
    data = [_rec("CVPR", 2026, "cvpr-2026-a")]
    _write_md(corpus, "CVPR", 2026, "cvpr-2026-a")
    counts = mod.sync_slice(data, "cvpr", 2026, corpus_root=corpus)
    assert counts["slice_records"] == 1 and counts["md_filled"] == 1


def test_sync_slice_year_must_match(tmp_path):
    corpus = tmp_path / "corpus"
    data = [_rec("CVPR", 2025, "x")]
    counts = mod.sync_slice(data, "CVPR", 2026, corpus_root=corpus)
    assert counts["slice_records"] == 0


# --- CLI: the artifact root is not always the canonical corpus root ----------
# `corpus_root/pdf` is a symlink to an SMB share that is currently unreachable
# ("Permission denied"), so the 2026 KDD/SIGIR download and marker runs wrote to
# data/papers/{pdf,markdown}/<conf>/<year> instead.  With the root hardcoded the
# backfill silently reports md_missing for every paper and writes nothing, which
# then makes preindex select 0 papers (it filters on manifest markdown_path).

def test_sync_slice_stores_the_real_on_disk_casing(tmp_path):
    """The stored path must survive a move to a case-SENSITIVE host.

    Candidates are tried original-case first, and on Windows `Path.exists()` is
    case-insensitive — so a record whose `conference` is "KDD" matched a directory
    actually named `kdd` and the manifest recorded ".../KDD/...". Shipping that
    manifest to Linux `.88` makes every path miss, and preindex (which filters on
    markdown_path) then selects 0 papers with no error anywhere.
    """
    corpus = tmp_path / "corpus"
    lower_dir = corpus / "markdown" / "kdd" / "2026"
    lower_dir.mkdir(parents=True)
    (lower_dir / "kdd-2026-a.md").write_text("# t", encoding="utf-8")

    data = [_rec("KDD", 2026, "kdd-2026-a")]
    counts = mod.sync_slice(data, "kdd", 2026, corpus_root=corpus)

    assert counts["md_filled"] == 1
    stored = data[0]["markdown_path"]
    assert "/kdd/" in stored, stored
    assert "/KDD/" not in stored, stored


def _run_cli(mod_, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sync_local_conf_md.py", *argv])
    return mod_.main()


def test_cli_corpus_root_override_finds_artifacts_outside_the_canonical_root(
    tmp_path, monkeypatch, capsys
):
    import json

    artifacts = tmp_path / "data" / "papers"
    _write_md(artifacts, "KDD", 2026, "kdd-2026-1")
    _write_pdf(artifacts, "KDD", 2026, "kdd-2026-1")

    manifest = tmp_path / "paper_manifest.json"
    manifest.write_text(json.dumps([
        _rec("KDD", 2026, "kdd-2026-1"),
        _rec("AAAI", 2022, "aaai-2022-z", markdown_path="//unreachable/share/z.md"),
    ]), encoding="utf-8")

    rc = _run_cli(mod, [
        "--conf", "kdd", "--year", "2026",
        "--manifest-path", str(manifest),
        "--corpus-root", str(artifacts),
    ], monkeypatch)
    assert rc == 0

    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written[0]["markdown_path"].lower() == (
        artifacts / "markdown" / "kdd" / "2026" / "kdd-2026-1.md").as_posix().lower()
    # An unrelated venue pointing at an unreachable share is never touched.
    assert written[1]["markdown_path"] == "//unreachable/share/z.md"


def test_cli_dry_run_reports_counts_without_writing(tmp_path, monkeypatch, capsys):
    import json

    artifacts = tmp_path / "data" / "papers"
    _write_md(artifacts, "KDD", 2026, "kdd-2026-1")

    manifest = tmp_path / "paper_manifest.json"
    original = json.dumps([_rec("KDD", 2026, "kdd-2026-1")])
    manifest.write_text(original, encoding="utf-8")

    rc = _run_cli(mod, [
        "--conf", "kdd", "--year", "2026",
        "--manifest-path", str(manifest),
        "--corpus-root", str(artifacts),
        "--dry-run",
    ], monkeypatch)
    assert rc == 0

    # The manifest is byte-identical and no backup was produced.
    assert manifest.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak*"))
    out = capsys.readouterr().out
    assert "md(filled=1" in out  # it still REPORTS what it would have filled
    assert "dry-run" in out
