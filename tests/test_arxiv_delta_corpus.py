"""arXiv corpus for the daily update: delta key, id-month paths, id-list conversion, manifest.

Real SQLite files, the real conversion code and CLI, and a real LaTeXML page
(Mixtral, 2401.04088: the harvested copy when present, else one request through
the local first hop). Nothing is faked.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import socket
import sqlite3
import sys
import urllib.request
from pathlib import Path

import pytest

from pipeline.paper_crawler.arxiv_corpus import (
    DELTA_KEY,
    DELTA_MIN_ID_PREFIX,
    LEGACY_SLICE,
    SlicePaths,
    batch_of,
    convert_rows,
    convert_slice,
    is_batch_key,
    month_of,
    rows_by_ids,
    slice_label,
    slice_rows,
    slice_sql,
    write_manifest,
    write_slice_manifest,
)
from pipeline.paper_crawler.arxiv_html import ArxivRawRecord, HarvestStore, html_path
from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper

REPO = Path(__file__).resolve().parents[1]
MIXTRAL = "2401.04088"
LOCAL_HTML = REPO / "data" / "arxiv" / "cs_html" / "html" / "2401" / f"{MIXTRAL}.html.gz"


def _first_hop_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 10808), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def mixtral_html() -> bytes:
    if LOCAL_HTML.exists():
        return gzip.decompress(LOCAL_HTML.read_bytes())
    if not _first_hop_up():
        pytest.skip("no harvested copy and local first hop 10808 is down")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://127.0.0.1:10808"}))
    return opener.open(f"https://export.arxiv.org/html/{MIXTRAL}", timeout=120).read()


def _rec(arxiv_id: str, *, latest="v1", content="v1", withdrawn=False,
         date="2026-04-02T00:00:00+00:00") -> ArxivRawRecord:
    return ArxivRawRecord(
        arxiv_id=arxiv_id, datestamp="2026-04-03", categories="cs.LG", title=f"title {arxiv_id}",
        authors="A B and C D", abstract="abstract", comments=None, license=None,
        latest_version=latest, latest_version_date=date, first_version_date="2026-04-01T00:00:00+00:00",
        latest_size="0kb" if withdrawn else "10kb", content_version=content,
        content_version_date=date, withdrawn=withdrawn,
    )


# ------------------------------------------------------------------ keys and SQL


def test_delta_key_labels_and_pinned_behaviour_on_zero():
    assert DELTA_KEY == 0
    assert slice_label(DELTA_KEY) == "delta"
    # 0 is neither a batch nor a month of anything: month_of leaves it alone.
    assert is_batch_key(DELTA_KEY) is False
    assert month_of(DELTA_KEY) == 0
    # Existing keys keep their labels.
    assert slice_label(202409) == "2024-09"
    assert slice_label(LEGACY_SLICE) == "≤2023-11"
    assert slice_label(20240101) == "2024-01 #1"


def test_delta_sql_selects_2026_plus_new_style_ids(tmp_path):
    where, params = slice_sql(DELTA_KEY)
    assert params == [DELTA_MIN_ID_PREFIX] == ["2601"]
    db = sqlite3.connect(tmp_path / "ids.sqlite3")
    db.execute("CREATE TABLE papers(arxiv_id TEXT)")
    ids = ["cs/0101001", "2312.00001", "2512.99999", "2601.00001", "2603.12345", "2701.00002"]
    db.executemany("INSERT INTO papers VALUES(?)", [(i,) for i in ids])
    picked = [r[0] for r in db.execute(f"SELECT arxiv_id FROM papers WHERE {where} ORDER BY 1", params)]
    db.close()
    assert picked == ["2601.00001", "2603.12345", "2701.00002"]
    # A delta id still maps to its own month slice and batch.
    assert batch_of("2603.12345") == 20260303


# ------------------------------------------------------------------ paths


def test_markdown_and_doc_follow_the_id_month_for_every_key_kind(tmp_path):
    root = tmp_path
    for key, run_id in [(202401, None), (20240101, None), (LEGACY_SLICE, None), (DELTA_KEY, 7)]:
        paths = SlicePaths(root, key, run_id)
        assert paths.markdown("2401.04088") == root / "markdown" / "arxiv" / "202401" / "2401.04088.md"
        assert paths.doc("2401.04088") == root / "doc" / "202401" / "2401.04088.json.gz"
        assert paths.markdown("cs/0101001") == root / "markdown" / "arxiv" / "202311" / "cs-0101001.md"
        assert paths.markdown("2305.00001") == root / "markdown" / "arxiv" / "202311" / "2305.00001.md"
    delta = SlicePaths(root, DELTA_KEY, 7)
    assert delta.markdown("2604.00009").parent == root / "markdown" / "arxiv" / "202604"


def test_manifest_and_report_names_per_key_and_per_delta_run(tmp_path):
    month = SlicePaths(tmp_path, 202401)
    assert month.manifest == tmp_path / "manifests" / "arxiv_202401.json"
    assert month.report == tmp_path / "reports" / "convert_202401.json"
    assert (month.markdown_dir, month.doc_dir) == (
        tmp_path / "markdown" / "arxiv" / "202401", tmp_path / "doc" / "202401")
    batch = SlicePaths(tmp_path, 20240101)
    assert batch.manifest == tmp_path / "manifests" / "arxiv_20240101.json"
    assert batch.markdown_dir == month.markdown_dir  # a batch lives in its month

    a, b = SlicePaths(tmp_path, DELTA_KEY, 7), SlicePaths(tmp_path, DELTA_KEY, 8)
    assert a.manifest == tmp_path / "manifests" / "arxiv_delta_7.json"
    assert a.report == tmp_path / "reports" / "convert_delta_7.json"
    assert b.manifest != a.manifest and b.report != a.report


def test_run_id_belongs_to_delta_and_only_to_delta(tmp_path):
    with pytest.raises(ValueError):
        SlicePaths(tmp_path, DELTA_KEY)
    with pytest.raises(ValueError):
        SlicePaths(tmp_path, 202401, 3)
    with pytest.raises(ValueError, match="no single markdown directory"):
        SlicePaths(tmp_path, DELTA_KEY, 3).markdown_dir


# ------------------------------------------------------------------ rows


def test_rows_by_ids_batches_under_the_sqlite_variable_limit(tmp_path):
    store = HarvestStore(tmp_path / "h.sqlite3")
    ids = [f"2602.{n:05d}" for n in range(1, 2002)]  # 2001 ids -> 3 IN-batches
    store.upsert([_rec(i) for i in ids])
    store.upsert([_rec("2602.00001", latest="v3", content="v2", withdrawn=True)])
    store.db.execute("UPDATE papers SET status='ok', html_version=content_version WHERE arxiv_id <= '2602.01500'")
    store.db.commit()
    store.close()
    db = tmp_path / "h.sqlite3"

    rows = rows_by_ids(db, ids + ["2699.99999"])  # an unknown id is skipped
    assert [r.arxiv_id for r in rows] == ids  # sorted, none lost across batches
    assert len(rows_by_ids(db, ids, statuses=["ok"])) == 1500
    assert len(rows_by_ids(db, ids[:10] + ids[:10])) == 10  # duplicates collapse
    assert rows_by_ids(db, []) == []

    first = rows[0]
    assert (first.latest_version, first.content_version, first.withdrawn) == ("v3", "v2", True)
    assert rows[1].withdrawn is False and rows[1].content_version == "v1"


def test_old_rows_without_content_version_read_as_latest_version(tmp_path):
    store = HarvestStore(tmp_path / "h.sqlite3")
    store.upsert([_rec("2602.00001", latest="v2", content="v2")])
    store.db.execute("UPDATE papers SET content_version=NULL, content_version_date=NULL")
    store.db.commit()
    store.close()
    [row] = slice_rows(tmp_path / "h.sqlite3", 202602)
    assert (row.content_version, row.withdrawn) == ("v2", False)
    [by_delta] = slice_rows(tmp_path / "h.sqlite3", DELTA_KEY)
    assert by_delta == row


# ------------------------------------------------------------------ convert + manifest


def _delta_harvest(tmp_path: Path, html: bytes) -> tuple[Path, Path]:
    """A harvest root where 2401.04088 (outside the delta id range, chosen by
    explicit id), 2604.00009 (withdrawn after v1) and 2604.00010 (no html) exist."""
    hroot, croot = tmp_path / "harvest", tmp_path / "corpus"
    store = HarvestStore(hroot / "arxiv_cs.sqlite3")
    store.upsert([
        _rec(MIXTRAL, date="2024-01-08T18:47:34+00:00"),
        _rec("2604.00009", latest="v2", content="v1", withdrawn=True),
        _rec("2604.00010"),
    ])
    for arxiv_id in (MIXTRAL, "2604.00009"):
        store.mark(arxiv_id, "ok", html_version="v1", raw_bytes=len(html), gz_bytes=0)
        dest = html_path(hroot, arxiv_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.compress(html))
    store.mark("2604.00010", "no_html", error="HTTP 404")
    store.close()
    return hroot, croot


def test_convert_rows_writes_id_month_files_and_a_delta_manifest(tmp_path, mixtral_html):
    hroot, croot = _delta_harvest(tmp_path, mixtral_html)
    paths = SlicePaths(croot, DELTA_KEY, 5)
    rows = rows_by_ids(hroot / "arxiv_cs.sqlite3", [MIXTRAL, "2604.00009", "2604.00010"], statuses=["ok"])
    assert [r.arxiv_id for r in rows] == [MIXTRAL, "2604.00009"]

    report = convert_rows(rows, paths, hroot=hroot, workers=1, log=lambda _m: None)
    assert (report["downloaded"], report["markdown"], report["failures"]) == (2, 2, {})
    assert report["slice"] == DELTA_KEY
    # Each paper's files are in its own month, not under the delta key.
    assert (croot / "markdown" / "arxiv" / "202401" / f"{MIXTRAL}.md").read_text(encoding="utf-8").startswith(
        "# Mixtral of Experts")
    assert (croot / "markdown" / "arxiv" / "202604" / "2604.00009.md").exists()
    assert (croot / "doc" / "202401" / f"{MIXTRAL}.json.gz").exists()
    assert paths.report == croot / "reports" / "convert_delta_5.json" and paths.report.exists()

    write_manifest(rows, paths, report["failures"], log=lambda _m: None)
    records = json.loads(paths.manifest.read_text(encoding="utf-8"))
    by_id = {r["arxiv_id"]: r for r in records}
    assert paths.manifest == croot / "manifests" / "arxiv_delta_5.json"
    assert set(by_id) == {MIXTRAL, "2604.00009"}
    assert (by_id[MIXTRAL]["source_version"], by_id[MIXTRAL]["withdrawn"]) == ("v1", False)
    # A withdrawn paper is in the manifest with its last content version's Markdown.
    withdrawn = by_id["2604.00009"]
    assert (withdrawn["source_version"], withdrawn["withdrawn"]) == ("v1", True)
    assert withdrawn["markdown_path"] == str(croot / "markdown" / "arxiv" / "202604" / "2604.00009.md")
    assert withdrawn["paper_id"] == "arxiv-2604.00009" and withdrawn["track_type"] == "preprint"

    # A second run (new run id) does not touch the first one's manifest and report.
    other = SlicePaths(croot, DELTA_KEY, 6)
    report2 = convert_rows(rows, other, hroot=hroot, workers=1, log=lambda _m: None)
    assert report2["converted_now"] == 0  # outputs are current: nothing to redo
    assert paths.manifest.exists() and other.report.exists() and other.report != paths.report


def test_month_slice_still_converts_into_its_month(tmp_path, mixtral_html):
    hroot, croot = _delta_harvest(tmp_path, mixtral_html)
    report = convert_slice(202401, hroot=hroot, croot=croot, workers=1, log=lambda _m: None)
    assert (report["downloaded"], report["markdown"]) == (1, 1)
    assert (croot / "markdown" / "arxiv" / "202401" / f"{MIXTRAL}.md").exists()
    write_slice_manifest(202401, hroot=hroot, croot=croot, log=lambda _m: None)
    [record] = json.loads((croot / "manifests" / "arxiv_202401.json").read_text(encoding="utf-8"))
    assert (record["source_version"], record["withdrawn"]) == ("v1", False)


# ------------------------------------------------------------------ CLI


def _cli(args: list[str], hroot: Path, croot: Path, monkeypatch) -> int:
    """Run scripts/collect/arxiv_cs_corpus.py's ``main`` with these arguments; an
    argparse error surfaces as SystemExit(2). (The same script is run as a real
    subprocess by test_arxiv_corpus_runtime.)"""
    spec = importlib.util.spec_from_file_location("arxiv_cs_corpus_cli", REPO / "scripts" / "collect" / "arxiv_cs_corpus.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("ARXIV_HARVEST_ROOT", str(hroot))
    monkeypatch.setenv("ARXIV_CORPUS_ROOT", str(croot))
    monkeypatch.setattr(sys, "argv", ["arxiv_cs_corpus.py", *args])
    try:
        return cli.main()
    except SystemExit as exc:
        return int(exc.code)


def test_cli_converts_an_id_list_as_a_delta_run(tmp_path, mixtral_html, monkeypatch):
    hroot, croot = _delta_harvest(tmp_path, mixtral_html)
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps([MIXTRAL, "2604.00009"]), encoding="utf-8")

    assert _cli(["convert", "--ids-file", str(ids_file), "--run-id", "11", "--workers", "1"],
                hroot, croot, monkeypatch) == 0
    records = json.loads((croot / "manifests" / "arxiv_delta_11.json").read_text(encoding="utf-8"))
    assert {r["arxiv_id"] for r in records} == {MIXTRAL, "2604.00009"}

    # The frozen-selection file of a pipeline run is an object carrying the list.
    ids_file.write_text(json.dumps({"ids": ["2604.00009"], "counts": {"new": 1}}), encoding="utf-8")
    assert _cli(["manifest", "--ids-file", str(ids_file), "--run-id", "12"], hroot, croot, monkeypatch) == 0
    records = json.loads((croot / "manifests" / "arxiv_delta_12.json").read_text(encoding="utf-8"))
    assert [r["arxiv_id"] for r in records] == ["2604.00009"]
    assert records[0]["withdrawn"] is True


def test_cli_rejects_mixed_or_incomplete_scope(tmp_path, mixtral_html, monkeypatch):
    hroot, croot = _delta_harvest(tmp_path, mixtral_html)
    ids_file = tmp_path / "ids.json"
    ids_file.write_text("[]", encoding="utf-8")
    for args in (
        ["convert", "--slice", "202401", "--ids-file", str(ids_file), "--run-id", "1"],
        ["convert", "--ids-file", str(ids_file)],
        ["convert", "--run-id", "1"],
        ["convert"],
    ):
        assert _cli(args, hroot, croot, monkeypatch) == 2, args


# ------------------------------------------------------------------ CorpusPaper


def _conference_paper(**extra) -> CorpusPaper:
    return CorpusPaper(paper_id="acl-2024-1", title="t", authors=["x"], conference="ACL", year=2024,
                       pdf_url="u", paper_page_url="p", source_name="s", **extra)


def test_conference_manifest_records_do_not_gain_keys():
    assert set(_conference_paper().to_manifest_record()) == {
        "abstract", "alternate_pdf_urls", "authors", "conference", "download_error", "downloaded_from_url",
        "extract_error", "index_error", "indexed_doc_id", "markdown_path", "paper_id", "paper_page_url",
        "pdf_path", "pdf_resolution_note", "pdf_resolution_status", "pdf_url", "retraction_notice_url",
        "selection_rule", "source_cycle", "source_name", "title", "venue_track", "year",
    }


def test_source_keys_appear_only_when_set_and_round_trip():
    record = _conference_paper(source_version="v3", withdrawn=False).to_manifest_record()
    # withdrawn=False is a real value ("checked, not withdrawn"), distinct from "not an arXiv row".
    assert record["source_version"] == "v3" and record["withdrawn"] is False
    paper = CorpusPaper.from_manifest_record(record)
    assert (paper.source_version, paper.withdrawn) == ("v3", False)
    assert "source_version" not in _conference_paper().to_manifest_record()
    assert "withdrawn" not in _conference_paper().to_manifest_record()
