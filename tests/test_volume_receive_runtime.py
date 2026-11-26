"""A conference slice received from a 115 volume, through a real OpenList node.

The writer (the GitHub Actions node) leaves ``<conf>/<year>/<paper>.pdf`` with a
``<paper>.provenance.json`` beside it in a Cfs volume; the download stage with
``options.download_from = "volume"`` copies them into the corpus tree through the
read-only user. The volume here is a Local storage on a real node, the files real
bytes; the reader is the real VolumeReader, only wrapped to record the order of its
reads.
"""
from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path

import pytest

from app.services.pipeline import stages as st
from app.services.pipeline.context import StageContext
from app.services.pipeline.store import TaskStore
from pipeline.openlist_reader import VolumeError, VolumeReader
from tests.openlist_node import start_node
from tests.sqlite_task_db import make_session_factory

VOLUME = "/cfs-tier1"
PDF = b"%PDF-1.4\n% fake body for a received paper\n%%EOF\n"


def paper(n: int) -> str:
    return f"vldb-2022-p{n}"


@pytest.fixture(scope="module")
def node(tmp_path_factory):
    node = start_node(tmp_path_factory)
    yield node
    node.stop()


@pytest.fixture(scope="module")
def reader_env(node):
    return {"OPENLIST_URL": node.url, "OPENLIST_READER_USER": "retrival-reader",
            "OPENLIST_READER_PASS_FILE": str(node.add_reader())}


@pytest.fixture()
def volume(node, tmp_path):
    """A fresh volume directory per test, mounted where the stage looks for /cfs-tier1."""
    mounts = {s["mount_path"]: s for s in node.api(
        "GET", "/api/admin/storage/list?page=1&per_page=100")["content"]}
    if VOLUME in mounts:
        node.api("POST", f"/api/admin/storage/delete?id={mounts[VOLUME]['id']}")
    root = node.mount_local(VOLUME, tmp_path / "volume")
    slice_dir = root / "vldb" / "2022"
    slice_dir.mkdir(parents=True)
    for n in (1, 2):
        (slice_dir / f"{paper(n)}.pdf").write_bytes(PDF + str(n).encode())
        (slice_dir / f"{paper(n)}.provenance.json").write_text(
            json.dumps({"paper_id": paper(n), "full_text_source": "volume-test"}), encoding="utf-8")
    (slice_dir / "stray-extra.pdf").write_bytes(PDF + b"extra")  # in the volume, not in the manifest
    return root


@pytest.fixture()
def store(tmp_path):
    return TaskStore(make_session_factory(tmp_path / "tasks.db"))


def make_ctx(tmp_path: Path, store, *, papers=(1, 2), **options) -> StageContext:
    corpus = tmp_path / "corpus"
    (corpus / "manifests").mkdir(parents=True, exist_ok=True)
    records = [{"paper_id": paper(n), "title": "t", "authors": [], "conference": "VLDB", "year": 2022,
                "pdf_url": "u", "paper_page_url": "p", "source_name": "s"} for n in papers]
    (corpus / "manifests" / "paper_manifest.json").write_text(json.dumps(records), encoding="utf-8")
    return StageContext(
        venue="VLDB", year=2022, store=store, task_id=store.create_task(task_type="ingest_pipeline"),
        corpus_root=corpus, project_root=tmp_path / "proj",
        options={"download_from": "volume", "volume": VOLUME, **options})


class Recording(VolumeReader):
    """The real reader that also notes what it fetched, in order."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.fetched: list[str] = []

    def fetch(self, path, dest, **kw):
        self.fetched.append(path.rsplit("/", 1)[1])
        return super().fetch(path, dest, **kw)


@pytest.fixture()
def reader(node, reader_env):
    return Recording(reader_env["OPENLIST_URL"], reader_env["OPENLIST_READER_USER"],
                     password_file=Path(reader_env["OPENLIST_READER_PASS_FILE"]))


def run_stage(spec, ctx):
    lines: list[str] = []
    spec.run(ctx, lambda message, level="info": lines.append(message))
    return lines


# ----------------------------------------------------------------------- options
def test_receive_options_accept_volume_only_for_the_download_stage(tmp_path):
    assert st.receive_mode({"download_from": "volume"}, "download_from") == "volume"
    assert st.receive_mode({"download_from": "push"}, "download_from") == "push"
    assert st.receive_mode({"download_from": "local"}, "download_from") is None
    with pytest.raises(ValueError, match="'local' or 'push'"):
        st.receive_mode({"extract_from": "volume"}, "extract_from")
    with pytest.raises(ValueError, match="'local' or 'push'"):
        st.receive_mode({"download_from": "ftp"}, "download_from")


def test_volume_receive_requires_the_volume_and_a_clean_prefix():
    ok = {"download_from": "volume", "volume": "/cfs-tier1"}
    st.validate_receive_options(ok, "local", conference=True)
    st.validate_receive_options({**ok, "volume_prefix": "vldb/2022"}, None, conference=True)
    st.validate_receive_options({"download_from": "volume"}, None, conference=True)  # options.volume is optional
    with pytest.raises(ValueError, match="options.volume"):
        st.validate_receive_options({"download_from": "volume", "volume": "cfs-tier1"}, None, conference=True)
    with pytest.raises(ValueError, match="volume_prefix"):
        st.validate_receive_options({**ok, "volume_prefix": "../etc"}, None, conference=True)
    with pytest.raises(ValueError, match="conference slices only"):
        st.validate_receive_options(ok, None, conference=False)
    with pytest.raises(ValueError, match="retired remote"):
        st.validate_receive_options(ok, "remote", conference=True)


# --------------------------------------------------------------------- the stage
def test_the_stage_lands_files_with_sidecars_and_counts_what_the_manifest_lacks(
        node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    ctx = make_ctx(tmp_path, store)
    spec = st.make_download_stage(ctx)  # the dispatcher picks the volume branch from the options
    assert spec.key == "download" and spec.resource_key == "net" and "volume" in spec.title
    assert spec.probe is st.download_probe  # the gate wraps download_gate with where the rest must come from

    lines = run_stage(spec, ctx)

    landed = sorted(p.name for p in ctx.pdf_dir.iterdir())
    assert landed == sorted([f"{paper(1)}.pdf", f"{paper(1)}.provenance.json", f"{paper(2)}.pdf",
                             f"{paper(2)}.provenance.json", "stray-extra.pdf"])
    assert (ctx.pdf_dir / f"{paper(2)}.pdf").read_bytes() == PDF + b"2"
    assert json.loads((ctx.pdf_dir / f"{paper(1)}.provenance.json").read_text())["paper_id"] == paper(1)
    assert any("1 not in the manifest slice" in line for line in lines), lines
    # The same probe and gate as a local download: 3 PDFs on disk against a slice of 2.
    assert st.download_probe(ctx).total == 2 and st.download_gate(ctx).ok


def test_sidecars_land_before_their_pdf_and_a_rerun_skips_what_is_there(
        node, volume, reader, store, tmp_path):
    ctx = make_ctx(tmp_path, store)
    sink_lines: list[str] = []
    first = st.receive_volume_files(
        reader, f"{VOLUME}/vldb/2022", ctx.pdf_dir, abort=ctx.abort, log=sink_lines.append)
    order = reader.fetched
    for n in (1, 2):
        assert order.index(f"{paper(n)}.provenance.json") < order.index(f"{paper(n)}.pdf")
    assert (first.fetched, first.skipped) == (5, 0)
    assert first.bytes == sum(p.stat().st_size for p in volume.rglob("*") if p.is_file())

    mtimes = {p.name: p.stat().st_mtime_ns for p in ctx.pdf_dir.iterdir()}
    reader.fetched.clear()
    second = st.receive_volume_files(reader, f"{VOLUME}/vldb/2022", ctx.pdf_dir, abort=ctx.abort, log=sink_lines.append)
    assert (second.fetched, second.skipped) == (0, 5) and reader.fetched == []
    assert {p.name: p.stat().st_mtime_ns for p in ctx.pdf_dir.iterdir()} == mtimes


def test_a_partial_local_file_is_fetched_again_and_no_part_files_remain(node, volume, reader, store, tmp_path):
    ctx = make_ctx(tmp_path, store)
    ctx.pdf_dir.mkdir(parents=True)
    (ctx.pdf_dir / f"{paper(1)}.pdf").write_bytes(b"%PDF-trunc")  # an interrupted earlier copy, wrong size
    result = st.receive_volume_files(reader, f"{VOLUME}/vldb/2022", ctx.pdf_dir, abort=ctx.abort, log=lambda m: None)
    assert (ctx.pdf_dir / f"{paper(1)}.pdf").read_bytes() == PDF + b"1"
    assert (result.fetched, result.skipped) == (5, 0)
    assert not list(ctx.pdf_dir.glob("*.part"))


def test_progress_is_logged_every_fifty_files(node, volume, reader, tmp_path):
    for n in range(3, 123):
        (volume / "vldb" / "2022" / f"{paper(n)}.pdf").write_bytes(PDF)
    lines: list[str] = []
    result = st.receive_volume_files(reader, f"{VOLUME}/vldb/2022", tmp_path / "pdf", abort=threading.Event(),
                                     log=lines.append)
    assert result.fetched == 125
    assert [line for line in lines if "received 50" in line or "received 100" in line], lines
    assert not [line for line in lines if "received 25 " in line]


def test_other_suffixes_and_nested_directories_keep_their_layout(node, volume, reader, tmp_path):
    medical = volume / "pdf" / "2024" / "lancet"
    medical.mkdir(parents=True)
    for suffix, body in ((".pdf", PDF), (".xml", b"<article/>"), (".md", b"# t")):
        (medical / f"pmc1{suffix}").write_bytes(body)
    (medical / "pmc1.txt").write_bytes(b"not wanted")
    result = st.receive_volume_files(
        reader, f"{VOLUME}/pdf", tmp_path / "med", suffixes=(".pdf", ".xml", ".md", ".provenance.json"),
        abort=threading.Event(), log=lambda m: None)
    assert sorted(p.relative_to(tmp_path / "med").as_posix() for p in (tmp_path / "med").rglob("*") if p.is_file()) \
        == ["2024/lancet/pmc1.md", "2024/lancet/pmc1.pdf", "2024/lancet/pmc1.xml"]
    assert result.fetched == 3


def test_a_cancelled_task_stops_between_files(node, volume, reader, tmp_path):
    abort = threading.Event()
    abort.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        st.receive_volume_files(reader, f"{VOLUME}/vldb/2022", tmp_path / "pdf", abort=abort, log=lambda m: None)
    assert not (tmp_path / "pdf").exists() or not any((tmp_path / "pdf").iterdir())


def test_a_slice_the_volume_does_not_hold_is_logged_and_left_to_the_gate(
        node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    ctx = make_ctx(tmp_path, store, volume_prefix="vldb/1999")
    spec = st.make_download_stage(ctx)
    lines = run_stage(spec, ctx)
    assert "volume /cfs-tier1/vldb/1999: no such directory, 0 files" in lines
    assert not spec.gate(ctx).ok  # 0/2 PDFs on disk


def test_the_gate_judges_a_short_volume_like_a_short_download(node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    ctx = make_ctx(tmp_path, store, papers=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))  # the volume holds 2 of 10
    spec = st.make_download_stage(ctx)
    run_stage(spec, ctx)
    verdict = spec.gate(ctx)
    # The gate counts the slice's own PDFs: the stray one the volume also holds is not a paper of the slice.
    assert not verdict.ok and "2/10" in verdict.reason


def test_the_default_prefix_is_the_lowercase_venue_and_year(node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    ctx = make_ctx(tmp_path, store)
    assert st.volume_source_dir(ctx) == f"{VOLUME}/vldb/2022"
    assert st.volume_source_dir(dataclasses.replace(ctx, options={**ctx.options, "volume": "/cfs-tier1/",
                                                                  "volume_prefix": "/other/dir/"})) \
        == f"{VOLUME}/other/dir"


# ------------------------------------------------- the volume is the default source
def test_a_conference_slice_receives_from_the_volume_unless_told_otherwise(
        node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    ctx = dataclasses.replace(make_ctx(tmp_path, store), options={})  # no download_from, no volume
    spec = st.make_download_stage(ctx)
    assert "volume" in spec.title and spec.resource_key == "net"
    assert st.volume_source_dir(ctx) == f"{VOLUME}/vldb/2022"  # DEFAULT_CONFERENCE_VOLUME/<conf lower>/<year>
    st.validate_receive_options({}, None, conference=True)

    run_stage(spec, ctx)

    assert (ctx.pdf_dir / f"{paper(1)}.pdf").read_bytes() == PDF + b"1"
    assert (ctx.pdf_dir / f"{paper(2)}.provenance.json").is_file()


def test_this_host_never_downloads_so_download_from_local_is_refused(store, tmp_path):
    options = {"download_from": "local"}
    with pytest.raises(ValueError, match="downloads run on cfscan into the 115 volume; this host only receives"):
        st.validate_receive_options(options, None, conference=True)
    # A task stored before the refusal (retry, restart) fails at the stage builder instead of downloading.
    ctx = dataclasses.replace(make_ctx(tmp_path, store), options=options)
    with pytest.raises(ValueError, match="downloads run on cfscan"):
        st.make_download_stage(ctx)


def test_a_malformed_volume_is_refused_whichever_way_the_volume_mode_was_chosen():
    with pytest.raises(ValueError, match="options.volume"):
        st.validate_receive_options({"volume": "cfs-tier1"}, None, conference=True)  # volume is the default mode
    with pytest.raises(ValueError, match="volume_prefix"):
        st.validate_receive_options({"volume_prefix": "../etc"}, None, conference=True)
    st.validate_receive_options({"download_from": "volume"}, None, conference=True)  # the default volume applies


def test_a_slice_directory_missing_on_the_volume_is_an_empty_receipt(node, volume, reader, tmp_path):
    lines: list[str] = []
    receipt = st.receive_volume_files(
        reader, f"{VOLUME}/vldb/1999", tmp_path / "pdf", abort=threading.Event(), log=lines.append)
    assert (receipt.fetched, receipt.skipped, receipt.bytes, receipt.pdfs) == (0, 0, 0, [])
    assert f"volume {VOLUME}/vldb/1999: no such directory, 0 files" in lines


def test_a_gate_failure_after_a_volume_receive_says_the_rest_must_reach_the_volume_first(
        node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    ctx = make_ctx(tmp_path, store, papers=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    spec = st.make_download_stage(ctx)
    run_stage(spec, ctx)
    verdict = spec.gate(ctx)
    assert not verdict.ok
    assert verdict.reason == ("only 2/10 PDFs downloaded (0 tried, 8 never tried); "
                              "the missing PDFs must be downloaded into the volume (cfscan) first")
    assert st.download_gate(ctx).reason == "only 2/10 PDFs downloaded (0 tried, 8 never tried)"  # the hint is the volume stage's


# ------------------------------- what cfscan tried and could not get travels with the PDFs
def put_attempts(volume, n: int, *, paper_id: str | None = None, attempts=None) -> None:
    directory = volume / "attempts" / "vldb" / "2022"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{paper(n)}.json").write_text(json.dumps({
        "paper_id": paper_id or paper(n), "conference": "VLDB", "year": 2022,
        "attempts": [{"at": "2026-11-16T01:00:00+00:00", "error": "HTTP 403"}] if attempts is None else attempts,
    }), encoding="utf-8")


def put_identity_rejection(volume, n: int) -> None:
    directory = volume / "rejected_identity" / "vldb" / "2022"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{paper(n)}.json").write_text(json.dumps({
        "paper_id": paper(n), "rejections": [{"url": "u", "full_text_source": "arxiv", "reason": "title"}]}),
        encoding="utf-8")
    (directory / f"{paper(n)}.pdf").write_bytes(PDF + b"rejected")


def test_the_stage_receives_attempt_and_identity_rejection_records_and_the_gate_lets_the_slice_go(
        node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    put_attempts(volume, 3)
    put_identity_rejection(volume, 4)
    ctx = make_ctx(tmp_path, store, papers=(1, 2, 3, 4))  # the volume holds 2 PDFs, a try for 3, a rejection for 4
    spec = st.make_download_stage(ctx)

    lines = run_stage(spec, ctx)

    attempts = ctx.corpus_root / "download_attempts" / "vldb" / "2022"
    rejected = ctx.corpus_root / "rejected_identity" / "vldb" / "2022"
    assert sorted(p.name for p in attempts.iterdir()) == [f"{paper(3)}.json"]
    assert json.loads((attempts / f"{paper(3)}.json").read_text())["attempts"][0]["error"] == "HTTP 403"
    assert sorted(p.name for p in rejected.iterdir()) == [f"{paper(4)}.json", f"{paper(4)}.pdf"]
    assert (rejected / f"{paper(4)}.pdf").read_bytes() == PDF + b"rejected"
    assert not (ctx.pdf_dir / f"{paper(4)}.pdf").exists()  # a rejected copy is not a download
    assert "1 download attempt records for VLDB 2022" in lines, lines
    assert "1 identity rejection records for VLDB 2022" in lines, lines
    verdict = spec.gate(ctx)
    assert verdict.ok
    assert verdict.reason == "2/4 PDFs; 2 still missing after download attempts (kept on the refill list)"


def test_one_paper_nobody_tried_still_stops_the_slice_after_the_records_are_received(
        node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    put_attempts(volume, 3)
    ctx = make_ctx(tmp_path, store, papers=(1, 2, 3, 4))  # 4 has neither a PDF nor a record
    spec = st.make_download_stage(ctx)
    run_stage(spec, ctx)
    verdict = spec.gate(ctx)
    assert not verdict.ok
    assert verdict.reason == ("only 2/4 PDFs downloaded (1 tried, 1 never tried); "
                              "the missing PDFs must be downloaded into the volume (cfscan) first")


def test_a_volume_without_record_directories_is_logged_and_receives_nothing_extra(
        node, volume, reader_env, store, tmp_path, monkeypatch):
    for key, value in reader_env.items():
        monkeypatch.setenv(key, value)
    ctx = make_ctx(tmp_path, store)
    lines = run_stage(st.make_download_stage(ctx), ctx)
    assert "volume /cfs-tier1/attempts/vldb/2022: no such directory, 0 files" in lines
    assert "volume /cfs-tier1/rejected_identity/vldb/2022: no such directory, 0 files" in lines
    assert "0 download attempt records for VLDB 2022" in lines
    assert not (ctx.corpus_root / "download_attempts").exists()


def test_a_stage_without_the_reader_configuration_says_which_variable(store, tmp_path, monkeypatch):
    for name in ("OPENLIST_READER_USER", "OPENLIST_READER_PASS_FILE"):
        monkeypatch.delenv(name, raising=False)
    ctx = make_ctx(tmp_path, store)
    with pytest.raises(VolumeError, match="OPENLIST_READER_USER"):
        run_stage(st.make_download_stage(ctx), ctx)
