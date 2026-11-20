"""A conference slice produced on another host reaches the ingest host as a verified tar.

Real files, real tar archives, real sha256 checks, real subprocesses (the push
script and the preindex script), a real SQLite task table and data database, the
real orchestrator over the production stage factories. The one stand-in is the
network hop: no sshd is reachable from the test machine, so the InboxTransfer runs
tests/local_ssh_transport.py in place of the ssh and scp executables. It is spawned
with the same argument vector as the OpenSSH client and executes the same
mkdir / copy-as-.part / rename sequence in a real shell; only the connection
between two hosts is absent.
"""
from __future__ import annotations

import dataclasses
import io
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from app.services.pipeline import stages
from app.services.pipeline.context import StageContext
from app.services.pipeline.service import PipelineRunner
from core.db.models import Chunk as DBChunk, Paper as DBPaper
from pipeline.paper_crawler.inbox_transfer import InboxTransfer
from pipeline.paper_crawler.slice_bundle import (
    MANIFEST_MEMBER,
    RECEIPT_FILE,
    bundle_file_name,
    import_slice_bundle,
    push_slice_files,
    receive_slice_bundle,
    sha256_file,
    write_slice_bundle,
)
from tests.test_host_turn_download import tiny_pdf
from tests.test_pipeline_external_md_runtime import (
    CORPUS_NAME,
    MARKDOWN,
    REPO,
    _record,
    _task_store,
)

HELPER = Path(__file__).resolve().parent / "local_ssh_transport.py"
PAPERS = {"cvpr-2026-aa": "Predictive Placement", "cvpr-2026-bb": "Migrating Instead Of Killing"}


def _transport(inbox: Path) -> InboxTransfer:
    return InboxTransfer(
        ssh_target="ingest-host", remote_inbox=inbox.as_posix(),
        ssh=[sys.executable, str(HELPER), "ssh"], scp=[sys.executable, str(HELPER), "scp"])


def _produce(root: Path) -> Path:
    """What the producing host has for (CVPR, 2026): PDFs with their provenance
    sidecars and the Markdown the marker wrote."""
    pdf_dir = root / "pdf" / "cvpr" / "2026"
    md_dir = root / "markdown" / "cvpr" / "2026"
    pdf_dir.mkdir(parents=True)
    md_dir.mkdir(parents=True)
    for pid, title in PAPERS.items():
        (pdf_dir / f"{pid}.pdf").write_bytes(tiny_pdf())
        (pdf_dir / f"{pid}.provenance.json").write_text(json.dumps({
            "paper_id": pid, "downloaded_from_url": f"https://example.org/{pid}.pdf",
            "full_text_source": "publisher", "full_text_check": "ok"}), encoding="utf-8")
        (md_dir / f"{pid}.md").write_text(MARKDOWN.format(title=title), encoding="utf-8")
    return root


@pytest.fixture
def producer(tmp_path) -> Path:
    return _produce(tmp_path / "producer")


def _bundle(producer: Path, tmp_path: Path, kind: str = "pdf") -> Path:
    directory = producer / ("pdf" if kind == "pdf" else "markdown") / "cvpr" / "2026"
    out = tmp_path / "wire" / bundle_file_name("cvpr", 2026, kind)
    files = sorted(directory.glob("*.pdf" if kind == "pdf" else "*.md"))
    if kind == "pdf":
        files += sorted(directory.glob("*.provenance.json"))
    write_slice_bundle(files, out, venue="CVPR", year=2026, kind=kind, source="producer-box")
    return out


# ------------------------------------------------------------ bundle: land + verify
def test_bundle_lands_pdfs_and_sidecars_with_a_receipt(producer, tmp_path):
    bundle = _bundle(producer, tmp_path)
    dest = tmp_path / "ingest" / "pdf" / "cvpr" / "2026"

    counts = import_slice_bundle(bundle, dest, venue="CVPR", year=2026, kind="pdf")

    assert counts["files"] == 2 and counts["sidecars"] == 2
    src = producer / "pdf" / "cvpr" / "2026"
    for pid in PAPERS:
        assert (dest / f"{pid}.pdf").read_bytes() == (src / f"{pid}.pdf").read_bytes()
        assert json.loads((dest / f"{pid}.provenance.json").read_text(encoding="utf-8"))["full_text_source"] == "publisher"
    receipts = json.loads((dest / RECEIPT_FILE).read_text(encoding="utf-8"))
    one = receipts["cvpr-2026-aa.pdf"]
    assert one["sha256"] == sha256_file(src / "cvpr-2026-aa.pdf")
    assert one["bundle"] == bundle.name and one["bundle_sha256"] == sha256_file(bundle)
    assert one["source"] == "producer-box"
    assert not [p for p in dest.parent.iterdir() if p.name.startswith(".incoming")]
    assert len(list(dest.glob("*.pdf"))) == 2  # the receipt and sidecars are not PDFs

    # The same bundle again changes nothing.
    assert import_slice_bundle(bundle, dest, venue="CVPR", year=2026, kind="pdf") == counts
    assert len(list(dest.glob("*.pdf"))) == 2


def _rewrite(bundle: Path, mutate) -> Path:
    """A copy of ``bundle`` whose members/manifest were altered by ``mutate(manifest, members)``."""
    with tarfile.open(bundle) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    manifest = json.loads(members.pop(MANIFEST_MEMBER))
    mutate(manifest, members)
    out = bundle.with_name("tampered.tar")
    with tarfile.open(out, "w") as tar:
        for name, payload in [(MANIFEST_MEMBER, json.dumps(manifest).encode()), *members.items()]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return out


def _flip_pdf_byte(manifest, members):
    name = "files/cvpr-2026-aa.pdf"
    data = bytearray(members[name])
    data[-3] ^= 0xFF  # still a PDF-looking file, but not the bytes that were hashed
    members[name] = bytes(data)


def _extra_member(manifest, members):
    members["files/unlisted.pdf"] = tiny_pdf()


def _missing_member(manifest, members):
    del members["files/cvpr-2026-bb.pdf"]


def _traversal(manifest, members):
    manifest["files"].append({"name": "../escape.pdf", "sha256": "0" * 64, "size": 1})


def _another_slice(manifest, members):
    manifest["year"] = 2025


def _not_a_pdf(manifest, members):
    members["files/cvpr-2026-bb.pdf"] = b"<html>an error page</html>"
    entry = next(e for e in manifest["files"] if e["name"] == "cvpr-2026-bb.pdf")
    import hashlib
    entry["sha256"] = hashlib.sha256(members["files/cvpr-2026-bb.pdf"]).hexdigest()
    entry["size"] = len(members["files/cvpr-2026-bb.pdf"])


@pytest.mark.parametrize("mutate,reason", [
    (_flip_pdf_byte, "sha256"),
    (_extra_member, "not in the manifest"),
    (_missing_member, "lacks"),
    (_traversal, "unsafe"),
    (_another_slice, "expected CVPR 2026 pdf"),
    (_not_a_pdf, "not a readable PDF"),
])
def test_a_bundle_that_does_not_verify_lands_nothing_and_is_set_aside(producer, tmp_path, mutate, reason):
    bundle = _rewrite(_bundle(producer, tmp_path), mutate)
    inbox = tmp_path / "ingest" / "inbox"
    inbox.mkdir(parents=True)
    pushed = inbox / bundle_file_name("cvpr", 2026, "pdf")
    shutil.copyfile(bundle, pushed)
    dest = tmp_path / "ingest" / "pdf" / "cvpr" / "2026"

    with pytest.raises(RuntimeError, match=rf"rejected.*{reason}"):
        receive_slice_bundle(pushed, dest, venue="CVPR", year=2026, kind="pdf",
                             abort=threading.Event(), log=lambda m, *a: None, poll_s=0.01)

    assert not list(dest.iterdir())  # not one file landed, not even the valid ones
    assert not [p for p in dest.parent.iterdir() if p.name.startswith(".incoming")]
    assert not pushed.exists() and (inbox / (pushed.name + ".rejected")).is_file()


def test_receive_waits_for_the_push_and_stops_when_aborted(producer, tmp_path):
    inbox = tmp_path / "ingest" / "inbox"
    bundle_path = inbox / bundle_file_name("cvpr", 2026, "pdf")
    dest = tmp_path / "ingest" / "pdf" / "cvpr" / "2026"
    logs: list[str] = []

    def push_later():
        time.sleep(0.4)
        push_slice_files(producer / "pdf" / "cvpr" / "2026", tmp_path / "outbox", _transport(inbox),
                         venue="CVPR", year=2026, kind="pdf", conf_lower="cvpr")

    pusher = threading.Thread(target=push_later)
    pusher.start()
    counts = receive_slice_bundle(bundle_path, dest, venue="CVPR", year=2026, kind="pdf",
                                  abort=threading.Event(), log=lambda m, *a: logs.append(m), poll_s=0.05)
    pusher.join()
    assert counts["files"] == 2 and any(line.startswith("waiting for") for line in logs)
    assert not list((tmp_path / "outbox").glob("*"))  # the pushed copy is not left behind
    assert not list(inbox.glob("*.part"))

    abort = threading.Event()
    threading.Timer(0.3, abort.set).start()
    with pytest.raises(RuntimeError, match="cancelled while waiting"):
        receive_slice_bundle(inbox / "nothing_2026_pdf.tar", dest, venue="CVPR", year=2026, kind="pdf",
                             abort=abort, log=lambda m, *a: None, poll_s=0.05)


def test_push_script_copies_the_bundle_into_the_inbox_through_a_subprocess(producer, tmp_path):
    inbox = tmp_path / "ingest" / "inbox"

    done = subprocess.run(
        [sys.executable, "scripts/sync/push_slice_bundle.py", "--conf", "CVPR", "--year", "2026",
         "--kind", "md", "--corpus-root", str(producer), "--ssh-target", "ingest-host",
         "--remote-inbox", inbox.as_posix(),
         "--ssh-command", sys.executable, str(HELPER), "ssh",
         "--scp-command", sys.executable, str(HELPER), "scp"],
        cwd=REPO, capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["files"] == 2
    assert sorted(p.name for p in inbox.iterdir()) == [bundle_file_name("cvpr", 2026, "md")]
    assert not list((producer / "outbox").glob("*"))
    with tarfile.open(inbox / bundle_file_name("cvpr", 2026, "md")) as tar:
        assert sorted(tar.getnames()) == sorted(
            [MANIFEST_MEMBER, "files/cvpr-2026-aa.md", "files/cvpr-2026-bb.md"])

    missing = subprocess.run(
        [sys.executable, "scripts/sync/push_slice_bundle.py", "--conf", "CVPR", "--year", "2030",
         "--kind", "md", "--corpus-root", str(producer), "--ssh-target", "x", "--remote-inbox", "/nowhere"],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert missing.returncode == 2 and "no md directory" in missing.stderr


def test_arxiv_batch_pusher_still_hands_batches_over_the_shared_transfer(tmp_path):
    """BatchPusher now sits on InboxTransfer: one resolved batch and the progress file arrive."""
    from pipeline.paper_crawler.arxiv_batches import BatchPusher, PROGRESS_FILE, bundle_name, import_bundle
    from pipeline.paper_crawler.arxiv_html import HarvestStore
    from tests.test_arxiv_batches_runtime import _record as arxiv_record

    source = tmp_path / "source"
    store = HarvestStore(source / "arxiv_cs.sqlite3")
    store.upsert([arxiv_record("2401.07999"), arxiv_record("2401.00001")])
    store.mark("2401.07999", "no_html", error="HTTP 404")  # batch #1 resolved; #0 still pending
    store.close()
    inbox = tmp_path / "ingest" / "inbox"
    logs: list[str] = []
    transport = _transport(inbox)
    pusher = BatchPusher(source / "arxiv_cs.sqlite3", source, ssh_target="ingest-host",
                         remote_inbox=inbox.as_posix(), scope=None, max_attempts=3, log=logs.append,
                         ssh=transport.ssh, scp=transport.scp)

    assert pusher.round_sync() == 1, logs
    assert sorted(p.name for p in inbox.iterdir()) == sorted([bundle_name(20240101), PROGRESS_FILE])
    assert json.loads((inbox / PROGRESS_FILE).read_text(encoding="utf-8"))["batches"] == {
        "20240100": {"pending": 1}, "20240101": {"no_html": 1}}
    assert pusher.round_sync() == 0  # a pushed batch is not pushed again
    got = HarvestStore(tmp_path / "ingest" / "arxiv_cs.sqlite3")
    assert import_bundle(got, tmp_path / "ingest", inbox / bundle_name(20240101)) == {"rows": 1, "html": 0}
    got.close()


# ------------------------------------------------------------------ the options
def test_options_choose_the_receive_stages_and_bad_values_are_refused(tmp_path):
    base = StageContext(venue="CVPR", year=2026, store=None, task_id=0, corpus_root=tmp_path / "corpus")
    push = dataclasses.replace(base, options={"download_from": "push", "extract_from": "push"})

    download, extract = stages.make_download_stage(push), stages.make_extract_stage_local(push)
    assert (download.key, download.resource_key, download.preflight) == ("download", "", None)
    assert (extract.key, extract.resource_key, extract.preflight) == ("extract", "", None)  # no GPU needed
    assert "pushed" in download.title and "pushed" in extract.title
    assert stages.receive_bundle_path(push, "pdf") == tmp_path / "corpus" / "inbox" / "cvpr_2026_pdf.tar"
    # Default and explicit "local": the production stages, GPU lock and all.
    assert stages.make_download_stage(base).resource_key == "net"
    local = stages.make_extract_stage_local(dataclasses.replace(base, options={"extract_from": "local"}))
    assert local.resource_key == "gpu-local" and local.preflight is not None

    with pytest.raises(ValueError, match="'local' or 'push'"):
        stages.validate_receive_options({"download_from": "ftp"}, None, conference=True)
    with pytest.raises(ValueError, match="conference slices only"):
        stages.validate_receive_options({"extract_from": "push"}, None, conference=False)
    with pytest.raises(ValueError, match="remote execution mode"):
        stages.validate_receive_options({"extract_from": "push"}, "remote", conference=True)
    stages.validate_receive_options({"extract_from": "push"}, "local", conference=True)
    stages.validate_receive_options(None, "remote", conference=False)


def test_start_ingest_refuses_unusable_receive_options_before_a_task_exists(tmp_path):
    store = _task_store(tmp_path / "tasks.db")
    runner = PipelineRunner(store=store)
    for venue, mode, options in [("CVPR", None, {"download_from": "ftp"}),
                                 ("arXiv", None, {"download_from": "push"}),
                                 ("CVPR", "remote", {"extract_from": "push"})]:
        with pytest.raises(ValueError):
            runner.start_ingest(venue, 202609 if venue == "arXiv" else 2026,
                                execution_mode=mode, options=options, ssh_target="x")
    assert store.list_running("ingest_pipeline") == []


# --------------------------------------------------- the whole thing, to preindex
@pytest.fixture
def ingest_site(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "db").mkdir(parents=True)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    corpus = data_dir / "papers" / CORPUS_NAME
    (corpus / "manifests").mkdir(parents=True)
    (corpus / "manifests" / "paper_manifest.json").write_text(
        json.dumps([_record(pid, title) for pid, title in PAPERS.items()]), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-c", "from core.db.database import engine, Base; import core.db.models; "
                               "Base.metadata.create_all(engine)"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    from contextlib import contextmanager

    engine = create_engine(f"sqlite:///{data_dir / 'db' / 'retrieval.db'}", connect_args={"check_same_thread": False})
    local_session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def data_factory():
        session = local_session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    store = _task_store(tmp_path / "tasks.db")

    def build(c):
        return [stages.make_download_stage(c), stages.make_extract_stage_local(c),
                stages.make_preindex_stage(c, session_factory=data_factory)]

    def context_factory(**kw):
        return dataclasses.replace(PipelineRunner._default_ctx(**kw), corpus_root=corpus)

    runner = PipelineRunner(store=store, pipeline_builder=build, context_factory=context_factory)
    yield runner, store, data_factory, corpus
    engine.dispose()


def _push(producer: Path, kind: str, inbox: Path) -> None:
    done = subprocess.run(
        [sys.executable, "scripts/sync/push_slice_bundle.py", "--conf", "CVPR", "--year", "2026",
         "--kind", kind, "--corpus-root", str(producer), "--ssh-target", "ingest-host",
         "--remote-inbox", inbox.as_posix(),
         "--ssh-command", sys.executable, str(HELPER), "ssh",
         "--scp-command", sys.executable, str(HELPER), "scp"],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def _wait(predicate, what: str, timeout: float = 240.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def test_pushed_pdf_and_markdown_slice_runs_through_preindex(ingest_site, producer):
    runner, store, data_factory, corpus = ingest_site
    inbox = corpus / "inbox"

    started = runner.start_ingest(
        "CVPR", 2026, ssh_target="unused",
        options={"download_from": "push", "extract_from": "push", "receive_poll_seconds": 0.1})
    task_id = started["task_id"]

    # The task is waiting at the download stage; nothing has landed.
    _wait(lambda: (store.get_subtask(task_id, "download") or {}).get("status") == "running", "download running")
    assert not (corpus / "pdf").exists() and store.get_task(task_id)["status"] == "running"

    _push(producer, "pdf", inbox)
    _wait(lambda: (store.get_subtask(task_id, "download") or {}).get("status") == "completed", "download done")
    assert len(list((corpus / "pdf" / "cvpr" / "2026").glob("*.pdf"))) == 2
    assert (store.get_subtask(task_id, "extract") or {}).get("status") != "completed"

    _push(producer, "md", inbox)
    _wait(lambda: store.get_task(task_id)["status"] in ("completed", "failed", "cancelled"), "task end")
    task = store.get_task(task_id)
    assert task["status"] == "completed", [store.get_subtask(task_id, k) for k in ("download", "extract", "preindex")]

    for key in ("download", "extract", "preindex"):
        assert store.get_subtask(task_id, key)["status"] == "completed"
    src_pdf = producer / "pdf" / "cvpr" / "2026"
    for pid in PAPERS:
        landed = corpus / "pdf" / "cvpr" / "2026" / f"{pid}.pdf"
        assert landed.read_bytes() == (src_pdf / f"{pid}.pdf").read_bytes()
        assert (corpus / "markdown" / "cvpr" / "2026" / f"{pid}.md").is_file()
    assert (corpus / "pdf" / "cvpr" / "2026" / RECEIPT_FILE).is_file()

    with data_factory() as db:
        assert {r[0] for r in db.query(DBPaper.paper_id)} == set(PAPERS)
        assert db.query(func.count(DBChunk.id)).scalar() > len(PAPERS)
    manifest = json.loads((corpus / "manifests" / "paper_manifest.json").read_text(encoding="utf-8"))
    for record in manifest:
        assert record["pdf_path"].endswith(f"{record['paper_id']}.pdf")
        assert record["markdown_path"].endswith(f"{record['paper_id']}.md")
        # The sidecar travelled with the PDF and the slice sync folded it into the record.
        assert record["full_text_source"] == "publisher"
        assert record["downloaded_from_url"] == f"https://example.org/{record['paper_id']}.pdf"
