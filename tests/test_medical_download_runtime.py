"""Black-box tests for the durable, proxy-only medical PDF download runtime.

These tests intentionally use a real ``sys.executable`` child that hosts both a
PDF origin and a forwarding HTTP proxy.  No HTTP client, downloader, proxy or
filesystem operation is mocked: the test proves httpx sends the request through
the configured proxy, bytes are streamed to disk, and the manifest checkpoint
survives a fresh process-like queue instance.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
import threading
import time
import shutil
from pathlib import Path

import pytest
import httpx
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from filelock import FileLock

from pipeline.medical_corpus.models import MedicalPaperRecord
from pipeline.medical_corpus.pdf_fetch import (
    DownloadCancelled,
    DownloadDiskGuardStopped,
    DownloadRateLimiter,
    download_pdf,
)
from pipeline.medical_corpus.prepare import PrepareConfig, run_download_pass
from pipeline.medical_corpus.proxy import MedicalProxyConfig


_SERVER = r'''
import http.server, json, pathlib, socketserver, sys, threading, urllib.parse, urllib.request

root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
log = root / "proxy-requests.jsonl"
payload_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4_375_000

def valid_pdf_payload(target_size):
    # Build a syntactically valid one-page PDF with a large stream.  The test
    # must exercise the same %PDF/parser/hash resume validation as production,
    # rather than relying on a header-only fake that resume rightly rejects.
    filler = b"x" * max(0, target_size - 512)
    while True:
        header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        objects = [
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] /Contents 4 0 R >>\nendobj\n",
            b"4 0 obj\n<< /Length " + str(len(filler)).encode() + b" >>\nstream\n" + filler + b"\nendstream\nendobj\n",
        ]
        offsets, cursor = [], len(header)
        for obj in objects:
            offsets.append(cursor)
            cursor += len(obj)
        xref = b"xref\n0 5\n0000000000 65535 f \n" + b"".join(
            f"{offset:010d} 00000 n \n".encode() for offset in offsets
        )
        trailer = b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n" + str(cursor).encode() + b"\n%%EOF\n"
        payload = header + b"".join(objects) + xref + trailer
        delta = target_size - len(payload)
        if delta == 0:
            return payload
        filler = filler + (b"x" * delta) if delta > 0 else filler[:delta]

payload = valid_pdf_payload(payload_size)
log_lock = threading.Lock()

def write_log(row):
    # Origin and proxy are separate real server threads.  Serialize only the
    # test telemetry append so every JSONL record remains an atomic evidence
    # row; transfer I/O is still fully concurrent and unmocked.
    with log_lock, log.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")

class Origin(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/malformed.pdf":
            body = b"%PDF-1.7\n" + (b"not-a-real-pdf\n" * 128)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            write_log({"kind": "origin_chunk", "bytes": len(body), "ts": __import__("time").monotonic()})
            return
        if self.path != "/paper.pdf":
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        for offset in range(0, len(payload), 65536):
            body = payload[offset:offset + 65536]
            self.wfile.write(body)
            write_log({"kind": "origin_chunk", "bytes": len(body), "ts": __import__("time").monotonic()})
    def log_message(self, *args): pass

class Proxy(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        target = self.path
        if not target.startswith(("http://", "https://")):
            self.send_error(400, "proxy requires absolute-form target"); return
        with urllib.request.urlopen(target, timeout=30) as upstream:
            self.send_response(upstream.status)
            self.send_header("Content-Type", upstream.headers.get_content_type())
            self.send_header("Content-Length", upstream.headers.get("Content-Length", ""))
            self.send_header("X-Test-Proxy", "used")
            self.end_headers()
            while True:
                body = upstream.read(65536)
                if not body: break
                self.wfile.write(body)
                write_log({"kind": "chunk", "target": target, "bytes": len(body), "ts": __import__("time").monotonic()})
        write_log({"kind": "complete", "target": target})
    def log_message(self, *args): pass

origin = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Origin)
proxy = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Proxy)
threading.Thread(target=origin.serve_forever, daemon=True).start()
threading.Thread(target=proxy.serve_forever, daemon=True).start()
print(json.dumps({"origin": origin.server_address[1], "proxy": proxy.server_address[1]}), flush=True)
threading.Event().wait()
'''


@contextmanager
def _running_proxy_origin(tmp_path: Path, payload_bytes: int = 4_375_000):
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", textwrap.dedent(_SERVER), str(tmp_path), str(payload_bytes)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready = proc.stdout.readline().strip()
        assert ready, proc.stderr.read() if proc.stderr else "server did not start"
        ports = json.loads(ready)
        yield {
            "url": f"http://127.0.0.1:{ports['origin']}/paper.pdf",
            "proxy": f"http://127.0.0.1:{ports['proxy']}",
            "log": tmp_path / "proxy-requests.jsonl",
        }
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture()
def local_proxy_origin(tmp_path: Path):
    with _running_proxy_origin(tmp_path) as server:
        yield server


def _record(url: str, paper_id: str = "med-pmc-runtime") -> MedicalPaperRecord:
    return MedicalPaperRecord(
        paper_id=paper_id,
        title="real proxy test",
        year=2026,
        journal="Test Journal",
        pdf_url=url,
        has_pdf=True,
        is_open_access=True,
    )


def test_streamed_download_uses_real_proxy_rate_limit_and_reports_progress(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    progress: list[int] = []
    started = time.monotonic()
    out = download_pdf(
        _record(str(local_proxy_origin["url"])),
        pdf_root=tmp_path / "pdf",
        proxy=MedicalProxyConfig(
            http_proxy=str(local_proxy_origin["proxy"]),
            https_proxy=str(local_proxy_origin["proxy"]),
        ),
        execute=True,
        allow_cf_bypass=False,
        max_bytes_per_second=2_000_000,
        progress_callback=lambda _paper, current, _total: progress.append(current),
    )
    elapsed = time.monotonic() - started

    assert out.download_status == "downloaded"
    assert out.pdf_path is not None
    artifact = Path(out.pdf_path)
    assert artifact.read_bytes().startswith(b"%PDF")
    assert out.pdf_bytes == artifact.stat().st_size
    assert out.pdf_sha256 and len(out.pdf_sha256) == 64
    assert progress and progress[-1] == out.pdf_bytes
    expected_minimum = max(0.0, out.pdf_bytes / 2_000_000)
    assert elapsed >= expected_minimum * 0.90
    requests = [json.loads(line) for line in Path(local_proxy_origin["log"]).read_text().splitlines()]
    assert requests[-1] == {"kind": "complete", "target": str(local_proxy_origin["url"])}
    assert sum(row["bytes"] for row in requests if row["kind"] == "chunk") == out.pdf_bytes


def test_main_corpus_download_refuses_an_empty_proxy_before_network(tmp_path: Path) -> None:
    import asyncio

    from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper, MainConferenceCorpusBuilder

    paper = CorpusPaper(
        paper_id="acl-2026-long-1",
        title="Proxy Required",
        authors=[],
        conference="ACL",
        year=2026,
        pdf_url="http://127.0.0.1:1/must-not-connect.pdf",
        paper_page_url="http://example.invalid/1",
        source_name="test",
    )
    builder = MainConferenceCorpusBuilder(
        corpus_root=tmp_path / "corpus",
        proxy_url="",
        years=(2026,),
        conferences=(),
    )

    with pytest.raises(RuntimeError, match="explicit HTTP proxy"):
        asyncio.run(builder.download_papers([paper], max_bytes_per_second=1_000_000))


def test_main_corpus_download_fails_fast_when_proxy_listener_is_unreachable(tmp_path: Path) -> None:
    import asyncio

    from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper, MainConferenceCorpusBuilder

    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        unused_port = unused.getsockname()[1]
    paper = CorpusPaper(
        paper_id="acl-2026-long-1",
        title="Proxy Listener Required",
        authors=[],
        conference="ACL",
        year=2026,
        pdf_url="http://example.invalid/must-not-connect.pdf",
        paper_page_url="http://example.invalid/1",
        source_name="test",
    )
    builder = MainConferenceCorpusBuilder(
        corpus_root=tmp_path / "corpus",
        proxy_url=f"http://127.0.0.1:{unused_port}",
        years=(2026,),
        conferences=(),
    )

    with pytest.raises(RuntimeError, match="proxy is unreachable"):
        asyncio.run(builder.download_papers([paper], max_bytes_per_second=1_000_000))


def test_real_proxy_malformed_pdf_never_becomes_downloaded_or_renamed(tmp_path: Path, local_proxy_origin: dict[str, object]) -> None:
    """A %PDF prefix alone is never a completion fact; PdfReader must accept it first."""
    malformed_url = str(local_proxy_origin["url"]).replace("/paper.pdf", "/malformed.pdf")
    out = download_pdf(
        _record(malformed_url, "med-malformed"),
        pdf_root=tmp_path / "pdf",
        proxy=MedicalProxyConfig(
            http_proxy=str(local_proxy_origin["proxy"]),
            https_proxy=str(local_proxy_origin["proxy"]),
        ),
        execute=True,
        allow_cf_bypass=False,
        max_retries=1,
    )

    target = tmp_path / "pdf" / "2026" / "Test_Journal" / "med-malformed.pdf"
    assert out.download_status == "failed"
    assert out.pdf_path is None
    assert "parser" in (out.download_error or "").lower()
    assert not target.exists()
    assert not list(target.parent.glob("*.write_in_progress"))


def test_two_real_pdfs_hold_2mbps_across_first_read_and_2_10_30s_windows(tmp_path: Path) -> None:
    """Measure actual downloader receipt timestamps with real origin/proxy traces.

    Two 33 MB PDFs keep the real flow alive beyond thirty seconds, so the
    assertions cover the first-read burst, rolling 2/10/30s windows and the
    full aggregate.  Origin and proxy write timestamps are separately captured
    to prove the path is truly streaming.  They cannot define client bandwidth
    on loopback: TCP can accept data into a kernel queue before application
    reads it.  The downloader's progress callback fires only after each real
    received chunk is written/digested, and is therefore the correct transfer
    clock for this application-level 2 MB/s contract.
    """
    rate = 2_000_000
    payload_bytes = 33_000_000
    with _running_proxy_origin(tmp_path, payload_bytes) as server:
        cfg = PrepareConfig(
            data_root=tmp_path / "medical",
            execute_download=True,
            proxy=MedicalProxyConfig(http_proxy=str(server["proxy"]), https_proxy=str(server["proxy"])),
            max_bytes_per_second=rate,
        )
        cfg.manifest_path.parent.mkdir(parents=True)
        cfg.manifest_path.write_text(
            "\n".join(json.dumps(_record(str(server["url"]), f"med-pmc-rate-{index}").to_dict()) for index in (1, 2)) + "\n",
            encoding="utf-8",
        )
        client_chunks: list[dict[str, float | int]] = []
        previous_by_paper: dict[str, int] = {}

        def on_progress(record: MedicalPaperRecord, current: int, _total: int | None) -> None:
            previous = previous_by_paper.get(record.paper_id, 0)
            client_chunks.append({"bytes": current - previous, "ts": time.monotonic()})
            previous_by_paper[record.paper_id] = current

        started = time.monotonic()
        summary = run_download_pass(cfg, progress_callback=on_progress)
        ended = time.monotonic()

    rows = [json.loads(line) for line in Path(server["log"]).read_text(encoding="utf-8").splitlines()]
    proxy_chunks = [row for row in rows if row["kind"] == "chunk"]
    origin_chunks = [row for row in rows if row["kind"] == "origin_chunk"]
    assert summary["downloaded"] == 2
    assert len([row for row in rows if row["kind"] == "complete"]) == 2
    assert sum(row["bytes"] for row in origin_chunks) == payload_bytes * 2
    assert sum(row["bytes"] for row in proxy_chunks) == payload_bytes * 2
    assert sum(int(row["bytes"]) for row in client_chunks) == payload_bytes * 2
    assert client_chunks[0]["ts"] - started >= (64 * 1024 / rate) * 0.75
    assert client_chunks[-1]["ts"] - client_chunks[0]["ts"] >= 30.0

    # A rolling window contains whole 64 KiB writes.  Two chunks is the
    # maximum dispatch tolerance (131,072 bytes); it is deliberately absolute,
    # not a loose percentage that grows with the window length.
    tolerance = 2 * 64 * 1024
    for window_s in (2.0, 10.0, 30.0):
        max_window_bytes = max(
            sum(int(row["bytes"]) for row in client_chunks if start <= row["ts"] < start + window_s)
            for start in (row["ts"] for row in client_chunks)
        )
        assert max_window_bytes <= rate * window_s + tolerance
    assert (payload_bytes * 2) / (ended - started) <= rate * 1.01


def test_atomic_manifest_checkpoint_makes_second_real_pass_resume_without_redownload(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    cfg = PrepareConfig(
        data_root=tmp_path / "medical",
        execute_download=True,
        download_limit=1,
        proxy=MedicalProxyConfig(
            http_proxy=str(local_proxy_origin["proxy"]),
            https_proxy=str(local_proxy_origin["proxy"]),
        ),
    )
    records = [_record(str(local_proxy_origin["url"]), f"med-pmc-resume-{n}") for n in (1, 2)]
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(
        "\n".join(json.dumps(row.to_dict()) for row in records) + "\n", encoding="utf-8",
    )

    first = run_download_pass(cfg)
    after_first = [json.loads(line) for line in cfg.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert first["attempted"] == 1
    assert [row["download_status"] for row in after_first] == ["downloaded", "pending"]
    assert not list(cfg.manifest_path.parent.glob("*.write_in_progress*"))

    cfg.download_limit = 0
    second = run_download_pass(cfg)
    after_second = [json.loads(line) for line in cfg.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert second["downloaded"] == 2
    assert [row["download_status"] for row in after_second] == ["downloaded", "downloaded"]
    requests = [json.loads(line) for line in Path(local_proxy_origin["log"]).read_text(encoding="utf-8").splitlines()]
    assert len([row for row in requests if row["kind"] == "complete"]) == 2  # resumed row was skipped; it was never fetched twice


def test_low_disk_guard_stops_before_new_real_proxy_transfer_and_preserves_manifest(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    cfg = PrepareConfig(
        data_root=tmp_path / "medical",
        execute_download=True,
        proxy=MedicalProxyConfig(
            http_proxy=str(local_proxy_origin["proxy"]),
            https_proxy=str(local_proxy_origin["proxy"]),
        ),
        min_free_bytes=shutil.disk_usage(tmp_path).total + 1,
    )
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(
        "\n".join(json.dumps(_record(str(local_proxy_origin["url"]), f"med-budget-{index}").to_dict()) for index in (1, 2)) + "\n",
        encoding="utf-8",
    )
    summary = run_download_pass(cfg)
    persisted = [json.loads(line) for line in cfg.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert summary["stopped_low_disk"] is True
    assert [row["download_status"] for row in persisted] == ["pending", "pending"]
    assert not Path(local_proxy_origin["log"]).exists()


def test_streaming_disk_guard_checks_content_length_then_each_real_proxy_chunk_and_cleans_temp(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    """A storage-floor breach mid-stream leaves neither a partial PDF nor a false manifest success."""
    reservations: list[int | None] = []

    def disk_guard(remaining_bytes: int | None) -> str | None:
        reservations.append(remaining_bytes)
        # The first call is the Content-Length preflight.  Abort later, after
        # actual proxied payload chunks have begun streaming to the temp file.
        return "simulated storage floor" if len(reservations) >= 3 else None

    with pytest.raises(DownloadDiskGuardStopped, match="storage floor"):
        download_pdf(
            _record(str(local_proxy_origin["url"]), "med-disk-stream"),
            pdf_root=tmp_path / "pdf",
            proxy=MedicalProxyConfig(
                http_proxy=str(local_proxy_origin["proxy"]),
                https_proxy=str(local_proxy_origin["proxy"]),
            ),
            execute=True,
            allow_cf_bypass=False,
            rate_limiter=DownloadRateLimiter(20_000_000),
            disk_guard=disk_guard,
        )

    assert reservations[0] == 4_375_000  # actual response Content-Length
    assert len(reservations) >= 3  # header preflight + at least two stream checks
    target = tmp_path / "pdf" / "2026" / "Test_Journal" / "med-disk-stream.pdf"
    assert not target.exists()
    assert not list(target.parent.glob("*.write_in_progress"))


def test_real_proxy_retry_backoff_wakes_immediately_when_task_abort_event_is_set(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    """Cancellation never waits through a retry deadline after a proxy-routed error."""
    abort = threading.Event()
    timer = threading.Timer(0.15, abort.set)
    retry_url = str(local_proxy_origin["url"]).replace("/paper.pdf", "/retry")
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(DownloadCancelled, match="cancelled"):
            download_pdf(
                _record(retry_url, "med-cancel-backoff"),
                pdf_root=tmp_path / "pdf",
                proxy=MedicalProxyConfig(
                    http_proxy=str(local_proxy_origin["proxy"]),
                    https_proxy=str(local_proxy_origin["proxy"]),
                ),
                execute=True,
                allow_cf_bypass=False,
                max_retries=3,
                abort_event=abort,
                should_cancel=abort.is_set,
            )
    finally:
        timer.cancel()

    assert time.monotonic() - started < 1.0


def test_medical_start_openapi_declares_typed_conflict_and_unavailable_errors() -> None:
    """Clients can distinguish an active task conflict from executor unavailability."""
    from fastapi import FastAPI
    from app.api.routes.pipeline import router

    app = FastAPI()
    app.include_router(router)
    responses = app.openapi()["paths"]["/api/pipeline/medical-download"]["post"]["responses"]

    assert responses["409"]["content"]["application/json"]["schema"]["$ref"].endswith("MedicalDownloadConflictResponse")
    assert responses["503"]["content"]["application/json"]["schema"]["$ref"].endswith("MedicalDownloadUnavailableResponse")


def test_real_durable_queue_persists_progress_and_can_be_reopened(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    from app.services.medical_corpus.service import MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base
    from core.db import models  # noqa: F401 - register task tables

    data_root = tmp_path / "medical"
    cfg = PrepareConfig(
        data_root=data_root,
        proxy=MedicalProxyConfig(
            http_proxy=str(local_proxy_origin["proxy"]),
            https_proxy=str(local_proxy_origin["proxy"]),
        ),
    )
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(
        "\n".join(json.dumps(_record(str(local_proxy_origin["url"]), f"med-budget-{index}").to_dict()) for index in (1, 2)) + "\n",
        encoding="utf-8",
    )
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = Local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    runner = MedicalDownloadPipelineRunner(
        store=TaskStore(session_factory),
        data_root=data_root,
        proxy=cfg.proxy,
        execution_guard=lambda: None,
        enforce_designated_proxy=False,
    )
    receipt = runner.start_download(download_limit=1, allow_cf_bypass=False, wait=True)
    task = runner.store.get_task(receipt["task_id"])
    state = runner.store.get_state(receipt["task_id"])
    subtask = runner.store.get_subtask(receipt["task_id"], "download")
    assert task is not None and task["status"] == "completed", task
    assert (task["progress"], task["total"]) == (1, 1)
    assert subtask is not None and subtask["status"] == "completed"
    assert state["summary"]["downloaded"] == 1
    assert state["aggregate_bytes"] > 1024
    assert state["current_paper_id"] is None
    assert state["target_total"] == 1 and len(state["target_paper_ids"]) == 1
    manifest = [json.loads(line) for line in cfg.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert [row["download_status"] for row in manifest] == ["downloaded", "pending"]

    # A fresh runner attaches to the same TaskStore: its persistent state is
    # DB-backed, not the old process's memory or a sidecar JSON queue.
    reopened = MedicalDownloadPipelineRunner(
        store=TaskStore(session_factory), data_root=data_root,
        proxy=cfg.proxy, execution_guard=lambda: None, enforce_designated_proxy=False,
    )
    assert reopened.resume_running(wait=True) == []
    persisted = reopened.store.get_task(receipt["task_id"])
    assert persisted is not None and persisted["status"] == "completed"


def test_real_durable_task_cancel_and_retry_preserves_manifest_resume(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    """The existing Tasks cancel/retry operations stop and resume a real transfer."""
    from app.services.medical_corpus.service import MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base
    from core.db import models  # noqa: F401

    data_root = tmp_path / "medical"
    cfg = PrepareConfig(
        data_root=data_root,
        proxy=MedicalProxyConfig(
            http_proxy=str(local_proxy_origin["proxy"]),
            https_proxy=str(local_proxy_origin["proxy"]),
        ),
    )
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(
        "\n".join(json.dumps(_record(
            str(local_proxy_origin["url"]), f"med-pmc-cancel-retry-{index}",
        ).to_dict()) for index in (1, 2)) + "\n",
        encoding="utf-8",
    )
    # Cancellation runs the real downloader and poller concurrently.  Use a
    # file SQLite database just like the service rather than StaticPool's one
    # shared in-memory connection, which cannot safely commit two statements
    # in progress on Windows.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'medical-cancel-retry.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = Local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    runner = MedicalDownloadPipelineRunner(
        store=TaskStore(session_factory), data_root=data_root, proxy=cfg.proxy,
        execution_guard=lambda: None, enforce_designated_proxy=False,
    )
    receipt = runner.start_download(download_limit=0, allow_cf_bypass=False)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if runner.store.get_state(receipt["task_id"]).get("aggregate_bytes", 0) >= 64 * 1024:
            break
        time.sleep(0.02)
    else:
        pytest.fail("real medical pipeline never started its proxy-routed download")

    # This is the production runner's actual StageSpec wiring, not a handmade
    # test stage: parent and existing pipeline_stage both expose manifest
    # artifact progress from the moment a two-paper download begins.
    running = runner.store.get_task(receipt["task_id"])
    stage = runner.store.get_subtask(receipt["task_id"], "download")
    assert running is not None and (running["progress"], running["total"]) == (0, 2)
    assert stage is not None and (stage["progress"], stage["total"]) == (0, 2)

    assert runner.cancel_download(receipt["task_id"]) is True
    while time.monotonic() < deadline:
        task = runner.store.get_task(receipt["task_id"])
        if task is not None and task["status"] == "cancelled":
            break
        time.sleep(0.02)
    else:
        pytest.fail("medical pipeline did not honour Task cancellation")
    assert not list(data_root.rglob("*.write_in_progress*"))
    pending = [json.loads(line) for line in cfg.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert [row["download_status"] for row in pending] == ["pending", "pending"]

    # The task row is terminal before the worker finally releases its in-process
    # resource lock.  A real UI retry sent during that tiny interval correctly
    # receives a conflict; wait for the worker teardown before testing retry.
    while time.monotonic() < deadline:
        with runner._lock:
            active = receipt["task_id"] in runner._active
        if not active:
            break
        time.sleep(0.02)
    assert not active
    assert runner.retry_download(receipt["task_id"]) is True
    retry_deadline = time.monotonic() + 10
    while time.monotonic() < retry_deadline:
        task = runner.store.get_task(receipt["task_id"])
        if task is not None and task["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    task = runner.store.get_task(receipt["task_id"])
    assert task is not None and task["status"] == "completed", task
    completed = [json.loads(line) for line in cfg.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert [row["download_status"] for row in completed] == ["downloaded", "downloaded"]


def test_real_uvicorn_unified_tasks_api_exposes_medical_metadata_and_conflicts_on_running_task(
    tmp_path: Path,
) -> None:
    """Real Tasks routes are the only medical control-plane and must fail closed."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(__import__("os").environ)
    env["DATA_DIR"] = str(tmp_path / "runtime-data")
    env["MEDICAL_DATA_ROOT"] = str(tmp_path / "medical-artifacts")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning", "--app-dir", "tests", "medical_api_subprocess_app:app"],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 15
        task_list_url = f"http://127.0.0.1:{port}/api/tasks?task_type=medical_corpus_download"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(task_list_url, timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            pytest.fail("real uvicorn unified Tasks API did not become reachable")
        assert response.status_code == 200
        tasks = response.json()["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_type"] == "medical_corpus_download"
        task_id = tasks[0]["id"]
        detail = httpx.get(f"http://127.0.0.1:{port}/api/tasks/{task_id}", timeout=2)
        assert detail.status_code == 200
        state = detail.json()["params"]["state"]
        assert state["current_paper_id"] == "med-pmc-api-visible"
        assert state["aggregate_bytes"] == 1_250_000
        assert state["max_bytes_per_second"] == 2_000_000
        assert state["proxy"] == "http://127.0.0.1:10808"

        response = httpx.post(
            f"http://127.0.0.1:{port}/api/pipeline/medical-download",
            headers={"X-Retrival-User-Id": "test-admin", "X-Retrival-User-Tier": "admin"},
            json={"download_limit": 1, "allow_cf_bypass": False},
            timeout=2,
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["code"] == "medical_download_already_running"
        assert detail["task_id"] == task_id
    finally:
        process.terminate()
        process.wait(timeout=15)


def test_real_uvicorn_malformed_manifest_admission_never_leaves_running_task(
    tmp_path: Path,
) -> None:
    """Invalid JSONL must be rejected before durable conflict state or worker launch."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    data_root = tmp_path / "medical-artifacts"
    manifest = data_root / "manifest" / "medical_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"paper_id": "unterminated"\n', encoding="utf-8")
    env = dict(__import__("os").environ)
    env["DATA_DIR"] = str(tmp_path / "runtime-data")
    env["MEDICAL_DATA_ROOT"] = str(data_root)
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "--host", "127.0.0.1",
            "--port", str(port), "--log-level", "warning", "--app-dir", "tests",
            "medical_malformed_manifest_api_app:app",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        task_list_url = f"http://127.0.0.1:{port}/api/tasks?task_type=medical_corpus_download"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                response = httpx.get(task_list_url, timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            pytest.fail("real uvicorn malformed-manifest API did not become reachable")

        headers = {"X-Retrival-User-Id": "test-admin", "X-Retrival-User-Tier": "admin"}
        body = {"download_limit": 1, "allow_cf_bypass": False}
        first = httpx.post(
            f"http://127.0.0.1:{port}/api/pipeline/medical-download",
            headers=headers,
            json=body,
            timeout=3,
        )
        second = httpx.post(
            f"http://127.0.0.1:{port}/api/pipeline/medical-download",
            headers=headers,
            json=body,
            timeout=3,
        )
        tasks = httpx.get(task_list_url, timeout=3).json()["tasks"]

        assert first.status_code == 503
        assert second.status_code == 503  # must not become a permanent 409 conflict
        assert tasks == []
        assert not list(manifest.parent.glob("medical-download-task-*.lock"))
    finally:
        process.terminate()
        process.wait(timeout=15)


@pytest.mark.parametrize("invalid_line", ["[1]", "42"])
def test_real_uvicorn_manifest_non_object_admission_is_typed_503_with_line_context(
    tmp_path: Path,
    invalid_line: str,
) -> None:
    """Valid JSON values still need record-object validation at admission."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    data_root = tmp_path / "medical-artifacts"
    manifest = data_root / "manifest" / "medical_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(invalid_line + "\n", encoding="utf-8")
    env = dict(__import__("os").environ)
    env["DATA_DIR"] = str(tmp_path / "runtime-data")
    env["MEDICAL_DATA_ROOT"] = str(data_root)
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "--host", "127.0.0.1",
            "--port", str(port), "--log-level", "warning", "--app-dir", "tests",
            "medical_malformed_manifest_api_app:app",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        task_list_url = f"http://127.0.0.1:{port}/api/tasks?task_type=medical_corpus_download"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                httpx.get(task_list_url, timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            pytest.fail("real uvicorn non-object-manifest API did not become reachable")

        response = httpx.post(
            f"http://127.0.0.1:{port}/api/pipeline/medical-download",
            headers={"X-Retrival-User-Id": "test-admin", "X-Retrival-User-Tier": "admin"},
            json={"download_limit": 1, "allow_cf_bypass": False},
            timeout=3,
        )
        detail = str(response.json().get("detail") or "")
        tasks = httpx.get(task_list_url, timeout=3).json()["tasks"]

        assert response.status_code == 503
        assert str(manifest) in detail
        assert "line 1" in detail
        assert tasks == []
        assert not list(manifest.parent.glob("medical-download-task-*.lock"))
    finally:
        process.terminate()
        process.wait(timeout=15)


def test_medical_request_schema_does_not_regress_existing_ingest_request_contract() -> None:
    """Adding a task type must not steal the established remote-ingest fields."""
    from app.api.routes.pipeline import IngestPipelineRequest, MedicalDownloadRequest

    ingest = IngestPipelineRequest(venue="CVPR", year=2026, ssh_target="4090")
    assert ingest.remote_repo is None
    assert ingest.concurrency == 8
    assert ingest.execution_mode == "remote"
    assert set(MedicalDownloadRequest.model_fields) == {"download_limit", "allow_cf_bypass"}


def test_os_file_lock_prevents_second_medical_pipeline_from_starting_transfer(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    """The durable Task must fail before any second process can open the network lane."""
    from app.services.medical_corpus.service import MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base
    from core.db import models  # noqa: F401

    data_root = tmp_path / "medical"
    cfg = PrepareConfig(
        data_root=data_root,
        proxy=MedicalProxyConfig(http_proxy=str(local_proxy_origin["proxy"]), https_proxy=str(local_proxy_origin["proxy"])),
    )
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(json.dumps(_record(str(local_proxy_origin["url"])).to_dict()) + "\n", encoding="utf-8")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = Local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    lock = FileLock(str(data_root / "manifest" / "medical-download.lock"))
    with lock.acquire(timeout=1):
        runner = MedicalDownloadPipelineRunner(
            store=TaskStore(session_factory), data_root=data_root, proxy=cfg.proxy,
            execution_guard=lambda: None, enforce_designated_proxy=False,
        )
        receipt = runner.start_download(download_limit=1, allow_cf_bypass=False, wait=True)
    task = runner.store.get_task(receipt["task_id"])
    assert task is not None and task["status"] == "failed"
    assert not Path(local_proxy_origin["log"]).exists()


def test_two_process_like_resume_attempts_share_one_task_lease_without_a_failed_event(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    """Duplicate restart recovery must attach/stand down, never fail the same durable task."""
    from app.services.medical_corpus.service import MEDICAL_DOWNLOAD_PIPELINE, MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base
    from core.db import models  # noqa: F401
    from core.db.models import TaskEvent

    data_root = tmp_path / "medical"
    cfg = PrepareConfig(
        data_root=data_root,
        proxy=MedicalProxyConfig(http_proxy=str(local_proxy_origin["proxy"]), https_proxy=str(local_proxy_origin["proxy"])),
    )
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(
        json.dumps(_record(str(local_proxy_origin["url"]), "med-resume-lease").to_dict()) + "\n",
        encoding="utf-8",
    )
    engine = create_engine(
        f"sqlite:///{tmp_path / 'medical-resume-lease.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = Local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    store = TaskStore(session_factory)
    task_id = store.create_task(
        task_type=MEDICAL_DOWNLOAD_PIPELINE,
        params={"data_root": str(data_root), "download_limit": 1, "allow_cf_bypass": False},
        total=1,
    )
    store.set_task_status(task_id, "running")
    runner_a = MedicalDownloadPipelineRunner(
        store=TaskStore(session_factory), data_root=data_root, proxy=cfg.proxy,
        execution_guard=lambda: None, enforce_designated_proxy=False,
    )
    runner_b = MedicalDownloadPipelineRunner(
        store=TaskStore(session_factory), data_root=data_root, proxy=cfg.proxy,
        execution_guard=lambda: None, enforce_designated_proxy=False,
    )

    assert runner_a.resume_running(wait=False) == [task_id]
    assert runner_b.resume_running(wait=False) == [task_id]
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        task = store.get_task(task_id)
        if task is not None and task["status"] == "completed":
            break
        time.sleep(0.05)
    task = store.get_task(task_id)
    assert task is not None and task["status"] == "completed"
    with session_factory() as session:
        messages = [row.message or "" for row in session.query(TaskEvent).filter(TaskEvent.task_id == task_id)]
    assert not any("another process owns the medical download lock" in message for message in messages)
    requests = [json.loads(line) for line in Path(local_proxy_origin["log"]).read_text(encoding="utf-8").splitlines()]
    assert len([row for row in requests if row["kind"] == "complete"]) == 1


def test_concurrent_real_runner_starts_create_exactly_one_task_and_one_conflict(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    """The root start lease covers observe/create/launch, not only transfer I/O."""
    from app.services.medical_corpus.service import MEDICAL_DOWNLOAD_PIPELINE, MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base
    from core.db import models  # noqa: F401
    from core.db.models import Task

    data_root = tmp_path / "medical"
    cfg = PrepareConfig(
        data_root=data_root,
        proxy=MedicalProxyConfig(http_proxy=str(local_proxy_origin["proxy"]), https_proxy=str(local_proxy_origin["proxy"])),
    )
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(
        json.dumps(_record(str(local_proxy_origin["url"]), "med-atomic-start").to_dict()) + "\n",
        encoding="utf-8",
    )
    engine = create_engine(
        f"sqlite:///{tmp_path / 'medical-atomic-start.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = Local()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Make the old list_running -> create gap large enough for a real concurrent
    # caller.  A correct root lease serializes this guard too.
    def slow_preflight() -> None:
        time.sleep(0.20)

    kwargs = dict(
        store=TaskStore(session_factory), data_root=data_root, proxy=cfg.proxy,
        execution_guard=slow_preflight, enforce_designated_proxy=False,
    )
    runner_a = MedicalDownloadPipelineRunner(**kwargs)
    runner_b = MedicalDownloadPipelineRunner(**kwargs)
    gate = threading.Barrier(2)
    replies: list[dict] = []

    def submit(runner: MedicalDownloadPipelineRunner) -> None:
        gate.wait(timeout=5)
        replies.append(runner.start_download(download_limit=1, allow_cf_bypass=False, wait=False))

    threads = [threading.Thread(target=submit, args=(runner,), daemon=True) for runner in (runner_a, runner_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
    assert len(replies) == 2
    assert sorted(reply["status"] for reply in replies) == ["conflict", "started"]
    with session_factory() as session:
        assert session.query(Task).filter(Task.task_type == MEDICAL_DOWNLOAD_PIPELINE).count() == 1


def test_second_runner_persistent_cancel_stops_first_runner_real_proxy_transfer(
    tmp_path: Path, local_proxy_origin: dict[str, object],
) -> None:
    """A Tasks cancel committed by another backend process reaches the transfer owner."""
    from app.services.medical_corpus.service import MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base
    from core.db import models  # noqa: F401

    data_root = tmp_path / "medical"
    cfg = PrepareConfig(data_root=data_root, proxy=MedicalProxyConfig(
        http_proxy=str(local_proxy_origin["proxy"]), https_proxy=str(local_proxy_origin["proxy"]),
    ))
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(json.dumps(_record(str(local_proxy_origin["url"]), "med-cross-cancel").to_dict()) + "\n", encoding="utf-8")
    engine = create_engine(f"sqlite:///{tmp_path / 'medical-cross-cancel.db'}", connect_args={"check_same_thread": False, "timeout": 30})
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = Local()
        try:
            yield session; session.commit()
        except Exception:
            session.rollback(); raise
        finally:
            session.close()

    owner = MedicalDownloadPipelineRunner(store=TaskStore(session_factory), data_root=data_root, proxy=cfg.proxy, execution_guard=lambda: None, enforce_designated_proxy=False)
    other = TaskStore(session_factory)
    receipt = owner.start_download(download_limit=1, allow_cf_bypass=False)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if owner.store.get_state(receipt["task_id"]).get("aggregate_bytes", 0) >= 64 * 1024:
            break
        time.sleep(0.02)
    other.set_task_status(receipt["task_id"], "cancelled")
    while time.monotonic() < deadline:
        task = other.get_task(receipt["task_id"])
        if task is not None and task["status"] == "cancelled":
            break
        time.sleep(0.02)
    task = other.get_task(receipt["task_id"])
    assert task is not None and task["status"] == "cancelled"
    while time.monotonic() < deadline:
        with owner._lock:
            active = receipt["task_id"] in owner._active
        if not active:
            break
        time.sleep(0.02)
    assert not active
    assert not list(data_root.rglob("*.write_in_progress*"))
