"""Fail-closed storage and resume guards for the medical download executor."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter

from pipeline.medical_corpus.models import MedicalPaperRecord
from pipeline.medical_corpus.prepare import PrepareConfig, run_download_pass


def test_runner_rejects_relative_medical_data_root_before_creating_a_task() -> None:
    from app.services.medical_corpus.service import MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore

    with pytest.raises(ValueError, match="MEDICAL_DATA_ROOT must be an absolute path"):
        MedicalDownloadPipelineRunner(
            store=TaskStore(),
            data_root=Path("relative-medical-root"),
            execution_guard=lambda: None,
            enforce_designated_proxy=False,
        )


def test_resume_revalidates_downloaded_manifest_artifact_before_skipping(tmp_path: Path) -> None:
    """A stale manifest must not treat a corrupt file as a verified resume success."""
    data_root = tmp_path / "medical"
    corrupt = data_root / "pdf" / "2026" / "Test" / "med-corrupt.pdf"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a PDF")
    record = MedicalPaperRecord(
        paper_id="med-corrupt", title="corrupt", year=2026, journal="Test",
        download_status="downloaded", pdf_path=str(corrupt), pdf_bytes=corrupt.stat().st_size,
        pdf_sha256=hashlib.sha256(corrupt.read_bytes()).hexdigest(),
    )
    cfg = PrepareConfig(data_root=data_root, execute_download=False)
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")

    summary = run_download_pass(cfg)
    persisted = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))

    assert summary["skipped"] == 0
    assert persisted["download_status"] == "failed"
    assert "revalidation" in str(persisted["download_error"])


def test_verified_artifact_resume_removes_only_its_sibling_write_in_progress(tmp_path: Path) -> None:
    """A completed PDF may coexist with a crash leftover; resume cleans the sibling."""
    data_root = tmp_path / "medical"
    pdf = data_root / "pdf" / "2026" / "Test" / "verified.pdf"
    pdf.parent.mkdir(parents=True)
    # This must be a genuinely verified artifact, not merely a %PDF prefix:
    # production resume invokes PdfReader before preserving ``skipped``.
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf.open("wb") as handle:
        writer.write(handle)
    sibling = pdf.with_name("verified.write_in_progress.pdf")
    sibling.write_bytes(b"partial")
    record = MedicalPaperRecord(paper_id="verified", title="v", year=2026, journal="Test", download_status="skipped", pdf_path=str(pdf), pdf_bytes=pdf.stat().st_size, pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    cfg = PrepareConfig(data_root=data_root, execute_download=False)
    cfg.manifest_path.parent.mkdir(parents=True)
    cfg.manifest_path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    run_download_pass(cfg)
    assert not sibling.exists()


def test_resume_reapplies_executor_preflight_before_relaunch(tmp_path: Path) -> None:
    """A durable running row is not authority to bypass today's host/disk guard."""
    from contextlib import contextmanager
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.services.medical_corpus.service import MEDICAL_DOWNLOAD_PIPELINE, MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    store = TaskStore(session_factory)
    task_id = store.create_task(task_type=MEDICAL_DOWNLOAD_PIPELINE, params={"data_root": str(tmp_path)}, total=1)
    store.set_task_status(task_id, "running")
    calls: list[str] = []
    runner = MedicalDownloadPipelineRunner(
        store=store, data_root=tmp_path, execution_guard=lambda: calls.append("host"),
        enforce_designated_proxy=False,
    )

    # No manifest is deliberately present: the real runner is allowed to launch,
    # then fails through the normal pipeline path.  The point is that resume made
    # a fresh host + root/disk validation before that re-execution.
    assert runner.resume_running(wait=True) == [task_id]
    assert calls == ["host"]
    assert store.get_task(task_id)["status"] == "failed"


def test_resume_rejects_a_persisted_data_root_that_differs_from_the_current_executor(tmp_path: Path) -> None:
    """A stale task must not resume onto an unintended disk after restart."""
    from contextlib import contextmanager
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.services.medical_corpus.service import MEDICAL_DOWNLOAD_PIPELINE, MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    task_root = tmp_path / "persisted-root"
    task_root.mkdir()
    executor_root = tmp_path / "configured-root"
    executor_root.mkdir()
    store = TaskStore(session_factory)
    task_id = store.create_task(
        task_type=MEDICAL_DOWNLOAD_PIPELINE, params={"data_root": str(task_root)}, total=1,
    )
    store.set_task_status(task_id, "running")
    runner = MedicalDownloadPipelineRunner(
        store=store, data_root=executor_root, execution_guard=lambda: None,
        enforce_designated_proxy=False,
    )

    assert runner.resume_running(wait=True) == []
    task = store.get_task(task_id)
    assert task is not None and task["status"] == "failed"
    assert "data-root mismatch" in task["error"]


def test_retry_rejects_a_persisted_data_root_that_differs_from_the_current_executor(tmp_path: Path) -> None:
    """Retry must preserve resume's root fence before reset or worker launch."""
    from contextlib import contextmanager

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.services.medical_corpus.service import MEDICAL_DOWNLOAD_PIPELINE, MedicalDownloadPipelineRunner
    from app.services.pipeline.store import TaskStore
    from core.db.database import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_factory():
        session = sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    task_root = tmp_path / "persisted-root"
    task_root.mkdir()
    executor_root = tmp_path / "configured-root"
    executor_root.mkdir()
    store = TaskStore(session_factory)
    task_id = store.create_task(
        task_type=MEDICAL_DOWNLOAD_PIPELINE,
        params={"data_root": str(task_root)},
        total=1,
    )
    store.set_task_status(task_id, "failed", error="previous disk guard failure")
    runner = MedicalDownloadPipelineRunner(
        store=store,
        data_root=executor_root,
        execution_guard=lambda: None,
        enforce_designated_proxy=False,
    )

    with pytest.raises(RuntimeError, match="data-root mismatch"):
        runner.retry_download(task_id)

    # No retry reset or background worker may touch the persisted, no-longer
    # approved artifact root.  Its task remains the original failed fact.
    task = store.get_task(task_id)
    assert task is not None and task["status"] == "failed"
    assert task["error"] == "previous disk guard failure"
    assert not (task_root / "manifest").exists()


def test_execute_download_cli_fail_closes_outside_dot88_but_dry_run_keeps_its_normal_path(tmp_path: Path) -> None:
    """The executable CLI shares the .88/root/proxy guard; planning does not."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "medical" / "prepare_medical_corpus.py"
    execute = subprocess.run(
        [sys.executable, str(script), "download", "--execute-download", "--data-root", str(tmp_path)],
        text=True, capture_output=True, timeout=20,
    )
    dry_run = subprocess.run(
        [sys.executable, str(script), "download", "--data-root", str(tmp_path)],
        text=True, capture_output=True, timeout=20,
    )

    assert execute.returncode != 0
    # pipeline/medical_corpus/executor.py words the refusal per platform.
    refusal = ("medical PDF execution is disabled on Windows" if os.name == "nt"
               else "medical PDF execution is disabled outside compute .88")
    assert refusal in (execute.stderr + execute.stdout)
    assert dry_run.returncode != 0  # no manifest is expected in a fresh directory
    assert refusal not in (dry_run.stderr + dry_run.stdout)


def test_medical_executor_is_compute88_or_a_declared_github_actions_runner() -> None:
    """The CLI downloads on .88 or on a GitHub Actions runner that says so; nowhere else."""
    from app.services.medical_corpus.service import designated_medical_executor

    assert designated_medical_executor({"MEDICAL_DOWNLOAD_HOST": "10.100.100.88"}, "posix") == "compute88"
    actions = {"MEDICAL_DOWNLOAD_HOST": "github-actions", "GITHUB_ACTIONS": "true"}
    assert designated_medical_executor(actions, "posix") == "github-actions"
    with pytest.raises(RuntimeError, match="outside compute .88"):  # not on a runner
        designated_medical_executor({"MEDICAL_DOWNLOAD_HOST": "github-actions"}, "posix")
    with pytest.raises(RuntimeError, match="outside compute .88"):  # a runner that did not opt in
        designated_medical_executor({"GITHUB_ACTIONS": "true"}, "posix")
    with pytest.raises(RuntimeError, match="outside compute .88"):  # the Task service stays on .88
        designated_medical_executor(actions, "posix", allow_actions=False)
    with pytest.raises(RuntimeError, match="disabled on Windows"):
        designated_medical_executor(actions, "nt")


def test_actions_cli_execution_keeps_the_proxy_but_takes_its_own_absolute_root(tmp_path: Path, monkeypatch) -> None:
    from pipeline.medical_corpus import executor as service
    from pipeline.medical_corpus.proxy import DEFAULT_PROXY_URL, MedicalProxyConfig

    monkeypatch.setattr(service, "designated_medical_executor", lambda **_: "github-actions")
    proxy = MedicalProxyConfig(http_proxy=DEFAULT_PROXY_URL, https_proxy=DEFAULT_PROXY_URL)
    assert service.require_designated_medical_cli_execution(tmp_path, proxy) == tmp_path
    with pytest.raises(ValueError, match="absolute"):
        service.require_designated_medical_cli_execution(Path("relative/medical"), proxy)
    other = MedicalProxyConfig(http_proxy="http://127.0.0.1:9999", https_proxy="http://127.0.0.1:9999")
    with pytest.raises(RuntimeError, match="requires proxy"):
        service.require_designated_medical_cli_execution(tmp_path, other)


def test_durable_low_disk_pipeline_fails_instead_of_promoting_limited_batch(
    tmp_path: Path,
) -> None:
    """A persisted low-disk canary must fail the real Pipeline, even for a 1-PDF batch.

    The forced floor is stored on the durable task row, then the production
    runner's real StageSpec / TaskStore / PipelineOrchestrator is invoked.  It
    therefore covers the formerly-dangerous path where a `download_limit=1`
    could turn a stopped, still-pending record into a false `1/1 completed`.
    No transfer is allowed to reach the deliberately unreachable local proxy.
    """
    from contextlib import contextmanager

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.services.medical_corpus.service import (
        MEDICAL_DOWNLOAD_PIPELINE,
        MedicalDownloadPipelineRunner,
    )
    from app.services.pipeline.store import TaskStore
    from core.db import models  # noqa: F401 - registers Task tables
    from core.db.database import Base
    from pipeline.medical_corpus.proxy import MedicalProxyConfig

    data_root = tmp_path / "medical"
    data_root.mkdir()
    unreachable_proxy = "http://127.0.0.1:9"
    manifest = data_root / "manifest" / "medical_manifest.jsonl"
    manifest.parent.mkdir()
    record = MedicalPaperRecord(
        paper_id="med-low-disk-canary",
        title="low disk canary",
        year=2026,
        journal="Test",
        is_open_access=True,
        has_pdf=True,
        pdf_url="http://127.0.0.1:9/paper.pdf",
        download_status="pending",
    )
    manifest.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
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
        params={
            "data_root": str(data_root),
            "download_limit": 1,
            "allow_cf_bypass": False,
            "proxy": unreachable_proxy,
            # This is deliberately above the actual temporary-volume free
            # space.  It models a mount whose free-space floor is crossed
            # between durable task creation and the first transfer.
            "min_free_bytes": 1 << 62,
        },
        total=1,
    )
    store.set_task_status(task_id, "running")
    runner = MedicalDownloadPipelineRunner(
        store=store,
        data_root=data_root,
        proxy=MedicalProxyConfig(
            http_proxy=unreachable_proxy,
            https_proxy=unreachable_proxy,
        ),
        execution_guard=lambda: None,
        enforce_designated_proxy=False,
    )

    # `_launch` is the runner's production wiring: it constructs the real
    # MedicalDownloadContext and generated StageSpec, instead of a hand-made
    # stage or a patched downloader.
    runner._launch(task_id, wait=True)

    task = TaskStore(session_factory).get_task(task_id)
    state = TaskStore(session_factory).get_state(task_id)
    stage = TaskStore(session_factory).get_subtask(task_id, "download")
    persisted = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert task is not None and task["status"] == "failed"
    assert (task["progress"], task["total"]) == (0, 1)
    assert stage is not None and stage["status"] == "failed"
    # The stage failing is necessary but not sufficient: it must fail *because
    # of the disk guard*.  `summary` is only written after run_download_pass
    # RETURNS (service.py:247), so a raised exception yields a failed stage with
    # no summary at all.  Reporting the stage message here is what distinguishes
    # "guard worked" from "something else broke first" — a bare
    # `state["summary"]` just raises KeyError and names neither.
    summary = state.get("summary")
    assert summary is not None, (
        "download stage failed without recording a summary, so it did not stop "
        f"at the disk guard. stage={stage} task_error={task.get('error')!r}"
    )
    assert summary["stopped_low_disk"] is True
    assert summary["disk_guard_reason"]
    assert persisted[0]["download_status"] == "pending"
