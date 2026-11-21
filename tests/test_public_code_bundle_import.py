"""Seeding public-code mirrors from an uploaded git bundle, against real git and SQLite."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.public_code.bundle_import import (
    PublicCodeBundleImportService,
    PublicCodeImportConflict,
    PublicCodeImportError,
    PublicCodeImportNotFound,
)
from app.services.public_code.git_mirror import GitMirrorSourceReader
from app.services.public_code.repository import PublicCodeRepository
from app.services.public_code.source_catalog import PublicCodeSource
from core.db.database import Base

UPSTREAM = "https://example.invalid/demo/demo.git"


def _repository() -> PublicCodeRepository:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def factory():
        session = local_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return PublicCodeRepository(factory)


def _git(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def _operator_clone(tmp_path: Path, files: dict[str, str]) -> Path:
    """A local working clone with one commit on `main`, like an operator's checkout."""
    clone = tmp_path / "clone"
    clone.mkdir()
    _git("init", "-b", "main", cwd=clone)
    _git("config", "user.email", "t@example.invalid", cwd=clone)
    _git("config", "user.name", "t", cwd=clone)
    for name, text in files.items():
        (clone / name).parent.mkdir(parents=True, exist_ok=True)
        (clone / name).write_text(text, encoding="utf-8")
    _git("add", ".", cwd=clone)
    _git("commit", "-m", "init", cwd=clone)
    return clone


def _commit(clone: Path, name: str, text: str) -> str:
    (clone / name).write_text(text, encoding="utf-8")
    _git("add", ".", cwd=clone)
    _git("commit", "-m", f"edit {name}", cwd=clone)
    return _git("rev-parse", "HEAD", cwd=clone)


def _bundle(clone: Path, target: Path) -> bytes:
    _git("bundle", "create", str(target), "HEAD", "main", cwd=clone)
    return target.read_bytes()


def _source(**overrides) -> PublicCodeSource:
    values = dict(source_id="demo", domain="machine-learning",
                  relative_path="machine-learning/demo", upstream_url=UPSTREAM)
    values.update(overrides)
    return PublicCodeSource(**values)


def _service(tmp_path: Path, repository=None, **overrides) -> PublicCodeBundleImportService:
    values = dict(import_root=tmp_path / "imports", mirror_root=tmp_path / "mirrors",
                  repository=repository or _repository(), max_bundle_bytes=10 * 1024**2,
                  free_space_margin_bytes=0)
    values.update(overrides)
    return PublicCodeBundleImportService(**values)


def _upload(service, source, payload: bytes, head: str, *, chunks: int = 2):
    state = service.create(source, head_commit=head, size_bytes=len(payload),
                           sha256=hashlib.sha256(payload).hexdigest())
    step = max(1, len(payload) // chunks + 1)
    for offset in range(0, len(payload), step):
        state = service.write(state.import_id, offset, payload[offset:offset + step])
    return state


def test_bundle_becomes_a_bare_mirror_the_indexer_reads(tmp_path):
    clone = _operator_clone(tmp_path, {"src/model.py": "def forward(x):\n    return x\n"})
    head = _git("rev-parse", "HEAD", cwd=clone)
    service = _service(tmp_path)
    source = _source()
    state = _upload(service, source, _bundle(clone, tmp_path / "demo.bundle"), head)
    assert state.received_bytes == state.size_bytes

    service.mark_submitted(state.import_id)
    published = service.publish_mirror(state.import_id)

    assert published.status == "published"
    # only the record stays; the bundle bytes now live in the mirror
    assert not (tmp_path / "imports" / state.import_id / "data.bundle").exists()
    assert service.get(state.import_id).received_bytes == state.size_bytes
    mirror = tmp_path / "mirrors" / "machine-learning" / "demo.git"
    assert _git("rev-parse", "--is-bare-repository", cwd=mirror) == "true"
    # later updates fetch upstream, exactly like a network-cloned mirror
    assert _git("remote", "get-url", "origin", cwd=mirror) == UPSTREAM
    manifest = GitMirrorSourceReader(mirror_root=tmp_path / "mirrors").build_manifest(source)
    assert manifest.revision_id == head
    assert [file.path for file in manifest.files] == ["src/model.py"]


def test_upload_resumes_only_at_the_received_offset(tmp_path):
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    payload = _bundle(clone, tmp_path / "demo.bundle")
    service = _service(tmp_path)
    state = service.create(_source(), head_commit=_git("rev-parse", "HEAD", cwd=clone),
                           size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    service.write(state.import_id, 0, payload[:100])

    with pytest.raises(PublicCodeImportConflict, match="resume from that offset"):
        service.write(state.import_id, 0, payload[:100])
    with pytest.raises(PublicCodeImportError, match="exceeds the declared size_bytes"):
        service.write(state.import_id, 100, payload[100:] + b"extra")
    with pytest.raises(PublicCodeImportError, match="upload incomplete: 100 of"):
        service.mark_submitted(state.import_id)
    with pytest.raises(PublicCodeImportConflict, match="resume from that offset"):
        service.write(state.import_id, 99, payload[99:])

    # a fresh service instance (backend restart) continues from the bytes on disk
    restarted = _service(tmp_path)
    assert restarted.get(state.import_id).received_bytes == 100
    assert restarted.write(state.import_id, 100, payload[100:]).received_bytes == len(payload)


def test_corrupted_upload_fails_without_creating_a_mirror(tmp_path):
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    payload = _bundle(clone, tmp_path / "demo.bundle")
    service = _service(tmp_path)
    state = service.create(_source(), head_commit=_git("rev-parse", "HEAD", cwd=clone),
                           size_bytes=len(payload), sha256="0" * 64)
    service.write(state.import_id, 0, payload)
    service.mark_submitted(state.import_id)

    with pytest.raises(PublicCodeImportError, match="sha256 mismatch"):
        service.publish_mirror(state.import_id)
    assert service.get(state.import_id).status == "failed"
    assert not (tmp_path / "mirrors" / "machine-learning" / "demo.git").exists()


def test_bundle_head_must_be_the_declared_commit(tmp_path):
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    stale = _git("rev-parse", "HEAD", cwd=clone)
    _commit(clone, "a.py", "x = 2\n")
    service = _service(tmp_path)
    state = _upload(service, _source(), _bundle(clone, tmp_path / "demo.bundle"), stale)
    service.mark_submitted(state.import_id)

    with pytest.raises(PublicCodeImportError, match="not the declared head_commit"):
        service.publish_mirror(state.import_id)
    assert not (tmp_path / "mirrors" / "machine-learning" / "demo.git").exists()


def test_existing_mirror_is_advanced_in_place(tmp_path):
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    service = _service(tmp_path)
    first = _upload(service, _source(), _bundle(clone, tmp_path / "one.bundle"),
                    _git("rev-parse", "HEAD", cwd=clone))
    service.mark_submitted(first.import_id)
    service.publish_mirror(first.import_id)

    newer = _commit(clone, "b.py", "y = 1\n")
    second = _upload(service, _source(), _bundle(clone, tmp_path / "two.bundle"), newer)
    service.mark_submitted(second.import_id)
    service.publish_mirror(second.import_id)

    mirror = tmp_path / "mirrors" / "machine-learning" / "demo.git"
    assert _git("rev-parse", "HEAD", cwd=mirror) == newer
    assert _git("remote", "get-url", "origin", cwd=mirror) == UPSTREAM


def test_existing_mirror_with_another_origin_is_not_touched(tmp_path):
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    mirror = tmp_path / "mirrors" / "machine-learning" / "demo.git"
    _git("clone", "--mirror", str(clone), str(mirror))
    before = _git("rev-parse", "HEAD", cwd=mirror)
    newer = _commit(clone, "b.py", "y = 1\n")
    service = _service(tmp_path)
    state = _upload(service, _source(), _bundle(clone, tmp_path / "demo.bundle"), newer)
    service.mark_submitted(state.import_id)

    with pytest.raises(PublicCodeImportConflict, match="has origin"):
        service.publish_mirror(state.import_id)
    assert _git("rev-parse", "HEAD", cwd=mirror) == before


def test_import_must_match_a_registered_source(tmp_path):
    repository = _repository()
    repository.create_source(_source(), update_policy="manual", update_interval_minutes=None)
    service = _service(tmp_path, repository=repository)

    with pytest.raises(PublicCodeImportConflict, match="already exists with upstream_url"):
        service.create(_source(upstream_url="https://example.invalid/fork/demo.git"),
                       head_commit="a" * 40, size_bytes=10, sha256="b" * 64)
    state = service.create(_source(), head_commit="a" * 40, size_bytes=10, sha256="b" * 64)
    assert state.source_exists


def test_declared_size_and_disk_space_are_checked_up_front(tmp_path):
    with pytest.raises(PublicCodeImportError, match="size_bytes must be between 1 and 100"):
        _service(tmp_path, max_bundle_bytes=100).create(
            _source(), head_commit="a" * 40, size_bytes=101, sha256="b" * 64)
    tight = _service(tmp_path, disk_free=lambda path: 150, free_space_margin_bytes=0)
    with pytest.raises(PublicCodeImportError, match="not enough disk space"):
        tight.create(_source(), head_commit="a" * 40, size_bytes=100, sha256="b" * 64)


def test_unknown_or_malformed_import_ids_are_not_found(tmp_path):
    service = _service(tmp_path)
    for import_id in ("0" * 32, "../../etc", "x"):
        with pytest.raises(PublicCodeImportNotFound):
            service.get(import_id)


def test_a_new_source_cannot_take_another_sources_mirror_path(tmp_path):
    repository = _repository()
    repository.create_source(_source(source_id="other"), update_policy="manual",
                             update_interval_minutes=None)
    with pytest.raises(PublicCodeImportConflict, match="already belongs to public-code source other"):
        _service(tmp_path, repository=repository).create(
            _source(), head_commit="a" * 40, size_bytes=10, sha256="b" * 64)


def test_submitted_uploads_are_closed_and_cannot_be_discarded(tmp_path):
    clone = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    service = _service(tmp_path)
    state = _upload(service, _source(), _bundle(clone, tmp_path / "demo.bundle"),
                    _git("rev-parse", "HEAD", cwd=clone))
    service.mark_submitted(state.import_id)
    service.attach_job(state.import_id, "job-1")

    assert service.get(state.import_id).job_id == "job-1"
    with pytest.raises(PublicCodeImportConflict, match="no longer accepts data"):
        service.write(state.import_id, state.size_bytes, b"x")
    with pytest.raises(PublicCodeImportConflict, match="being published by job job-1"):
        service.discard(state.import_id)
    service.fail(state.import_id, "interrupted")
    service.discard(state.import_id)
    with pytest.raises(PublicCodeImportNotFound):
        service.get(state.import_id)


def _shallow_clone(tmp_path: Path, upstream: Path, name: str) -> Path:
    """What an operator gets from `git clone --depth 1` of a large repository."""
    target = tmp_path / name
    _git("clone", "--depth", "1", "--branch", "main", upstream.as_uri(), str(target))
    assert _git("rev-parse", "--is-shallow-repository", cwd=target) == "true"
    return target


def test_bundle_from_a_shallow_clone_becomes_a_shallow_mirror(tmp_path):
    upstream = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    _commit(upstream, "b.py", "y = 1\n")
    head = _commit(upstream, "c.py", "z = 1\n")
    shallow = _shallow_clone(tmp_path, upstream, "shallow")
    service = _service(tmp_path)
    state = _upload(service, _source(), _bundle(shallow, tmp_path / "demo.bundle"), head)
    service.mark_submitted(state.import_id)

    service.publish_mirror(state.import_id)

    mirror = tmp_path / "mirrors" / "machine-learning" / "demo.git"
    # git clone/fetch from this bundle fail on the missing parents; the import keeps
    # the same boundary a --depth 1 clone records instead
    assert (mirror / "shallow").read_text(encoding="ascii").split() == [head]
    assert _git("rev-parse", "HEAD", cwd=mirror) == head
    assert _git("config", "remote.origin.mirror", cwd=mirror) == "true"
    manifest = GitMirrorSourceReader(mirror_root=tmp_path / "mirrors").build_manifest(_source())
    assert sorted(file.path for file in manifest.files) == ["a.py", "b.py", "c.py"]


def test_shallow_bundle_advances_a_mirror_that_has_its_parent(tmp_path):
    upstream = _operator_clone(tmp_path, {"a.py": "x = 1\n"})
    service = _service(tmp_path)
    first = _upload(service, _source(), _bundle(upstream, tmp_path / "one.bundle"),
                    _git("rev-parse", "HEAD", cwd=upstream))
    service.mark_submitted(first.import_id)
    service.publish_mirror(first.import_id)

    newer = _commit(upstream, "b.py", "y = 1\n")
    shallow = _shallow_clone(tmp_path, upstream, "shallow")
    second = _upload(service, _source(), _bundle(shallow, tmp_path / "two.bundle"), newer)
    service.mark_submitted(second.import_id)
    service.publish_mirror(second.import_id)

    mirror = tmp_path / "mirrors" / "machine-learning" / "demo.git"
    assert _git("rev-parse", "HEAD", cwd=mirror) == newer
    # the parent is already in the mirror, so no shallow boundary is needed
    assert not (mirror / "shallow").exists()
    assert _git("rev-list", "--count", "HEAD", cwd=mirror) == "2"
