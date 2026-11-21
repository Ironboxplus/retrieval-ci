"""Seed or update a public-code bare mirror from an uploaded Git bundle.

The server normally clones each source from its upstream. Large repositories
(pytorch, linux, llvm-project) time out doing that through the proxy, while an
operator often already has a clean clone. The operator uploads that clone as a
`git bundle`; the server turns it into the same bare mirror a network clone
would have produced (origin = upstream, so later updates still fetch
upstream), then chunking, embedding and indexing run server-side as usual.

Uploads are resumable: each import is a directory holding `meta.json` and the
bundle bytes received so far, so a dropped connection or a backend restart
continues at the byte count on disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Protocol
import uuid

from app.services.public_code.git_mirror import mirror_path_for
from app.services.public_code.source_catalog import PublicCodeSource


_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_BUNDLE_NAME = "data.bundle"
_META_NAME = "meta.json"


class PublicCodeImportError(ValueError):
    """A client-correctable import problem (maps to HTTP 4xx)."""


class PublicCodeImportConflict(PublicCodeImportError):
    """The request contradicts existing state (maps to HTTP 409)."""


class PublicCodeImportNotFound(PublicCodeImportError):
    """No such import (maps to HTTP 404)."""


class PublicCodeSourceLookup(Protocol):
    def get_source(self, source_id: str) -> Any: ...

    def list_sources(self) -> Any: ...


@dataclass(frozen=True)
class PublicCodeImportState:
    import_id: str
    source_id: str
    domain: str
    relative_path: str
    upstream_url: str
    default_ref: str
    head_commit: str
    size_bytes: int
    sha256: str
    received_bytes: int
    status: str
    source_exists: bool
    update_policy: str
    update_interval_minutes: int | None
    language_overrides: dict[str, str]
    created_by_user_id: str | None
    created_at: float
    updated_at: float
    job_id: str | None = None
    error: str | None = None
    private_project: str | None = None

    def source(self) -> PublicCodeSource:
        return PublicCodeSource(
            source_id=self.source_id,
            domain=self.domain,
            relative_path=self.relative_path,
            upstream_url=self.upstream_url,
            default_ref=self.default_ref,
            language_overrides=self.language_overrides,
        )


class PublicCodeBundleImportService:
    """Receive a bundle, verify it, and publish it as a source's bare mirror."""

    def __init__(
        self,
        *,
        import_root: Path,
        mirror_root: Path,
        repository: PublicCodeSourceLookup,
        max_bundle_bytes: int,
        git_timeout_seconds: float = 3600.0,
        free_space_margin_bytes: int = 10 * 1024**3,
        disk_free: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
        command_runner: Callable[..., Any] = subprocess.run,
    ):
        if max_bundle_bytes < 1:
            raise ValueError("max_bundle_bytes must be positive")
        self._import_root = import_root
        self._mirror_root = mirror_root
        self._repository = repository
        self._max_bundle_bytes = max_bundle_bytes
        self._git_timeout_seconds = git_timeout_seconds
        self._free_space_margin_bytes = free_space_margin_bytes
        self._disk_free = disk_free
        self._command_runner = command_runner
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ---- upload ----

    def create(
        self,
        source: PublicCodeSource,
        *,
        head_commit: str,
        size_bytes: int,
        sha256: str,
        update_policy: str = "manual",
        update_interval_minutes: int | None = None,
        created_by_user_id: str | None = None,
        private_project: str | None = None,
    ) -> PublicCodeImportState:
        head_commit = head_commit.strip().lower()
        sha256 = sha256.strip().lower()
        if not _COMMIT_PATTERN.fullmatch(head_commit):
            raise PublicCodeImportError("head_commit must be a full 40-character commit id")
        if not _SHA256_PATTERN.fullmatch(sha256):
            raise PublicCodeImportError("sha256 must be 64 lowercase hex characters")
        if not 1 <= size_bytes <= self._max_bundle_bytes:
            raise PublicCodeImportError(
                f"size_bytes must be between 1 and {self._max_bundle_bytes}")
        existing = self._repository.get_source(source.source_id)
        if existing is None:
            for other in self._repository.list_sources():
                if other.relative_path == source.relative_path:
                    raise PublicCodeImportConflict(
                        f"relative_path {source.relative_path} already belongs to "
                        f"public-code source {other.source_id}")
        else:
            for field_name in ("upstream_url", "relative_path", "default_ref", "domain"):
                if getattr(existing, field_name) != getattr(source, field_name):
                    raise PublicCodeImportConflict(
                        f"public-code source {source.source_id} already exists with "
                        f"{field_name}={getattr(existing, field_name)!r}; the import "
                        f"declares {getattr(source, field_name)!r}")
        self._import_root.mkdir(parents=True, exist_ok=True)
        # the bundle, then the mirror built from it, both land on this disk
        needed = 2 * size_bytes + self._free_space_margin_bytes
        free = self._disk_free(self._import_root)
        if free < needed:
            raise PublicCodeImportError(
                f"not enough disk space for a {size_bytes}-byte bundle: "
                f"{free} bytes free, {needed} needed")
        now = time.time()
        state = PublicCodeImportState(
            import_id=uuid.uuid4().hex,
            source_id=source.source_id,
            domain=source.domain,
            relative_path=source.relative_path,
            upstream_url=source.upstream_url,
            default_ref=source.default_ref,
            head_commit=head_commit,
            size_bytes=size_bytes,
            sha256=sha256,
            received_bytes=0,
            status="uploading",
            source_exists=existing is not None,
            update_policy=update_policy,
            update_interval_minutes=update_interval_minutes,
            language_overrides=dict(source.language_overrides),
            created_by_user_id=created_by_user_id,
            created_at=now,
            updated_at=now,
            private_project=private_project,
        )
        directory = self._directory(state.import_id, must_exist=False)
        directory.mkdir()
        (directory / _BUNDLE_NAME).touch()
        self._write_meta(state)
        return state

    def get(self, import_id: str) -> PublicCodeImportState:
        directory = self._directory(import_id)
        state = self._read_meta(directory)
        bundle = directory / _BUNDLE_NAME
        if bundle.is_file():
            return replace(state, received_bytes=bundle.stat().st_size)
        # a published import keeps only its record; the bundle is now the mirror
        return replace(state, received_bytes=state.size_bytes if state.status == "published" else 0)

    def write(self, import_id: str, offset: int, data: bytes) -> PublicCodeImportState:
        """Append `data` at `offset`, which must equal the bytes already received."""

        with self._lock(import_id):
            state = self.get(import_id)
            if state.status != "uploading":
                raise PublicCodeImportConflict(
                    f"import {import_id} is {state.status}; it no longer accepts data")
            if offset != state.received_bytes:
                raise PublicCodeImportConflict(
                    f"offset {offset} does not match the {state.received_bytes} bytes "
                    "already received; resume from that offset")
            if state.received_bytes + len(data) > state.size_bytes:
                raise PublicCodeImportError(
                    f"upload exceeds the declared size_bytes={state.size_bytes}")
            with (self._directory(import_id) / _BUNDLE_NAME).open("ab") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            return self.get(import_id)

    def mark_submitted(self, import_id: str) -> PublicCodeImportState:
        """Close a complete upload; from here on only the import job touches it."""

        with self._lock(import_id):
            state = self.get(import_id)
            if state.status != "uploading":
                raise PublicCodeImportConflict(f"import {import_id} is already {state.status}")
            if state.received_bytes != state.size_bytes:
                raise PublicCodeImportError(
                    f"upload incomplete: {state.received_bytes} of {state.size_bytes} bytes")
            state = replace(state, status="submitted", updated_at=time.time())
            self._write_meta(state)
            return state

    def attach_job(self, import_id: str, job_id: str) -> PublicCodeImportState:
        return self._update(import_id, job_id=job_id)

    def fail(self, import_id: str, error: str) -> PublicCodeImportState:
        return self._update(import_id, status="failed", error=error)

    def discard(self, import_id: str) -> None:
        with self._lock(import_id):
            state = self.get(import_id)
            if state.status == "submitted":
                raise PublicCodeImportConflict(
                    f"import {import_id} is being published by job {state.job_id}")
            shutil.rmtree(self._directory(import_id))

    # ---- publish (runs inside the import job) ----

    def publish_mirror(
        self,
        import_id: str,
        progress: Callable[[str], None] = lambda message: None,
    ) -> PublicCodeImportState:
        """Verify the upload and make it the source's bare mirror. Idempotent per commit."""

        state = self.get(import_id)
        bundle = self._directory(import_id) / _BUNDLE_NAME
        try:
            progress("verifying upload size and sha256")
            if state.received_bytes != state.size_bytes:
                raise PublicCodeImportError(
                    f"upload incomplete: {state.received_bytes} of {state.size_bytes} bytes")
            digest = _sha256_file(bundle)
            if digest != state.sha256:
                raise PublicCodeImportError(
                    f"sha256 mismatch: declared {state.sha256}, received {digest}")
            progress("checking the bundle's refs")
            heads = self._list_heads(bundle)
            self._require_head(state, heads)
            source = state.source()
            mirror = mirror_path_for(self._mirror_root, source)
            if mirror.is_dir():
                progress("adding the bundle to the existing mirror")
                self._fetch_into(mirror, bundle, state, heads)
            else:
                progress("building a bare mirror from the bundle")
                self._clone_into(mirror, bundle, state, heads)
            revision = self._rev_parse(mirror, f"{state.default_ref}^{{commit}}")
            if revision != state.head_commit:
                raise PublicCodeImportError(
                    f"after import {state.default_ref} is {revision}, "
                    f"not the declared {state.head_commit}")
        except Exception as exc:
            self._update(import_id, status="failed", error=str(exc))
            raise
        state = self._update(import_id, status="published", error=None)
        bundle.unlink()
        return state

    def _list_heads(self, bundle: Path) -> dict[str, str]:
        result = self._git("bundle", "list-heads", str(bundle))
        if result.returncode != 0:
            raise PublicCodeImportError(f"not a readable git bundle: {_stderr(result)}")
        heads: dict[str, str] = {}
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            commit, _, ref = line.strip().partition(" ")
            if commit and ref:
                heads[ref] = commit
        return heads

    @staticmethod
    def _require_head(state: PublicCodeImportState, heads: dict[str, str]) -> None:
        ref = state.default_ref
        candidates = [ref] if ref == "HEAD" or ref.startswith("refs/") else [
            f"refs/heads/{ref}", f"refs/tags/{ref}", ref]
        for candidate in candidates:
            if candidate in heads:
                if heads[candidate] != state.head_commit:
                    raise PublicCodeImportError(
                        f"bundle {candidate} is {heads[candidate]}, "
                        f"not the declared head_commit {state.head_commit}")
                return
        raise PublicCodeImportError(
            f"bundle has no {ref} ref; create it with "
            f"`git bundle create <file> {ref if ref != 'HEAD' else 'HEAD <branch>'}`")

    def _clone_into(
        self, mirror: Path, bundle: Path, state: PublicCodeImportState, heads: dict[str, str],
    ) -> None:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        candidate = mirror.with_name(f"{mirror.name}.candidate")
        if candidate.exists():
            shutil.rmtree(candidate)
        try:
            self._check(self._git("init", "--quiet", "--bare", str(candidate)),
                        "cannot create the bare mirror")
            # the configuration `git clone --mirror <upstream>` writes, so later
            # updates fetch upstream exactly as for a network-cloned mirror
            for key, value in (("remote.origin.url", state.upstream_url),
                               ("remote.origin.fetch", "+refs/*:refs/*"),
                               ("remote.origin.mirror", "true")):
                self._check(self._git("-C", str(candidate), "config", key, value),
                            f"cannot set {key}")
            self._unbundle(candidate, bundle, heads)
            os.replace(candidate, mirror)
        except Exception:
            shutil.rmtree(candidate, ignore_errors=True)
            raise

    def _fetch_into(
        self, mirror: Path, bundle: Path, state: PublicCodeImportState, heads: dict[str, str],
    ) -> None:
        origin = self._git("-C", str(mirror), "remote", "get-url", "origin")
        origin_url = origin.stdout.decode("utf-8", errors="replace").strip()
        if origin.returncode != 0 or origin_url != state.upstream_url:
            raise PublicCodeImportConflict(
                f"existing mirror {mirror} has origin {origin_url!r}, "
                f"not {state.upstream_url!r}")
        self._unbundle(mirror, bundle, heads)

    def _unbundle(self, repository: Path, bundle: Path, heads: dict[str, str]) -> None:
        """Store the bundle's objects and refs; bundles from shallow clones included.

        Large repositories are usually cloned with --depth 1, and `git clone`/`fetch`
        from such a bundle fail their connectivity check on the absent parents.
        `bundle unbundle` stores the objects without that check; the commits whose
        parents are absent are then recorded in `shallow`, the same boundary the
        operator's --depth clone keeps. Finally every object the refs need must be
        present, so a truncated or partial bundle still fails here."""

        repo = str(repository)
        self._check(self._git("-C", repo, "bundle", "unbundle", str(bundle)),
                    "cannot unpack the bundle")
        branches = {ref: commit for ref, commit in heads.items() if ref.startswith("refs/")}
        for ref, commit in branches.items():
            self._check(self._git("-C", repo, "update-ref", ref, commit), f"cannot set {ref}")
        head = heads.get("HEAD")
        if head is not None:
            branch = next((ref for ref in sorted(branches, key=_branch_preference)
                           if ref.startswith("refs/heads/") and branches[ref] == head), None)
            if branch is not None:
                self._check(self._git("-C", repo, "symbolic-ref", "HEAD", branch),
                            "cannot point HEAD at the bundle branch")
            else:
                self._check(self._git("-C", repo, "update-ref", "--no-deref", "HEAD", head),
                            "cannot set HEAD")
        tips = sorted(set(branches.values()) | ({head} if head else set()))
        if self._git("-C", repo, "rev-list", "--quiet", *tips).returncode != 0:
            self._record_shallow_boundaries(repository)
        self._check(self._git("-C", repo, "rev-list", "--objects", "--quiet", *tips),
                    "the bundle does not contain every object its refs need")

    def _record_shallow_boundaries(self, repository: Path) -> None:
        repo = str(repository)
        listing = self._git("-C", repo, "cat-file", "--batch-all-objects",
                            "--batch-check=%(objecttype) %(objectname)")
        self._check(listing, "cannot list the mirror objects")
        commits = [line.split()[1] for line in listing.stdout.decode("ascii").splitlines()
                   if line.startswith("commit ")]
        # --no-walk reads each commit's parent ids without loading the parents
        parents = self._git("-C", repo, "rev-list", "--parents", "--no-walk=unsorted", "--stdin",
                            input_bytes="\n".join(commits).encode("ascii") + b"\n")
        self._check(parents, "cannot read commit parents")
        known = set(commits)
        boundary = set()
        for line in parents.stdout.decode("ascii").splitlines():
            commit, *parent_ids = line.split()
            if any(parent not in known for parent in parent_ids):
                boundary.add(commit)
        shallow_file = repository / "shallow"
        if shallow_file.is_file():
            boundary.update(shallow_file.read_text(encoding="ascii").split())
        shallow_file.write_text("".join(f"{commit}\n" for commit in sorted(boundary)),
                                encoding="ascii")

    def _rev_parse(self, mirror: Path, spec: str) -> str:
        result = self._git("-C", str(mirror), "rev-parse", "--verify", "--quiet", spec)
        return result.stdout.decode("ascii", errors="replace").strip()

    def _git(
        self, *arguments: str, input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        stdin = {"input": input_bytes} if input_bytes is not None else {"stdin": subprocess.DEVNULL}
        try:
            return self._command_runner(
                ["git", *arguments],
                **stdin,
                capture_output=True,
                timeout=self._git_timeout_seconds,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PublicCodeImportError(
                f"git {arguments[0]} timed out after {self._git_timeout_seconds:g} seconds"
            ) from exc

    @staticmethod
    def _check(result: subprocess.CompletedProcess[bytes], message: str) -> None:
        if result.returncode != 0:
            raise PublicCodeImportError(f"{message}: {_stderr(result)}")

    # ---- persistence ----

    def _directory(self, import_id: str, *, must_exist: bool = True) -> Path:
        if not _IMPORT_ID_PATTERN.fullmatch(import_id):
            raise PublicCodeImportNotFound(f"public-code import not found: {import_id}")
        directory = self._import_root / import_id
        if must_exist and not (directory / _META_NAME).is_file():
            raise PublicCodeImportNotFound(f"public-code import not found: {import_id}")
        return directory

    def _lock(self, import_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(import_id, threading.Lock())

    def _update(self, import_id: str, **changes: Any) -> PublicCodeImportState:
        with self._lock(import_id):
            state = replace(self.get(import_id), updated_at=time.time(), **changes)
            self._write_meta(state)
            return state

    def _write_meta(self, state: PublicCodeImportState) -> None:
        directory = self._directory(state.import_id, must_exist=False)
        temporary = directory / f"{_META_NAME}.tmp"
        temporary.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, directory / _META_NAME)

    @staticmethod
    def _read_meta(directory: Path) -> PublicCodeImportState:
        values = json.loads((directory / _META_NAME).read_text(encoding="utf-8"))
        return PublicCodeImportState(**values)


def _branch_preference(ref: str) -> tuple[int, str]:
    """When several branches share HEAD's commit, point HEAD at the usual default."""

    order = {"refs/heads/main": 0, "refs/heads/master": 1}
    return (order.get(ref, 2), ref)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stderr(result: subprocess.CompletedProcess[bytes]) -> str:
    return (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip() \
        or "git command failed"
