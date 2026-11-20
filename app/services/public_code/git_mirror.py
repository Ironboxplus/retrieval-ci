"""Read committed source directly from local bare Git mirrors.

Public source mirrors stay bare because a Windows worktree can collapse paths
that differ only by case.  Git object reads preserve the repository's exact
tree while allowing the existing Core code chunker to consume source strings.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable

from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.source_catalog import PublicCodeSource
from core.code_kb.source_text import BinarySourceTextError, decode_source_bytes
from core.code_kb.source_policy import FALLBACK_EXTS, MAX_FILE_BYTES, resolve_language
from core.indexing.revision_manifest import RevisionFile, RevisionManifest


class GitMirrorReadError(RuntimeError):
    """Raised when a configured bare mirror cannot provide a source revision."""


_TRANSIENT_GIT_NETWORK_ERRORS = (
    b"connection reset",
    b"connection refused",
    b"connection timed out",
    b"could not resolve host",
    b"could not resolve proxy",
    b"early eof",
    b"failed to connect",
    b"gnutls recv error",
    b"operation timed out",
    b"proxy connect aborted",
    b"recv failure",
    b"rpc failed",
    b"send failure",
    b"the remote end hung up unexpectedly",
    b"unexpected disconnect",
)
_TRANSIENT_HTTP_ERROR = re.compile(rb"(?:returned error:|response)\s*5\d\d\b")


class GitMirrorSourceReader:
    """Build code manifests and read UTF-8 source blobs from local bare mirrors."""

    def __init__(
        self,
        *,
        mirror_root: Path,
        proxy_url: str | None = None,
        timeout_seconds: float = 30.0,
        network_timeout_seconds: float = 900.0,
        network_retry_attempts: int = 3,
        network_retry_backoff_seconds: float = 2.0,
        access_profiles: GitAccessProfileResolver | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
        retry_sleep: Callable[[float], None] = time.sleep,
    ):
        if (
            not math.isfinite(timeout_seconds)
            or not math.isfinite(network_timeout_seconds)
            or timeout_seconds <= 0
            or network_timeout_seconds <= 0
        ):
            raise ValueError("Git command timeouts must be finite and positive")
        if network_retry_attempts < 1:
            raise ValueError("network_retry_attempts must be at least 1")
        if (
            not math.isfinite(network_retry_backoff_seconds)
            or network_retry_backoff_seconds < 0
        ):
            raise ValueError(
                "network_retry_backoff_seconds must be finite and non-negative"
            )
        self._mirror_root = mirror_root
        self._timeout_seconds = timeout_seconds
        self._network_timeout_seconds = network_timeout_seconds
        self._network_retry_attempts = network_retry_attempts
        self._network_retry_backoff_seconds = network_retry_backoff_seconds
        self._access_profiles = access_profiles or GitAccessProfileResolver(
            default_proxy_url=proxy_url,
        )
        self._command_runner = command_runner
        self._retry_sleep = retry_sleep

    def refresh_source(self, source: PublicCodeSource) -> None:
        """Fetch the configured upstream into its bare mirror before indexing HEAD."""

        mirror_path = self._mirror_path(source, require_exists=False)
        if not mirror_path.is_dir():
            if mirror_path.exists():
                raise GitMirrorReadError(
                    f"bare mirror path is not a directory for {source.source_id}: {mirror_path}"
                )
            self._clone_source(source, mirror_path)
        origin_result = self._run_git_bytes(
            source,
            "-C",
            str(mirror_path),
            "remote",
            "get-url",
            "origin",
        )
        origin_url = origin_result.stdout.decode("utf-8", errors="replace").strip()
        if origin_result.returncode != 0 or not origin_url:
            raise GitMirrorReadError(
                f"cannot read bare mirror origin for {source.source_id}: "
                f"{_git_error(origin_result)}"
            )
        if origin_url != source.upstream_url:
            raise GitMirrorReadError(
                "bare mirror origin differs from the public source catalog for "
                f"{source.source_id}: expected {source.upstream_url!r}, got {origin_url!r}"
            )
        result = self._run_network_git_bytes(
            source,
            "-C",
            str(mirror_path),
            "fetch",
            "--prune",
            "--force",
            "origin",
        )
        if result.returncode != 0:
            raise GitMirrorReadError(
                f"cannot refresh bare mirror for {source.source_id}: {_git_error(result)}"
            )

    def build_manifest(
        self,
        source: PublicCodeSource,
        *,
        revision: str | None = None,
    ) -> RevisionManifest:
        """Return all tracked code candidates for one committed source revision."""

        mirror_path = self._mirror_path(source)
        resolved_revision = revision or self._resolve_revision(
            source,
            mirror_path,
            source.default_ref,
        )
        source_files: list[RevisionFile] = []
        for mode, object_type, object_id, size_bytes, path in self._tree_entries(
            source,
            mirror_path,
            resolved_revision,
        ):
            if mode not in {"100644", "100755"} or object_type != "blob":
                continue
            if size_bytes is None:
                raise GitMirrorReadError(
                    f"Git blob has no reported size for {source.source_id}:{path}"
                )
            if size_bytes > MAX_FILE_BYTES:
                continue
            language = source.language_overrides.get(Path(path).suffix.lower()) or resolve_language(path)
            if language is None and Path(path).suffix.lower() not in FALLBACK_EXTS:
                continue
            source_files.append(
                RevisionFile(
                    path=path,
                    content_hash=object_id,
                    language=language or "text",
                    content_role="source",
                    size_bytes=size_bytes,
                )
            )

        return RevisionManifest(revision_id=resolved_revision, files=tuple(source_files))

    def read_source(self, source: PublicCodeSource, file: RevisionFile) -> str:
        """Read one manifest blob as UTF-8 text without creating a worktree."""

        result = self._run_git_bytes(
            source,
            "-C",
            str(self._mirror_path(source)),
            "cat-file",
            "blob",
            file.content_hash,
        )
        if result.returncode != 0:
            raise GitMirrorReadError(
                f"cannot read blob for {source.source_id}:{file.path}: {_git_error(result)}"
            )
        try:
            return decode_source_bytes(result.stdout)
        except BinarySourceTextError as exc:
            raise GitMirrorReadError(
                f"source blob is binary for {source.source_id}:{file.path}"
            ) from exc

    def _mirror_path(self, source: PublicCodeSource, *, require_exists: bool = True) -> Path:
        mirror_path = mirror_path_for(self._mirror_root, source)
        if require_exists and not mirror_path.is_dir():
            raise GitMirrorReadError(
                f"bare mirror is missing for {source.source_id}: {mirror_path}"
            )
        return mirror_path

    def _clone_source(self, source: PublicCodeSource, mirror_path: Path) -> None:
        """Clone into a verified sibling candidate and atomically publish on success."""

        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path = mirror_path.with_name(f"{mirror_path.name}.candidate")
        self._remove_clone_candidate(candidate_path, mirror_path=mirror_path)
        try:
            clone_result = self._run_network_git_bytes(
                source,
                "clone",
                "--mirror",
                "--no-tags",
                source.upstream_url,
                str(candidate_path),
                cwd=self._mirror_root,
                prepare_retry=lambda: self._remove_clone_candidate(
                    candidate_path,
                    mirror_path=mirror_path,
                ),
            )
        except GitMirrorReadError:
            self._remove_clone_candidate(candidate_path, mirror_path=mirror_path)
            raise
        if clone_result.returncode != 0:
            self._remove_clone_candidate(candidate_path, mirror_path=mirror_path)
            raise GitMirrorReadError(
                f"cannot clone bare mirror for {source.source_id}: {_git_error(clone_result)}"
            )
        try:
            os.replace(candidate_path, mirror_path)
        except OSError as exc:
            self._remove_clone_candidate(candidate_path, mirror_path=mirror_path)
            raise GitMirrorReadError(
                f"cannot publish bare mirror for {source.source_id}: {exc}"
            ) from exc

    def _remove_clone_candidate(self, candidate_path: Path, *, mirror_path: Path) -> None:
        expected_name = f"{mirror_path.name}.candidate"
        resolved_root = self._mirror_root.resolve()
        resolved_candidate = candidate_path.resolve(strict=False)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise GitMirrorReadError(
                f"clone candidate escaped mirror root: {candidate_path}"
            ) from exc
        if candidate_path.name != expected_name:
            raise GitMirrorReadError(f"unexpected clone candidate path: {candidate_path}")
        if candidate_path.is_dir():
            shutil.rmtree(candidate_path)
        elif candidate_path.exists() or candidate_path.is_symlink():
            candidate_path.unlink()

    def _resolve_revision(
        self,
        source: PublicCodeSource,
        mirror_path: Path,
        ref: str,
    ) -> str:
        result = self._run_git_bytes(
            source,
            "-C",
            str(mirror_path),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{ref}^{{commit}}",
        )
        revision = result.stdout.decode("ascii", errors="replace").strip()
        if result.returncode != 0 or not revision:
            raise GitMirrorReadError(
                f"bare mirror has no verified commit for {ref}: {mirror_path}"
            )
        return revision

    def _tree_entries(
        self,
        source: PublicCodeSource,
        mirror_path: Path,
        revision: str,
    ) -> list[tuple[str, str, str, int | None, str]]:
        result = self._run_git_bytes(
            source,
            "-C",
            str(mirror_path),
            "ls-tree",
            "-r",
            "-z",
            "-l",
            revision,
        )
        if result.returncode != 0:
            raise GitMirrorReadError(
                f"cannot list source revision {revision} in {mirror_path}: {_git_error(result)}"
            )

        entries: list[tuple[str, str, str, int | None, str]] = []
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            metadata, separator, raw_path = record.partition(b"\t")
            fields = metadata.split()
            if not separator or len(fields) != 4:
                raise GitMirrorReadError(
                    f"unexpected git tree record in {mirror_path} at revision {revision}"
                )
            try:
                path = raw_path.decode("utf-8")
                mode = fields[0].decode("ascii")
                object_type = fields[1].decode("ascii")
                object_id = fields[2].decode("ascii")
                size_bytes = None if fields[3] == b"-" else int(fields[3])
            except (UnicodeDecodeError, ValueError) as exc:
                raise GitMirrorReadError(
                    f"unsupported Git tree path or size in {mirror_path} at revision {revision}"
                ) from exc
            entries.append((mode, object_type, object_id, size_bytes, path))
        return entries

    def _run_network_git_bytes(
        self,
        source: PublicCodeSource,
        *arguments: str,
        cwd: Path | None = None,
        prepare_retry: Callable[[], None] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run retry-safe network Git commands with bounded exponential backoff."""

        for attempt in range(1, self._network_retry_attempts + 1):
            try:
                result = self._run_git_bytes(
                    source,
                    *arguments,
                    cwd=cwd,
                    network=True,
                )
            except GitMirrorReadError as exc:
                if attempt == self._network_retry_attempts or not isinstance(
                    exc.__cause__, subprocess.TimeoutExpired
                ):
                    raise
            else:
                if (
                    result.returncode == 0
                    or attempt == self._network_retry_attempts
                    or not _is_transient_network_failure(result)
                ):
                    return result

            if prepare_retry is not None:
                prepare_retry()
            retry_delay = self._network_retry_backoff_seconds * (2 ** (attempt - 1))
            self._retry_sleep(retry_delay)

        raise AssertionError("network Git retry loop exited without a result")

    def _run_git_bytes(
        self,
        source: PublicCodeSource,
        *arguments: str,
        cwd: Path | None = None,
        network: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        command_timeout = (
            self._network_timeout_seconds if network else self._timeout_seconds
        )
        try:
            environment = self._access_profiles.environment_for(source, network=network)
            return self._command_runner(
                ["git", *arguments],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=command_timeout,
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            command_kind = "network git command" if network else "git command"
            raise GitMirrorReadError(
                f"{command_kind} timed out after {command_timeout:g} seconds "
                f"for {source.source_id}"
            ) from exc
        except OSError as exc:
            raise GitMirrorReadError(f"git command failed for {source.source_id}: {exc}") from exc


def mirror_path_for(mirror_root: Path, source: PublicCodeSource) -> Path:
    """The one bare-mirror location of a source, shared by fetch and bundle import."""

    return mirror_root.joinpath(*source.relative_path.split("/")).with_suffix(".git")


def _git_error(result: subprocess.CompletedProcess[bytes]) -> str:
    return (result.stderr or result.stdout).decode("utf-8", errors="replace").strip() or "git command failed"


def _is_transient_network_failure(result: subprocess.CompletedProcess[bytes]) -> bool:
    output = (result.stdout or b"") + b"\n" + (result.stderr or b"")
    lowered_output = output.lower()
    return bool(_TRANSIENT_HTTP_ERROR.search(lowered_output)) or any(
        marker in lowered_output for marker in _TRANSIENT_GIT_NETWORK_ERRORS
    )
