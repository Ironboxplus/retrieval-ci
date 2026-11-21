"""Read-only public repository update checks.

An update check reads the local bare-mirror revision and probes the remote
``HEAD`` ref only.  It deliberately does not fetch, checkout, hash, or invoke
the indexing pipeline; those are candidate-update job responsibilities.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable, Protocol

from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.source_catalog import PublicCodeSource, PublicCodeSourceSelector


@dataclass(frozen=True, slots=True)
class SourceRevisionSnapshot:
    """The currently mirrored commit and the remote commit observed by a check."""

    local_revision: str | None
    remote_revision: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PublicCodeUpdateCheck:
    """A source-facing update state suitable for API and background-job callers."""

    source_id: str
    domain: str
    local_revision: str | None
    remote_revision: str | None
    status: str
    message: str | None = None


class SourceRevisionProbe(Protocol):
    """Infrastructure boundary for inspecting one source without mutating it."""

    def inspect(self, source: PublicCodeSource) -> SourceRevisionSnapshot:
        """Return the local and remote revisions for ``source``."""


class PublicCodeUpdateCheckService:
    """Apply catalog selection and normalize update-check results for callers."""

    def __init__(
        self,
        *,
        source_registry: PublicCodeSourceSelector,
        revision_probe: SourceRevisionProbe,
        max_workers: int = 2,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._source_registry = source_registry
        self._revision_probe = revision_probe
        self._max_workers = max_workers

    def check(
        self,
        *,
        source_ids: Iterable[str] = (),
        domains: Iterable[str] = (),
        progress_cb: Callable[[PublicCodeUpdateCheck, int, int], None] | None = None,
    ) -> tuple[PublicCodeUpdateCheck, ...]:
        """Check one, many, or all selected sources without initiating updates."""

        selected_sources = self._source_registry.select(source_ids=source_ids, domains=domains)
        if len(selected_sources) <= 1:
            checks = tuple(self._check_source(source) for source in selected_sources)
            if progress_cb is not None:
                for completed, check in enumerate(checks, start=1):
                    progress_cb(check, completed, len(checks))
            return checks
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(selected_sources)),
            thread_name_prefix="public-code-update-check",
        ) as executor:
            futures = {
                executor.submit(self._check_source, source): index
                for index, source in enumerate(selected_sources)
            }
            ordered_checks: list[PublicCodeUpdateCheck | None] = [None] * len(selected_sources)
            for completed, future in enumerate(as_completed(futures), start=1):
                check = future.result()
                ordered_checks[futures[future]] = check
                if progress_cb is not None:
                    progress_cb(check, completed, len(selected_sources))
        return tuple(check for check in ordered_checks if check is not None)

    def _check_source(self, source: PublicCodeSource) -> PublicCodeUpdateCheck:
        try:
            snapshot = self._revision_probe.inspect(source)
        except Exception as exc:  # Infrastructure failures must not hide other source states.
            return PublicCodeUpdateCheck(
                source_id=source.source_id,
                domain=source.domain,
                local_revision=None,
                remote_revision=None,
                status="check_failed",
                message=str(exc),
            )

        status, message = _classify_revision_snapshot(snapshot)
        return PublicCodeUpdateCheck(
            source_id=source.source_id,
            domain=source.domain,
            local_revision=snapshot.local_revision,
            remote_revision=snapshot.remote_revision,
            status=status,
            message=message,
        )


class GitMirrorRevisionProbe:
    """Subprocess adapter for a local bare mirror and its upstream ``HEAD`` ref."""

    def __init__(
        self,
        *,
        mirror_root: Path,
        proxy_url: str | None = None,
        timeout_seconds: float = 20.0,
        access_profiles: GitAccessProfileResolver | None = None,
        command_runner: Callable[..., Any] = subprocess.run,
    ):
        self._mirror_root = mirror_root
        self._timeout_seconds = timeout_seconds
        self._access_profiles = access_profiles or GitAccessProfileResolver(
            default_proxy_url=proxy_url,
        )
        self._command_runner = command_runner

    def inspect(self, source: PublicCodeSource) -> SourceRevisionSnapshot:
        local_revision = self._read_local_revision(source, self._mirror_path(source))
        try:
            remote_revision = self._read_remote_revision(source)
        except RuntimeError as exc:
            return SourceRevisionSnapshot(
                local_revision=local_revision,
                remote_revision=None,
                error=str(exc),
            )
        return SourceRevisionSnapshot(local_revision=local_revision, remote_revision=remote_revision)

    def _mirror_path(self, source: PublicCodeSource) -> Path:
        return self._mirror_root.joinpath(*source.relative_path.split("/")).with_suffix(".git")

    def _read_local_revision(self, source: PublicCodeSource, mirror_path: Path) -> str | None:
        if not mirror_path.is_dir():
            return None
        result = self._run_git(
            source,
            "-C",
            str(mirror_path),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{source.default_ref}^{{commit}}",
            network=False,
        )
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None

    def _read_remote_revision(self, source: PublicCodeSource) -> str:
        result = self._run_git(
            source,
            "ls-remote",
            source.upstream_url,
            source.default_ref,
            network=True,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "git ls-remote failed"
            raise RuntimeError(details)
        fields = result.stdout.strip().split()
        if not fields:
            raise RuntimeError("remote HEAD did not resolve to a commit")
        return fields[0]

    def _run_git(
        self,
        source: PublicCodeSource,
        *arguments: str,
        network: bool,
    ) -> subprocess.CompletedProcess[str]:
        environment = self._access_profiles.environment_for(source, network=network)
        return self._command_runner(
            ["git", *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
            env=environment,
            check=False,
        )


def _classify_revision_snapshot(snapshot: SourceRevisionSnapshot) -> tuple[str, str | None]:
    if snapshot.error:
        return "check_failed", snapshot.error
    if snapshot.remote_revision is None:
        return "check_failed", "remote HEAD did not resolve to a commit"
    if snapshot.local_revision is None:
        return "not_mirrored", "No verified local bare mirror commit"
    if snapshot.local_revision == snapshot.remote_revision:
        return "up_to_date", None
    return "update_available", None
