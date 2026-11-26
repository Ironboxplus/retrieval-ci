"""TDD contracts for public-code source selection and update checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import time

import pytest

from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.source_catalog import (
    PublicCodeSource,
    PublicCodeSourceCatalog,
    PublicSourceSelectionError,
)
from app.services.public_code.update_checks import (
    GitMirrorRevisionProbe,
    PublicCodeUpdateCheckService,
    SourceRevisionSnapshot,
)


def source(source_id: str, domain: str, path: str) -> PublicCodeSource:
    return PublicCodeSource(
        source_id=source_id,
        domain=domain,
        relative_path=path,
        upstream_url=f"https://example.invalid/{source_id}.git",
    )


def catalog() -> PublicCodeSourceCatalog:
    return PublicCodeSourceCatalog(
        catalog_id="fixture",
        sources=(
            source("linux", "systems", "systems/linux"),
            source("ffmpeg", "multimedia", "multimedia/ffmpeg"),
            source("cutlass", "gpu", "gpu/cutlass"),
            source("mpich", "parallel", "parallel/mpich"),
        ),
    )


def test_selects_all_sources_when_no_source_or_domain_filter_is_supplied():
    assert [item.source_id for item in catalog().select()] == [
        "cutlass",
        "ffmpeg",
        "linux",
        "mpich",
    ]


def test_source_identity_preserves_canonical_mixed_case_project_names():
    configured = source("oneTBB", "parallel", "parallel/oneTBB")

    assert configured.source_id == "oneTBB"


def test_relative_path_rejects_git_storage_suffix_aliases():
    with pytest.raises(ValueError, match="must omit the .git storage suffix"):
        source("sample", "fixture", "fixture/sample.git")


def test_selects_union_of_explicit_sources_and_multiple_domains():
    selected = catalog().select(source_ids=("linux",), domains=("gpu", "parallel"))

    assert [item.source_id for item in selected] == ["cutlass", "linux", "mpich"]


def test_selection_rejects_unknown_source_or_domain_instead_of_silently_narrowing_scope():
    with pytest.raises(PublicSourceSelectionError, match="Unknown public-code source IDs: absent"):
        catalog().select(source_ids=("absent",))

    with pytest.raises(PublicSourceSelectionError, match="Unknown public-code domains: missing"):
        catalog().select(domains=("missing",))


@dataclass
class FakeRevisionProbe:
    snapshots: dict[str, SourceRevisionSnapshot]

    def inspect(self, source: PublicCodeSource) -> SourceRevisionSnapshot:
        return self.snapshots[source.source_id]


def test_update_check_classifies_mirror_state_without_fetching_or_indexing():
    service = PublicCodeUpdateCheckService(
        source_registry=catalog(),
        revision_probe=FakeRevisionProbe(
            {
                "linux": SourceRevisionSnapshot(local_revision="same", remote_revision="same"),
                "ffmpeg": SourceRevisionSnapshot(local_revision="old", remote_revision="new"),
                "cutlass": SourceRevisionSnapshot(local_revision=None, remote_revision="new"),
                "mpich": SourceRevisionSnapshot(local_revision="known", remote_revision=None, error="remote unavailable"),
            }
        ),
    )

    checks = service.check(domains=("systems", "multimedia", "gpu", "parallel"))

    assert [(check.source_id, check.status) for check in checks] == [
        ("cutlass", "not_mirrored"),
        ("ffmpeg", "update_available"),
        ("linux", "up_to_date"),
        ("mpich", "check_failed"),
    ]
    assert checks[1].remote_revision == "new"
    assert checks[3].message == "remote unavailable"


def test_update_checks_use_bounded_parallel_probes_without_reordering_results():
    lock = threading.Lock()
    active = 0
    max_active = 0

    class Probe:
        def inspect(self, source):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return SourceRevisionSnapshot(local_revision=source.source_id, remote_revision=source.source_id)

    service = PublicCodeUpdateCheckService(
        source_registry=catalog(),
        revision_probe=Probe(),
        max_workers=2,
    )

    checks = service.check(domains=("systems", "multimedia", "gpu", "parallel"))

    assert [check.source_id for check in checks] == ["cutlass", "ffmpeg", "linux", "mpich"]
    assert max_active == 2


def test_git_revision_probe_uses_source_ref_and_server_side_access_profiles(tmp_path):
    mirror_path = tmp_path / "mirrors" / "fixture" / "sample.git"
    mirror_path.mkdir(parents=True)
    commands: list[tuple[list[str], dict[str, str]]] = []

    def command_runner(arguments, **kwargs):
        command = list(arguments)
        commands.append((command, dict(kwargs["env"])))
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout="local-revision\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="remote-revision\trefs/heads/stable\n",
            stderr="",
        )

    configured_source = PublicCodeSource(
        source_id="sample",
        domain="fixture",
        relative_path="fixture/sample",
        upstream_url="https://example.invalid/sample.git",
        default_ref="refs/heads/stable",
        proxy_profile="local-proxy",
    )
    probe = GitMirrorRevisionProbe(
        mirror_root=tmp_path / "mirrors",
        access_profiles=GitAccessProfileResolver(
            default_proxy_url=None,
            environ={"PUBLIC_CODE_PROXY_PROFILE_LOCAL_PROXY": "http://127.0.0.1:10808"},
        ),
        command_runner=command_runner,
    )

    snapshot = probe.inspect(configured_source)

    assert snapshot.local_revision == "local-revision"
    assert snapshot.remote_revision == "remote-revision"
    assert any("refs/heads/stable^{commit}" in command for command, _ in commands)
    assert any(command[-1] == "refs/heads/stable" for command, _ in commands)
    local_environment = next(environment for command, environment in commands if "rev-parse" in command)
    remote_environment = next(environment for command, environment in commands if "ls-remote" in command)
    assert "https_proxy" not in local_environment
    assert remote_environment["https_proxy"] == "http://127.0.0.1:10808"
