"""TDD coverage for reading indexable source directly from bare Git mirrors."""

from __future__ import annotations

from pathlib import Path
import subprocess

from pydantic import ValidationError
import pytest

from app.services.public_code.access_profiles import GitAccessProfileResolver
from app.services.public_code.git_mirror import GitMirrorReadError, GitMirrorSourceReader
from app.services.public_code.source_catalog import PublicCodeSource
from config import Settings


def run_git(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_mirror(tmp_path: Path) -> tuple[Path, str]:
    working_tree = tmp_path / "upstream"
    working_tree.mkdir()
    run_git("init", cwd=working_tree)
    run_git("config", "user.email", "retrieval-test@example.invalid", cwd=working_tree)
    run_git("config", "user.name", "Retrieval Test", cwd=working_tree)
    (working_tree / "src").mkdir()
    (working_tree / "vendor").mkdir()
    (working_tree / "docs").mkdir()
    (working_tree / "src" / "module.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (working_tree / "vendor" / "dependency.c").write_text("int add(int a, int b) { return a + b; }\n", encoding="utf-8")
    (working_tree / "docs" / "README.md").write_text("# Docs\n", encoding="utf-8")
    run_git("add", ".", cwd=working_tree)
    run_git("commit", "-m", "fixture", cwd=working_tree)
    revision = run_git("rev-parse", "HEAD", cwd=working_tree)

    mirror_root = tmp_path / "mirrors"
    destination = mirror_root / "fixture" / "sample.git"
    destination.parent.mkdir(parents=True)
    run_git("clone", "--mirror", str(working_tree), str(destination))
    return mirror_root, revision


def fixture_source() -> PublicCodeSource:
    return PublicCodeSource(
        source_id="sample",
        domain="fixture",
        relative_path="fixture/sample",
        upstream_url="https://example.invalid/sample.git",
    )


def test_manifest_reads_supported_tracked_files_without_checking_out_a_worktree(tmp_path):
    mirror_root, revision = create_mirror(tmp_path)
    reader = GitMirrorSourceReader(mirror_root=mirror_root)

    manifest = reader.build_manifest(fixture_source())

    assert manifest.revision_id == revision
    assert [file.path for file in manifest.files] == ["src/module.py", "vendor/dependency.c"]
    assert {file.path: file.language for file in manifest.files} == {
        "src/module.py": "python",
        "vendor/dependency.c": "c",
    }
    assert all(file.content_hash for file in manifest.files)


def test_manifest_skips_source_blobs_above_shared_code_kb_limit(tmp_path):
    from core.code_kb.chunker import MAX_FILE_BYTES

    working_tree = tmp_path / "oversized-upstream"
    working_tree.mkdir()
    run_git("init", cwd=working_tree)
    run_git("config", "user.email", "retrieval-test@example.invalid", cwd=working_tree)
    run_git("config", "user.name", "Retrieval Test", cwd=working_tree)
    (working_tree / "small.py").write_text("print('small')\n", encoding="utf-8")
    (working_tree / "generated.py").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    run_git("add", ".", cwd=working_tree)
    run_git("commit", "-m", "oversized fixture", cwd=working_tree)
    mirror_root = tmp_path / "mirrors"
    destination = mirror_root / "fixture" / "sample.git"
    destination.parent.mkdir(parents=True)
    run_git("clone", "--mirror", str(working_tree), str(destination))

    manifest = GitMirrorSourceReader(mirror_root=mirror_root).build_manifest(fixture_source())

    assert [file.path for file in manifest.files] == ["small.py"]


def test_manifest_skips_gitlink_entries_without_treating_missing_size_as_invalid(tmp_path):
    working_tree = tmp_path / "gitlink-upstream"
    working_tree.mkdir()
    run_git("init", cwd=working_tree)
    run_git("config", "user.email", "retrieval-test@example.invalid", cwd=working_tree)
    run_git("config", "user.name", "Retrieval Test", cwd=working_tree)
    (working_tree / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    run_git("add", "main.py", cwd=working_tree)
    run_git("commit", "-m", "source fixture", cwd=working_tree)
    gitlink_revision = run_git("rev-parse", "HEAD", cwd=working_tree)
    run_git(
        "update-index",
        "--add",
        "--cacheinfo",
        "160000",
        gitlink_revision,
        "vendor/external",
        cwd=working_tree,
    )
    run_git("commit", "-m", "gitlink fixture", cwd=working_tree)

    mirror_root = tmp_path / "mirrors"
    destination = mirror_root / "fixture" / "sample.git"
    destination.parent.mkdir(parents=True)
    run_git("clone", "--mirror", str(working_tree), str(destination))

    manifest = GitMirrorSourceReader(mirror_root=mirror_root).build_manifest(fixture_source())

    assert [file.path for file in manifest.files] == ["main.py"]


def test_refresh_rejects_origin_that_differs_from_catalog(tmp_path):
    mirror_root, _ = create_mirror(tmp_path)

    with pytest.raises(GitMirrorReadError, match="origin differs"):
        GitMirrorSourceReader(mirror_root=mirror_root).refresh_source(fixture_source())


def test_blob_content_uses_manifest_object_identity_and_retains_vendor_source(tmp_path):
    mirror_root, _ = create_mirror(tmp_path)
    reader = GitMirrorSourceReader(mirror_root=mirror_root)
    manifest = reader.build_manifest(fixture_source())
    vendor_file = next(file for file in manifest.files if file.path == "vendor/dependency.c")

    source = reader.read_source(fixture_source(), vendor_file)

    assert source == "int add(int a, int b) { return a + b; }\n"


def test_blob_content_decodes_legacy_source_comments(tmp_path):
    working_tree = tmp_path / "legacy-upstream"
    working_tree.mkdir()
    run_git("init", cwd=working_tree)
    run_git("config", "user.email", "retrieval-test@example.invalid", cwd=working_tree)
    run_git("config", "user.name", "Retrieval Test", cwd=working_tree)
    legacy_source = "// Copyright Hervé Brönnimann\nint value = 1;\n"
    (working_tree / "legacy.hpp").write_bytes(legacy_source.encode("cp1252"))
    run_git("add", ".", cwd=working_tree)
    run_git("commit", "-m", "legacy encoding fixture", cwd=working_tree)

    mirror_root = tmp_path / "mirrors"
    destination = mirror_root / "fixture" / "sample.git"
    destination.parent.mkdir(parents=True)
    run_git("clone", "--mirror", str(working_tree), str(destination))
    reader = GitMirrorSourceReader(mirror_root=mirror_root)
    manifest = reader.build_manifest(fixture_source())

    source = reader.read_source(fixture_source(), manifest.files[0])

    assert source == legacy_source


def test_source_language_override_distinguishes_octave_m_files(tmp_path):
    working_tree = tmp_path / "octave-upstream"
    working_tree.mkdir()
    run_git("init", cwd=working_tree)
    run_git("config", "user.email", "retrieval-test@example.invalid", cwd=working_tree)
    run_git("config", "user.name", "Retrieval Test", cwd=working_tree)
    (working_tree / "solver.m").write_text(
        "function y = solver(x)\ny = x + 1;\nend\n",
        encoding="utf-8",
    )
    run_git("add", ".", cwd=working_tree)
    run_git("commit", "-m", "octave fixture", cwd=working_tree)
    mirror_root = tmp_path / "mirrors"
    destination = mirror_root / "fixture" / "octave.git"
    destination.parent.mkdir(parents=True)
    run_git("clone", "--mirror", str(working_tree), str(destination))
    source = PublicCodeSource(
        source_id="octave",
        domain="numerical-environments",
        relative_path="fixture/octave",
        upstream_url="https://example.invalid/octave.git",
        language_overrides={".m": "matlab"},
    )

    manifest = GitMirrorSourceReader(mirror_root=mirror_root).build_manifest(source)

    assert {file.path: file.language for file in manifest.files} == {"solver.m": "matlab"}


def test_refresh_clones_missing_bare_mirror_and_indexes_configured_ref(tmp_path, monkeypatch):
    working_tree = tmp_path / "clone-upstream"
    working_tree.mkdir()
    run_git("init", cwd=working_tree)
    run_git("config", "user.email", "retrieval-test@example.invalid", cwd=working_tree)
    run_git("config", "user.name", "Retrieval Test", cwd=working_tree)
    (working_tree / "main.py").write_text("MAIN_VALUE = 1\n", encoding="utf-8")
    run_git("add", ".", cwd=working_tree)
    run_git("commit", "-m", "main fixture", cwd=working_tree)
    run_git("branch", "-M", "main", cwd=working_tree)
    run_git("checkout", "-b", "feature/index-this", cwd=working_tree)
    (working_tree / "feature.py").write_text("FEATURE_VALUE = 2\n", encoding="utf-8")
    run_git("add", ".", cwd=working_tree)
    run_git("commit", "-m", "feature fixture", cwd=working_tree)
    feature_revision = run_git("rev-parse", "HEAD", cwd=working_tree)

    captured_commands: list[tuple[list[str], dict[str, object]]] = []

    def command_runner(arguments, **kwargs):
        command = list(arguments)
        captured_commands.append((command, dict(kwargs)))
        if "clone" in command:
            command[command.index("https://example.invalid/sample.git")] = str(working_tree)
            result = subprocess.run(command, **kwargs)
            run_git(
                "remote",
                "set-url",
                "origin",
                "https://example.invalid/sample.git",
                cwd=Path(command[-1]),
            )
            return result
        if "fetch" in command:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        return subprocess.run(command, **kwargs)

    source = PublicCodeSource(
        source_id="sample",
        domain="fixture",
        relative_path="fixture/sample",
        upstream_url="https://example.invalid/sample.git",
        default_ref="refs/heads/feature/index-this",
        proxy_profile="local-proxy",
        credential_profile="fixture-read",
    )
    reader = GitMirrorSourceReader(
        mirror_root=tmp_path / "mirrors",
        timeout_seconds=7,
        network_timeout_seconds=123,
        access_profiles=GitAccessProfileResolver(
            default_proxy_url=None,
            environ={
                "PUBLIC_CODE_PROXY_PROFILE_LOCAL_PROXY": "http://127.0.0.1:10808",
                "PUBLIC_CODE_CREDENTIAL_PROFILE_FIXTURE_READ":
                    '{"allowed_hosts":["example.invalid"],'
                    '"authorization_header":"Bearer hidden-token"}',
            },
        ),
        command_runner=command_runner,
    )

    reader.refresh_source(source)
    manifest = reader.build_manifest(source)

    assert manifest.revision_id == feature_revision
    assert [item.path for item in manifest.files] == ["feature.py", "main.py"]
    clone_command, clone_options = captured_commands[0]
    clone_environment = clone_options["env"]
    assert "clone" in clone_command
    assert clone_options["timeout"] == 123
    assert "hidden-token" not in " ".join(clone_command)
    assert clone_environment["GIT_CONFIG_VALUE_0"] == "Authorization: Bearer hidden-token"

    local_command, local_options = next(
        (command, options)
        for command, options in captured_commands
        if "remote" in command and "get-url" in command
    )
    assert "remote" in local_command
    assert local_options["timeout"] == 7


def test_failed_clone_removes_only_the_staged_candidate_mirror(tmp_path):
    mirror_root = tmp_path / "mirrors"
    source = PublicCodeSource(
        source_id="sample",
        domain="fixture",
        relative_path="fixture/sample",
        upstream_url="https://example.invalid/sample.git",
    )

    def failing_runner(arguments, **kwargs):
        destination = Path(arguments[-1])
        destination.mkdir(parents=True)
        (destination / "partial").write_text("incomplete", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 1, stdout=b"", stderr=b"clone failed")

    reader = GitMirrorSourceReader(
        mirror_root=mirror_root,
        proxy_url=None,
        command_runner=failing_runner,
    )

    with pytest.raises(GitMirrorReadError, match="cannot clone bare mirror"):
        reader.refresh_source(source)

    assert not (mirror_root / "fixture" / "sample.git").exists()
    assert not (mirror_root / "fixture" / "sample.git.candidate").exists()


@pytest.mark.parametrize(
    "transient_error",
    [
        (
            b"error: RPC failed; curl 56 GnuTLS recv error (-54): "
            b"Error in the pull function.\nfatal: early EOF\n"
        ),
        b"fatal: unable to access upstream: The requested URL returned error: 503\n",
        b"fatal: unable to access upstream: Could not resolve proxy: proxy.example\n",
        b"fatal: unable to access upstream: CONNECT tunnel failed, response 502\n",
    ],
)
def test_clone_retries_transient_transport_failure_after_cleaning_candidate(
    tmp_path,
    transient_error,
):
    mirror_root = tmp_path / "mirrors"
    source = PublicCodeSource(
        source_id="sample",
        domain="fixture",
        relative_path="fixture/sample",
        upstream_url="https://example.invalid/sample.git",
    )
    clone_attempts = 0
    retry_delays: list[float] = []

    def flaky_runner(arguments, **kwargs):
        nonlocal clone_attempts
        command = list(arguments)
        if "clone" in command:
            clone_attempts += 1
            destination = Path(command[-1])
            if clone_attempts == 2:
                assert not destination.exists()
            destination.mkdir(parents=True)
            if clone_attempts == 1:
                (destination / "partial").write_text("incomplete", encoding="utf-8")
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout=b"",
                    stderr=transient_error,
                )
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if "get-url" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{source.upstream_url}\n".encode(),
                stderr=b"",
            )
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    reader = GitMirrorSourceReader(
        mirror_root=mirror_root,
        proxy_url=None,
        network_retry_attempts=3,
        network_retry_backoff_seconds=0.25,
        retry_sleep=retry_delays.append,
        command_runner=flaky_runner,
    )

    reader.refresh_source(source)

    assert clone_attempts == 2
    assert retry_delays == [0.25]
    assert (mirror_root / "fixture" / "sample.git").is_dir()
    assert not (mirror_root / "fixture" / "sample.git.candidate").exists()


def test_fetch_retries_transient_transport_failure_without_recloning(tmp_path):
    mirror_root, _ = create_mirror(tmp_path)
    source = fixture_source()
    fetch_attempts = 0
    retry_delays: list[float] = []

    def flaky_runner(arguments, **kwargs):
        nonlocal fetch_attempts
        command = list(arguments)
        if "get-url" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{source.upstream_url}\n".encode(),
                stderr=b"",
            )
        if "fetch" in command:
            fetch_attempts += 1
            if fetch_attempts == 1:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout=b"",
                    stderr=b"fatal: the remote end hung up unexpectedly\n",
                )
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        return subprocess.run(command, **kwargs)

    reader = GitMirrorSourceReader(
        mirror_root=mirror_root,
        network_retry_attempts=3,
        network_retry_backoff_seconds=0.5,
        retry_sleep=retry_delays.append,
        command_runner=flaky_runner,
    )

    reader.refresh_source(source)

    assert fetch_attempts == 2
    assert retry_delays == [0.5]


@pytest.mark.parametrize(
    ("setting_name", "invalid_value"),
    [
        ("PUBLIC_CODE__GIT_COMMAND_TIMEOUT_SECONDS", "0"),
        ("PUBLIC_CODE__GIT_NETWORK_TIMEOUT_SECONDS", "-1"),
        ("PUBLIC_CODE__GIT_NETWORK_TIMEOUT_SECONDS", "inf"),
        ("PUBLIC_CODE__GIT_NETWORK_RETRY_BACKOFF_SECONDS", "nan"),
    ],
)
def test_settings_reject_invalid_git_network_numbers(monkeypatch, setting_name, invalid_value):
    monkeypatch.setenv(setting_name, invalid_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("setting_name", "invalid_value"),
    [
        ("timeout_seconds", 0),
        ("network_timeout_seconds", float("inf")),
        ("network_retry_backoff_seconds", float("nan")),
    ],
)
def test_reader_rejects_invalid_git_network_numbers(tmp_path, setting_name, invalid_value):
    arguments = {setting_name: invalid_value}

    with pytest.raises(ValueError, match="finite"):
        GitMirrorSourceReader(mirror_root=tmp_path / "mirrors", **arguments)


def test_clone_timeout_removes_the_staged_candidate_mirror(tmp_path):
    mirror_root = tmp_path / "mirrors"
    source = PublicCodeSource(
        source_id="sample",
        domain="fixture",
        relative_path="fixture/sample",
        upstream_url="https://example.invalid/sample.git",
    )

    def timeout_runner(arguments, **kwargs):
        destination = Path(arguments[-1])
        destination.mkdir(parents=True)
        (destination / "partial").write_text("incomplete", encoding="utf-8")
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

    reader = GitMirrorSourceReader(
        mirror_root=mirror_root,
        proxy_url=None,
        network_timeout_seconds=123,
        network_retry_attempts=1,
        command_runner=timeout_runner,
    )

    with pytest.raises(GitMirrorReadError, match="timed out after 123"):
        reader.refresh_source(source)

    assert not (mirror_root / "fixture" / "sample.git").exists()
    assert not (mirror_root / "fixture" / "sample.git.candidate").exists()
