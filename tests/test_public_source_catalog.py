"""Validation for the local first-wave public repository mirror catalog."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "public_sources"
    / "first_wave_repositories.json"
)
CLONE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sync"
    / "clone_public_source_repositories.ps1"
)
MINIMUM_CURATED_REPOSITORIES = 75
EXCLUDED_OVERSIZED_SOURCES = frozenset({"chromium", "freebsd", "gcc", "rust"})


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_first_wave_catalog_has_unique_safe_repository_locations():
    catalog = load_catalog()
    repositories = catalog["repositories"]

    source_ids = [repository["source_id"] for repository in repositories]
    relative_paths = [repository["relative_path"] for repository in repositories]

    assert catalog["catalog_id"] == "public-code-first-wave"
    assert len(repositories) >= MINIMUM_CURATED_REPOSITORIES
    assert len(source_ids) == len(set(source_ids))
    assert EXCLUDED_OVERSIZED_SOURCES.isdisjoint(source_ids)
    assert len(relative_paths) == len(set(relative_paths))
    assert all(not Path(path).is_absolute() for path in relative_paths)
    assert all(".." not in Path(path).parts for path in relative_paths)
    assert all(repository["upstream_url"].startswith("https://") for repository in repositories)


def test_first_wave_catalog_covers_each_foundation_domain():
    catalog = load_catalog()
    repositories = {repository["source_id"]: repository for repository in catalog["repositories"]}

    required_sources = {
        "linux",
        "ffmpeg",
        "sqlite",
        "llvm-project",
        "pytorch",
        "triton",
        "cutlass",
        "mpich",
        "openmpi",
        "openblas",
        "petsc",
        "mfem",
        "julia",
        "octave",
        "pion-webrtc",
        "svt-av1",
        "jax",
        "cuda-samples",
        "oneTBB",
        "lapack",
    }

    assert required_sources <= repositories.keys()
    assert repositories["svt-av1"]["upstream_url"] == "https://gitlab.com/AOMediaCodec/SVT-AV1.git"
    assert repositories["x264"]["upstream_url"] == "https://code.videolan.org/videolan/x264.git"
    assert repositories["pion-webrtc"]["upstream_url"] == "https://github.com/pion/webrtc.git"
    assert repositories["eigen"]["upstream_url"] == "https://gitlab.com/libeigen/eigen.git"
    assert {repository["domain"] for repository in repositories.values()} >= {
        "systems",
        "multimedia",
        "compilers",
        "machine-learning",
        "gpu",
        "parallel",
        "scientific-computing",
        "numerical-environments",
    }


def test_clone_script_selects_the_full_catalog_without_source_ids(tmp_path):
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(CLONE_SCRIPT_PATH),
            "-DestinationRoot",
            str(tmp_path / "sources"),
            "-WhatIf",
        ],
        check=False,
        cwd=CATALOG_PATH.parents[2],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr
    assert "Mirror linux" in result.stdout
    assert "Mirror scilab" in result.stdout
    assert str(tmp_path / "sources" / "systems" / "linux.git") in result.stdout


def test_clone_script_uses_bare_mirrors_to_preserve_case_sensitive_git_paths():
    script = CLONE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "clone --mirror --depth 1 --no-tags" in script
    assert "is-bare-repository" in script
    assert "Destination exists but is not a bare Git mirror" in script


def test_clone_script_creates_two_local_bare_mirrors_with_configured_concurrency(tmp_path):
    source_root = tmp_path / "upstreams"
    repositories = []
    for source_id in ("alpha", "beta"):
        repository = source_root / source_id
        repository.mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "retrieval-test@example.invalid"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Retrieval Test"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
        (repository / f"{source_id}.txt").write_text(source_id, encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "fixture"], cwd=repository, check=True, capture_output=True
        )
        repositories.append(
            {
                "source_id": source_id,
                "domain": "fixture",
                "relative_path": f"fixture/{source_id}",
                "upstream_url": repository.as_uri(),
            }
        )

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"catalog_id": "public-code-first-wave", "repositories": repositories}),
        encoding="utf-8",
    )
    destination_root = tmp_path / "mirrors"
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(CLONE_SCRIPT_PATH),
            "-DestinationRoot",
            str(destination_root),
            "-CatalogPath",
            str(catalog_path),
            "-MaxConcurrentMirrors",
            "2",
        ],
        check=False,
        cwd=CATALOG_PATH.parents[2],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr
    assert "concurrency 2" in result.stdout
    for repository in repositories:
        mirror = destination_root / f"{repository['relative_path']}.git"
        is_bare = subprocess.run(
            ["git", "-C", str(mirror), "rev-parse", "--is-bare-repository"],
            check=True,
            capture_output=True,
            text=True,
        )
        origin = subprocess.run(
            ["git", "-C", str(mirror), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert is_bare.stdout.strip() == "true"
        assert origin.stdout.strip() == repository["upstream_url"]


def test_clone_script_rejects_a_bare_mirror_without_a_verified_head_commit(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    target = tmp_path / "mirrors" / "fixture" / "incomplete.git"
    target.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "--bare", str(target)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(target), "remote", "add", "origin", upstream.as_uri()],
        check=True,
        capture_output=True,
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "public-code-first-wave",
                "repositories": [
                    {
                        "source_id": "incomplete",
                        "domain": "fixture",
                        "relative_path": "fixture/incomplete",
                        "upstream_url": upstream.as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(CLONE_SCRIPT_PATH),
            "-DestinationRoot",
            str(tmp_path / "mirrors"),
            "-CatalogPath",
            str(catalog_path),
            "-WhatIf",
        ],
        check=False,
        cwd=CATALOG_PATH.parents[2],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert "bare Git mirror has no verified HEAD commit" in result.stderr


def test_clone_script_reports_a_non_git_destination_without_null_output_errors(tmp_path):
    target = tmp_path / "mirrors" / "fixture" / "not-a-repository.git"
    target.mkdir(parents=True)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "public-code-first-wave",
                "repositories": [
                    {
                        "source_id": "not-a-repository",
                        "domain": "fixture",
                        "relative_path": "fixture/not-a-repository",
                        "upstream_url": "https://example.invalid/not-a-repository.git",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(CLONE_SCRIPT_PATH),
            "-DestinationRoot",
            str(tmp_path / "mirrors"),
            "-CatalogPath",
            str(catalog_path),
            "-WhatIf",
        ],
        check=False,
        cwd=CATALOG_PATH.parents[2],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert "Destination exists but is not a bare Git mirror" in result.stderr
    assert "null-valued expression" not in result.stderr


def test_clone_script_rejects_catalog_path_escape(tmp_path):
    destination_root = tmp_path / "mirrors"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({
            "catalog_id": "public-code-first-wave",
            "repositories": [{
                "source_id": "escape",
                "domain": "fixture",
                "relative_path": "../outside",
                "upstream_url": "https://example.invalid/escape.git",
            }],
        }),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(CLONE_SCRIPT_PATH),
            "-DestinationRoot", str(destination_root),
            "-CatalogPath", str(catalog_path),
            "-WhatIf",
        ],
        check=False,
        cwd=CATALOG_PATH.parents[2],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert "relative_path" in result.stderr
    assert not (tmp_path / "outside.git").exists()
