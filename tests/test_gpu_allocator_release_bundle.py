"""The `.88` allocator deployment must use an explicit, hash-checked delta."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def bundle_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_gpu_allocator_release_bundle.py"
    spec = importlib.util.spec_from_file_location("gpu_allocator_release_bundle_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The release list names private notes; the public CI snapshot leaves them out. Skip
    # (visibly) only when exactly those are missing — any other missing file still fails.
    private_notes = {"env.md", "overview.md", "INDEX.md", "plan.md", "journal.md"}
    missing = {line.strip() for line in module.FILE_LIST_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.strip().startswith("#")
               and not (module.PROJECT_ROOT / line.strip()).is_file()}
    if missing and missing <= private_notes:
        pytest.skip("private doc, excluded from the public CI snapshot "
                    "(ci_snapshot.PUBLIC_EXCLUDE_PATHSPECS)")
    return module


def test_release_delta_has_an_explicit_allow_list_and_excludes_main(bundle_module):
    paths = bundle_module._release_files()

    assert Path("core/gpu_memory.py") in paths
    assert Path("deploy/compute88/arc-retrival-backend.service") in paths
    assert Path("main.py") not in paths


def test_release_bundle_is_hash_verified_before_upload(tmp_path, bundle_module):
    bundle = tmp_path / "gpu-allocator-release.tar.gz"

    built = bundle_module.build_bundle(bundle)
    verified = bundle_module.verify_bundle(bundle)

    assert built["manifest"]["files"]
    assert verified["file_count"] == len(built["manifest"]["files"])
    with pytest.raises(FileExistsError):
        bundle_module.build_bundle(bundle)
