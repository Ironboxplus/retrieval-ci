"""Build and verify the exact `.88` GPU allocator release delta.

The working tree intentionally contains independent unfinished work. This tool
creates one hash-verified tarball from the explicit allow-list rather than
copying a directory or relying on tracked-file discovery. It excludes main.py
and cannot silently pick up unrelated files.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_LIST_PATH = PROJECT_ROOT / "deploy" / "compute88" / "gpu_allocator_release_files.txt"
ARCHIVE_ROOT = "retrieval-gpu-allocator-release"
MANIFEST_NAME = f"{ARCHIVE_ROOT}/manifest.json"


def _release_files() -> tuple[Path, ...]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_line in FILE_LIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        relative = Path(line)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("main.py"):
            raise ValueError(f"unsafe or excluded release path: {line}")
        if relative in seen:
            raise ValueError(f"duplicate release path: {line}")
        source = (PROJECT_ROOT / relative).resolve()
        if not source.is_file() or PROJECT_ROOT not in source.parents:
            raise ValueError(f"release path is not a project file: {line}")
        seen.add(relative)
        paths.append(relative)
    if not paths:
        raise ValueError("release file list is empty")
    return tuple(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_for(paths: tuple[Path, ...]) -> dict[str, Any]:
    files = []
    for relative in paths:
        source = PROJECT_ROOT / relative
        files.append({
            "path": relative.as_posix(),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        })
    return {
        "schema_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "files": files,
    }


def build_bundle(output: Path) -> dict[str, Any]:
    """Create a new atomic, hash-described release tarball without overwrite."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing release bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = _release_files()
    manifest = _manifest_for(paths)
    temporary = output.with_name(f"{output.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"remove or inspect stale partial release bundle: {temporary}")
    try:
        with tarfile.open(temporary, mode="w:gz") as archive:
            for relative in paths:
                archive.add(
                    PROJECT_ROOT / relative,
                    arcname=f"{ARCHIVE_ROOT}/{relative.as_posix()}",
                    recursive=False,
                )
            payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"bundle": str(output), "manifest": manifest}


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Check every archived member against its internal manifest before upload."""
    with tarfile.open(bundle, mode="r:gz") as archive:
        manifest_member = archive.getmember(MANIFEST_NAME)
        manifest_stream = archive.extractfile(manifest_member)
        if manifest_stream is None:
            raise ValueError("release bundle manifest is unreadable")
        manifest = json.loads(manifest_stream.read().decode("utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("archive_root") != ARCHIVE_ROOT:
            raise ValueError("release bundle manifest schema is unsupported")
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("release bundle manifest contains no files")
        expected_names = {MANIFEST_NAME}
        for entry in entries:
            path = entry.get("path")
            if not isinstance(path, str) or path == "main.py" or path.startswith("/") or ".." in Path(path).parts:
                raise ValueError("release bundle manifest contains an unsafe path")
            expected_names.add(f"{ARCHIVE_ROOT}/{path}")
        actual_names = {member.name for member in archive.getmembers()}
        if actual_names != expected_names:
            raise ValueError("release bundle members differ from its explicit manifest")
        for entry in entries:
            member = archive.getmember(f"{ARCHIVE_ROOT}/{entry['path']}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"release bundle member is unreadable: {entry['path']}")
            payload = stream.read()
            if len(payload) != entry.get("size_bytes") or hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                raise ValueError(f"release bundle hash mismatch: {entry['path']}")
    return {"bundle": str(bundle), "file_count": len(entries), "sha256": _sha256(bundle)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--bundle", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_bundle(args.bundle) if args.mode == "build" else verify_bundle(args.bundle)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
