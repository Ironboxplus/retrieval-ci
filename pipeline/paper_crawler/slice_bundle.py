"""Hand a conference slice's PDFs or Markdown to the ingest host as one verified tar.

The files of a (venue, year) slice are produced on another machine (a
downloading host with the proxy pool, a GPU box that ran the marker). They cross
over as a bundle and land in the ingest host's own corpus tree:

  bundle   <conf>_<year>_<kind>.tar = manifest.json + files/<name>
           manifest.json lists every file with its sha256 and size
  push     the producer builds the bundle and copies it into the ingest host's
           inbox (``InboxTransfer``: written as .part, then renamed)
  receive  the ingest pipeline's download / extract stage waits for the bundle,
           verifies every member against the manifest, then lands the files
           (``import_slice_bundle``): all are staged first and renamed into place
           only when the whole bundle verified, and ``.received.json`` beside
           them records which bundle, from which host, brought each file

A PDF bundle also carries each PDF's ``<stem>.provenance.json`` sidecar (what the
downloader recorded: source URL, full-text source and check), because the slice
manifest sync folds exactly that file into the record before preindex.

A bundle that fails verification is renamed ``.rejected``, so a retry waits for
a fresh push instead of failing on the same bytes again.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import socket
import tarfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pipeline.paper_crawler.inbox_transfer import Executable, InboxTransfer
from pipeline.paper_crawler.runtime_files import is_valid_pdf_artifact

MANIFEST_MEMBER = "manifest.json"
FILES_PREFIX = "files/"
RECEIPT_FILE = ".received.json"
SIDECAR_SUFFIX = ".provenance.json"  # ccf_a_main_corpus.provenance_path: beside the PDF
KIND_SUFFIX = {"pdf": ".pdf", "md": ".md"}
FORMAT = 1
# One path component that is valid on every filesystem the corpus lives on (the
# corpus's own file names are alphanumerics plus "-._"; nothing here needs more).
_SAFE_NAME = re.compile(r'^[^.\x00-\x1f/\\:*?"<>|][^\x00-\x1f/\\:*?"<>|]*$')


def bundle_file_name(conf_lower: str, year: int, kind: str) -> str:
    return f"{conf_lower}_{year}_{kind}.tar"


def inbox_dir(corpus_root: Path) -> Path:
    return corpus_root / "inbox"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_kind(kind: str) -> str:
    if kind not in KIND_SUFFIX:
        raise ValueError(f"bundle kind must be one of {sorted(KIND_SUFFIX)}, got {kind!r}")
    return KIND_SUFFIX[kind]


def _is_sidecar(name: str, kind: str) -> bool:
    return kind == "pdf" and name.endswith(SIDECAR_SUFFIX)


def _check_name(name: str, kind: str) -> None:
    if name != Path(name).name or not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(f"unsafe file name in bundle: {name!r}")
    if not (name.endswith(_check_kind(kind)) or _is_sidecar(name, kind)):
        raise ValueError(f"{name!r} is not a {kind} file")


# ------------------------------------------------------------------------ write
def write_slice_bundle(files: list[Path], out: Path, *, venue: str, year: int, kind: str,
                       source: Optional[str] = None) -> dict:
    """Write ``files`` as one bundle at ``out`` (as .part, then renamed); returns its manifest."""
    if not files:
        raise ValueError(f"no {kind} files to bundle for {venue} {year}")
    entries = []
    for f in sorted(files, key=lambda p: p.name):
        _check_name(f.name, kind)
        entries.append({"name": f.name, "sha256": sha256_file(f), "size": f.stat().st_size})
    manifest = {
        "format": FORMAT, "kind": kind, "venue": venue, "year": year,
        "source": source or socket.gethostname(),
        "created_at": datetime.now(timezone.utc).isoformat(), "files": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with tarfile.open(tmp, "w") as tar:
        payload = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_MEMBER)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        for f in sorted(files, key=lambda p: p.name):
            tar.add(f, arcname=FILES_PREFIX + f.name)
    os.replace(tmp, out)
    return manifest


def push_slice_files(source_dir: Path, outbox: Path, transfer: InboxTransfer, *, venue: str, year: int,
                     kind: str, conf_lower: str, source: Optional[str] = None,
                     only: Optional[set[str]] = None) -> dict:
    """Bundle the slice's files in ``source_dir`` and copy the bundle into the ingest host's inbox."""
    suffix = _check_kind(kind)
    files = sorted(p for p in source_dir.glob(f"*{suffix}") if only is None or p.name in only)
    if kind == "pdf":
        files += [side for f in list(files) if (side := f.with_suffix(SIDECAR_SUFFIX)).is_file()]
    name = bundle_file_name(conf_lower, year, kind)
    local = outbox / name
    manifest = write_slice_bundle(files, local, venue=venue, year=year, kind=kind, source=source)
    size = local.stat().st_size
    transfer.send(local, name)
    local.unlink()
    return {"bundle": name, "files": len(manifest["files"]), "bytes": size}


# ----------------------------------------------------------------------- verify
def read_manifest(bundle: Path) -> dict:
    with tarfile.open(bundle, "r") as tar:
        try:
            member = tar.getmember(MANIFEST_MEMBER)
        except KeyError:
            raise ValueError(f"{bundle.name}: no {MANIFEST_MEMBER}") from None
        manifest = json.loads(tar.extractfile(member).read().decode("utf-8"))
    if manifest.get("format") != FORMAT or not isinstance(manifest.get("files"), list):
        raise ValueError(f"{bundle.name}: unsupported manifest")
    return manifest


def verify_slice_bundle(bundle: Path, *, venue: str, year: int, kind: str, stage_dir: Path) -> dict:
    """Extract every member into ``stage_dir`` and check it against the manifest.

    Raises ValueError (nothing is landed) when the bundle is for another slice or
    kind, has a member the manifest does not list or lists a member it lacks, has
    an unsafe name, or any sha256/size differs. Returns the manifest."""
    manifest = read_manifest(bundle)
    if (str(manifest.get("venue")).lower(), manifest.get("year"), manifest.get("kind")) != (venue.lower(), year, kind):
        raise ValueError(
            f"{bundle.name} holds {manifest.get('venue')} {manifest.get('year')} {manifest.get('kind')}, "
            f"expected {venue} {year} {kind}")
    listed = {}
    for entry in manifest["files"]:
        _check_name(entry["name"], kind)
        listed[entry["name"]] = entry
    if len(listed) != len(manifest["files"]):
        raise ValueError(f"{bundle.name}: duplicate names in the manifest")
    stage_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with tarfile.open(bundle, "r") as tar:
        for m in tar.getmembers():
            if m.name == MANIFEST_MEMBER:
                continue
            if not (m.isfile() and m.name.startswith(FILES_PREFIX)):
                raise ValueError(f"{bundle.name}: unexpected member {m.name!r}")
            name = m.name[len(FILES_PREFIX):]
            if name not in listed or name in seen:
                raise ValueError(f"{bundle.name}: member {name!r} is not in the manifest")
            _check_name(name, kind)
            seen.add(name)
            target = stage_dir / name
            digest = hashlib.sha256()
            with tar.extractfile(m) as src, open(target, "wb") as dst:
                for block in iter(lambda: src.read(1 << 20), b""):
                    digest.update(block)
                    dst.write(block)
            entry = listed[name]
            if target.stat().st_size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"{bundle.name}: {name} does not match its manifest sha256/size")
            if _is_sidecar(name, kind):
                if not isinstance(json.loads(target.read_text(encoding="utf-8")), dict):
                    raise ValueError(f"{bundle.name}: {name} is not a JSON object")
            elif kind == "pdf" and not is_valid_pdf_artifact(target):
                raise ValueError(f"{bundle.name}: {name} is not a readable PDF")
            if kind == "md" and not target.read_bytes().decode("utf-8", errors="strict").strip():
                raise ValueError(f"{bundle.name}: {name} is empty")
    missing = sorted(set(listed) - seen)
    if missing:
        raise ValueError(f"{bundle.name}: manifest lists {len(missing)} file(s) the tar lacks, e.g. {missing[0]}")
    return manifest


# ----------------------------------------------------------------------- import
def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def import_slice_bundle(bundle: Path, dest_dir: Path, *, venue: str, year: int, kind: str) -> dict[str, int]:
    """Verify ``bundle`` and land its files in ``dest_dir``; returns {"files", "sidecars", "bytes"}.

    Staged beside ``dest_dir`` (same filesystem), so each rename is atomic; a
    verification failure leaves ``dest_dir`` untouched. Re-importing the same
    bundle replaces the files with identical bytes."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stage = dest_dir.parent / f".incoming-{dest_dir.name}-{uuid.uuid4().hex[:8]}"
    try:
        manifest = verify_slice_bundle(bundle, venue=venue, year=year, kind=kind, stage_dir=stage)
        receipt_path = dest_dir / RECEIPT_FILE
        try:
            receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipts = {}
        bundle_sha = sha256_file(bundle)
        received = datetime.now(timezone.utc).isoformat()
        total = sidecars = 0
        for entry in manifest["files"]:
            os.replace(stage / entry["name"], dest_dir / entry["name"])
            total += entry["size"]
            sidecars += _is_sidecar(entry["name"], kind)
            receipts[entry["name"]] = {
                "sha256": entry["sha256"], "size": entry["size"], "bundle": bundle.name,
                "bundle_sha256": bundle_sha, "source": manifest.get("source"),
                "created_at": manifest.get("created_at"), "received_at": received,
            }
        _write_json_atomic(receipt_path, receipts)
        return {"files": len(manifest["files"]) - sidecars, "sidecars": sidecars, "bytes": total}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def receive_slice_bundle(
    bundle: Path, dest_dir: Path, *, venue: str, year: int, kind: str,
    abort: threading.Event, log: Callable[[str], None],
    poll_s: float = 15.0, sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Wait for the producer to push ``bundle``, then verify and land it.

    Waiting holds nothing and ends only when the bundle appears or ``abort`` is
    set. A bundle that does not verify is renamed ``.rejected`` and the error is
    raised, so the stage fails once and a retry waits for a new push."""
    waited = 0.0
    while not bundle.is_file():
        if abort.is_set():
            raise RuntimeError("cancelled while waiting for the bundle")
        if waited % 600 < poll_s:
            log(f"waiting for {bundle} (pushed by the producing host)")
        sleep_fn(poll_s)
        waited += poll_s
    try:
        counts = import_slice_bundle(bundle, dest_dir, venue=venue, year=year, kind=kind)
    except (ValueError, tarfile.TarError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        rejected = bundle.with_name(bundle.name + ".rejected")
        os.replace(bundle, rejected)
        raise RuntimeError(f"{bundle.name} rejected ({rejected.name}): {exc}") from exc
    log(f"received {bundle.name}: {counts['files']:,} {kind} files, {counts['bytes'] / 1e6:.1f} MB -> {dest_dir}")
    return counts


__all__ = [
    "Executable", "InboxTransfer", "RECEIPT_FILE", "bundle_file_name", "import_slice_bundle",
    "inbox_dir", "push_slice_files", "receive_slice_bundle", "sha256_file", "verify_slice_bundle",
    "write_slice_bundle",
]
