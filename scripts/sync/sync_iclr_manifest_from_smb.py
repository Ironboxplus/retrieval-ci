"""Scan SMB marker output and write markdown_path back to manifest.

After ``run_marker_kss.sh`` finishes, the ICLR MDs sit on the SMB share
at ``//10.100.100.9/lsb/retrival/papers/markdown/iclr/{year}/...md``.
The manifest still has ``markdown_path: None`` for those records, so
``preindex_md_to_sql_only.py`` would skip them. This script walks the
SMB MD tree, matches each MD to its manifest entry by ``paper_id``
(filename stem), and writes ``markdown_path`` back atomically.

Usage:
    python scripts/sync_iclr_manifest_from_smb.py            # use SMB UNC
    python scripts/sync_iclr_manifest_from_smb.py --md-root data/papers/...md

The SMB path is preferred for default (avoids a 500K-file rsync); the
preindex script reads via Windows SMB redirector with no perf hit since
each file is read once and is small.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CORPUS_ROOT = Path("data/papers/ccf_a_main_conference_corpus")
MANIFEST_PATH = CORPUS_ROOT / "manifests" / "paper_manifest.json"
DEFAULT_MD_ROOT_SMB = Path(r"\\10.100.100.9\lsb\retrival\papers\markdown\iclr")
DEFAULT_MD_ROOT_LOCAL_FALLBACK = (
    CORPUS_ROOT / "markdown" / "iclr"
)

log = logging.getLogger("sync_iclr_manifest")


def _scan_md_files(md_root: Path) -> dict[str, Path]:
    """Return {paper_id: md_path}. paper_id == md_filename_stem."""
    found: dict[str, Path] = {}
    if not md_root.exists():
        log.warning("md_root not accessible: %s", md_root)
        return found
    for md in md_root.rglob("*.md"):
        if md.stem.startswith("done"):
            continue  # done.rankN.txt accidentally suffixed
        found[md.stem] = md
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--md-root", type=Path, default=DEFAULT_MD_ROOT_SMB,
        help=f"Root dir of ICLR MDs. Default: {DEFAULT_MD_ROOT_SMB} "
             f"(falls back to {DEFAULT_MD_ROOT_LOCAL_FALLBACK} if SMB inaccessible)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    md_root = args.md_root
    if not md_root.exists():
        log.warning("primary --md-root not accessible, trying local fallback")
        md_root = DEFAULT_MD_ROOT_LOCAL_FALLBACK
        if not md_root.exists():
            log.error("neither SMB nor local md_root exists; nothing to sync")
            return 2

    log.info("scanning %s for *.md ...", md_root)
    md_index = _scan_md_files(md_root)
    log.info("found %d MD files under %s", len(md_index), md_root)

    log.info("loading manifest: %s", MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    log.info("manifest total entries: %d", len(manifest))

    updated = 0
    unchanged = 0
    missing = 0
    for paper in manifest:
        if (paper.get("conference") or "").upper() != "ICLR":
            continue
        pid = paper.get("paper_id")
        if not pid:
            continue
        md_path = md_index.get(pid)
        if md_path is None:
            missing += 1
            continue
        new_path = str(md_path)
        if paper.get("markdown_path") == new_path:
            unchanged += 1
            continue
        paper["markdown_path"] = new_path
        paper["extract_error"] = None
        updated += 1

    log.info("ICLR: updated=%d unchanged=%d missing_md=%d",
             updated, unchanged, missing)

    if args.dry_run:
        log.info("(dry-run, not writing manifest)")
        return 0

    if updated == 0:
        log.info("nothing to write")
        return 0

    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)
    log.info("manifest written: %s", MANIFEST_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
