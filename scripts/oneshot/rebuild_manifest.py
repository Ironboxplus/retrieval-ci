"""Rebuild manifest from iclr-r7 backup + re-collect new conferences + scan disk for PDFs.

Runs sequentially (no concurrency), no risk of manifest corruption.

Usage:
    python scripts/rebuild_manifest.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.paper_crawler.ccf_a_main_corpus import (
    CorpusPaper,
    DblpAcmProceedingsSource,
    EmnlpMainConferenceSource,
    OfficialProceedingsClient,
    build_default_corpus_root,
    configure_logging,
    sanitize_file_component,
    is_valid_pdf_artifact,
)
from pipeline.paper_crawler.runtime_files import write_json_atomic

LOGGER = logging.getLogger(__name__)
PROXY = "http://127.0.0.1:10808"
YEARS = (2022, 2023, 2024, 2025)


def main():
    configure_logging(verbose=False)
    corpus_root = build_default_corpus_root()
    manifest_path = corpus_root / "manifests" / "paper_manifest.json"
    pdf_root = corpus_root / "pdf"

    # Step 1: Load iclr-r7 backup (64,421 papers, the last known-good full baseline)
    backup = manifest_path.with_suffix(".json.bak.iclr-r7.1779184644")
    assert backup.exists(), f"Backup not found: {backup}"
    with backup.open(encoding="utf-8") as f:
        records = json.load(f)
    LOGGER.info("Loaded baseline: %d papers from %s", len(records), backup.name)
    record_map = {r["paper_id"]: r for r in records}

    # Step 2: Collect EMNLP (4 years)
    LOGGER.info("Collecting EMNLP...")
    source = EmnlpMainConferenceSource()
    client = OfficialProceedingsClient(proxy_url=PROXY)
    try:
        for year in YEARS:
            papers = source.collect_papers(year, client)
            added = 0
            for p in papers:
                if p.paper_id not in record_map:
                    record_map[p.paper_id] = p.to_manifest_record()
                    added += 1
            LOGGER.info("  EMNLP %d: %d papers (%d new)", year, len(papers), added)
            time.sleep(2)
    finally:
        client.close()

    # Step 3: Collect ACM (KDD, SIGMOD, SIGIR)
    acm_configs = [
        ("KDD", "kdd", {}),
        ("SIGMOD", "sigmod", {
            2023: "journals/pacmmod/pacmmod1",
            2024: "journals/pacmmod/pacmmod2",
            2025: "journals/pacmmod/pacmmod3",
        }),
        ("SIGIR", "sigir", {}),
    ]
    client = OfficialProceedingsClient(proxy_url=PROXY)
    try:
        for conf_name, venue_key, overrides in acm_configs:
            src = DblpAcmProceedingsSource(
                conference_name=conf_name,
                dblp_venue_key=venue_key,
                dblp_toc_overrides=overrides or None,
            )
            for year in YEARS:
                LOGGER.info("Collecting %s %d...", conf_name, year)
                try:
                    papers = src.collect_papers(year, client)
                    added = 0
                    for p in papers:
                        if p.paper_id not in record_map:
                            record_map[p.paper_id] = p.to_manifest_record()
                            added += 1
                    LOGGER.info("  %s %d: %d papers (%d new)", conf_name, year, len(papers), added)
                except Exception as e:
                    LOGGER.warning("  %s %d: FAILED (%s)", conf_name, year, e)
                time.sleep(5)
    finally:
        client.close()

    # Step 4: Scan disk for existing PDFs via fast directory listing (not per-file stat over SMB)
    LOGGER.info("Scanning disk for PDFs (fast dir listing)...")
    existing_pdfs: set[str] = set()
    for conf_dir in pdf_root.iterdir():
        if not conf_dir.is_dir():
            continue
        for year_dir in conf_dir.iterdir():
            if not year_dir.is_dir():
                continue
            try:
                for pdf_file in year_dir.glob("*.pdf"):
                    existing_pdfs.add(str(pdf_file))
            except OSError as e:
                LOGGER.warning("  Skipping %s: %s", year_dir, e)
    LOGGER.info("  Found %d PDF files on disk", len(existing_pdfs))

    pdf_found = 0
    for rec in record_map.values():
        conf = rec.get("conference", "").lower()
        year = rec.get("year", 0)
        pid = rec.get("paper_id", "")
        pdf_path = pdf_root / conf / str(year) / f"{sanitize_file_component(pid)}.pdf"
        if str(pdf_path) in existing_pdfs:
            rec["pdf_path"] = str(pdf_path)
            rec["pdf_resolution_status"] = rec.get("pdf_resolution_status") or "downloaded"
            pdf_found += 1
    LOGGER.info("  Matched %d papers to PDFs", pdf_found)

    # Step 5: Backup current manifest + write new one
    if manifest_path.exists():
        bak = manifest_path.with_suffix(f".json.bak.rebuild-{int(time.time())}")
        import shutil
        shutil.copy2(manifest_path, bak)
        LOGGER.info("Backed up current manifest to %s", bak.name)

    write_json_atomic(manifest_path, list(record_map.values()))

    # Summary
    confs = {}
    pdfs = {}
    for r in record_map.values():
        c = r.get("conference", "?")
        confs[c] = confs.get(c, 0) + 1
        if r.get("pdf_path"):
            pdfs[c] = pdfs.get(c, 0) + 1

    LOGGER.info("=== Manifest rebuilt: %d total ===", len(record_map))
    for c in sorted(confs.keys()):
        LOGGER.info("  %s: %d papers, %d PDFs", c, confs[c], pdfs.get(c, 0))


if __name__ == "__main__":
    main()
