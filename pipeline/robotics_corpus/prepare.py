"""Orchestrate robotics corpus preparation (full-year catalog + optional download).

Default scope is **all of ICRA / CoRL / RSS for 2022–2026**. Collect builds a
complete JSONL manifest for every available year; missing editions (e.g. RSS
2026 not yet published) are recorded in the summary and do not abort the rest.
Download is dry-run unless ``execute_download=True`` — then it walks the full
manifest (download_limit=0 means all rows).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from pipeline.paper_crawler.runtime_files import write_json_atomic
from pipeline.robotics_corpus.models import RoboticsPaperRecord, RoboticsVenue
from pipeline.robotics_corpus.pdf_fetch import download_pdf, plan_pdf_fetch
from pipeline.robotics_corpus.proxy import RoboticsProxyConfig, load_robotics_proxy_from_settings
from pipeline.robotics_corpus.sources import (
    DEFAULT_TARGET_YEARS,
    DEFAULT_VENUES,
    RoboticsHttpClient,
    collect_venue_year,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("data/robotics")


@dataclass(slots=True)
class PrepareConfig:
    data_root: Path = field(default_factory=lambda: DEFAULT_DATA_ROOT)
    # Full bulk window — not a smoke subset.
    years: Sequence[int] = field(default_factory=lambda: tuple(DEFAULT_TARGET_YEARS))
    venues: Sequence[RoboticsVenue | str] = field(default_factory=lambda: tuple(DEFAULT_VENUES))
    execute_download: bool = False
    allow_cf_bypass: bool = True
    download_limit: int = 0  # 0 = entire manifest
    # Polite default; raise on 429 only (JSON ICRA is 1–2 req/year).
    min_interval_s: float = 0.25
    # Concurrent PDF workers for OA venues (RSS/CoRL). IEEE/CF stays careful.
    download_concurrency: int = 8
    # Optional per-(venue,year) sample cap for smoke only; 0 = full year.
    sample_limit_per_venue_year: int = 0
    proxy: Optional[RoboticsProxyConfig] = None

    def resolved_venues(self) -> list[RoboticsVenue]:
        out: list[RoboticsVenue] = []
        for v in self.venues:
            out.append(v if isinstance(v, RoboticsVenue) else RoboticsVenue(str(v)))
        return out

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifest" / "robotics_manifest.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.data_root / "manifest" / "robotics_prepare_summary.json"

    @property
    def pdf_root(self) -> Path:
        return self.data_root / "pdf"


def _write_jsonl(path: Path, records: list[RoboticsPaperRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_jsonl(path: Path) -> list[RoboticsPaperRecord]:
    if not path.exists():
        return []
    out: list[RoboticsPaperRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(RoboticsPaperRecord.from_dict(json.loads(line)))
    return out


def collect_manifest(cfg: PrepareConfig) -> dict[str, Any]:
    """Collect **all** configured venue×year cells into one JSONL manifest.

    Default = ICRA+CoRL+RSS × 2022..2026. Each cell is collected fully unless
    ``sample_limit_per_venue_year > 0`` (smoke only). Failures for a single cell
    are logged and listed under ``skipped_cells`` so the rest still complete.
    """
    proxy = cfg.proxy or load_robotics_proxy_from_settings()
    t0 = time.perf_counter()
    venues = cfg.resolved_venues()
    years = list(cfg.years)
    records: list[RoboticsPaperRecord] = []
    by_cell: dict[str, int] = {}
    skipped_cells: list[dict[str, Any]] = []

    with RoboticsHttpClient(proxy=proxy, min_interval_s=cfg.min_interval_s) as client:
        for venue in venues:
            for year in years:
                cell = f"{venue.value}-{year}"
                logger.info("collecting robotics cell %s", cell)
                try:
                    papers = collect_venue_year(
                        venue,
                        year,
                        client,
                        max_papers=cfg.sample_limit_per_venue_year,
                    )
                    papers.sort(key=lambda p: p.paper_id)
                    for rec in papers:
                        plan = plan_pdf_fetch(rec, pdf_root=cfg.pdf_root)
                        if plan.action == "would_download":
                            rec.download_status = "planned"
                        elif plan.action == "skip_exists":
                            rec.download_status = "skipped"
                            rec.pdf_path = str(plan.dest_path)
                        elif plan.action == "no_url":
                            rec.download_status = "no_pdf"
                            rec.download_error = plan.note
                        records.append(rec)
                    by_cell[cell] = len(papers)
                    logger.info("  %s -> %d papers", cell, len(papers))
                except Exception as exc:
                    logger.warning("skip cell %s: %s", cell, exc)
                    skipped_cells.append({
                        "venue": venue.value,
                        "year": year,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    by_cell[cell] = 0

    _write_jsonl(cfg.manifest_path, records)
    summary: dict[str, Any] = {
        "stage": "collect",
        "years": years,
        "venues": [v.value for v in venues],
        "scope_note": "full bulk window; sample_limit only for smoke",
        "sample_limit_per_venue_year": cfg.sample_limit_per_venue_year,
        "records_written": len(records),
        "by_cell": by_cell,
        "skipped_cells": skipped_cells,
        "manifest_path": str(cfg.manifest_path),
        "pdf_root": str(cfg.pdf_root),
        "proxy": proxy.http_proxy,
        "planned_pdf": sum(1 for r in records if r.download_status == "planned"),
        "skipped_existing": sum(1 for r in records if r.download_status == "skipped"),
        "no_pdf_url": sum(1 for r in records if r.download_status == "no_pdf" or not r.pdf_url),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    write_json_atomic(cfg.summary_path, summary)
    logger.info(
        "robotics collect done records=%s cells=%s skipped=%s",
        len(records), len(by_cell), len(skipped_cells),
    )
    return summary


def run_download_pass(cfg: PrepareConfig) -> dict[str, Any]:
    """Download planned rows. ``download_limit=0`` = entire manifest in one pass.

    Uses a thread pool for OA-friendly sources (RSS/CoRL/PMLR). ICRA/IEEE rows
    still go through the same worker but CF bypass is serialized inside the
    downloader (one headless session attempt per paper max).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    proxy = cfg.proxy or load_robotics_proxy_from_settings()
    records = _load_jsonl(cfg.manifest_path)
    if not records:
        raise FileNotFoundError(f"no manifest at {cfg.manifest_path}; run collect first")
    t0 = time.perf_counter()
    budget = cfg.download_limit if cfg.download_limit > 0 else len(records)

    # Preserve order: build work list of indices to process.
    work_indices: list[int] = []
    for i, rec in enumerate(records):
        if rec.download_status in {"downloaded", "skipped"} and rec.pdf_path:
            continue
        if not rec.pdf_url and not rec.alternate_pdf_urls:
            rec.download_status = "no_pdf"
            rec.download_error = rec.download_error or "no pdf_url"
            continue
        if len(work_indices) >= budget:
            break
        work_indices.append(i)

    def _one(idx: int) -> tuple[int, RoboticsPaperRecord]:
        out = download_pdf(
            records[idx],
            pdf_root=cfg.pdf_root,
            proxy=proxy,
            execute=cfg.execute_download,
            allow_cf_bypass=cfg.allow_cf_bypass,
        )
        return idx, out

    concurrency = max(1, int(cfg.download_concurrency or 1))
    if not cfg.execute_download:
        # Dry-run is CPU-cheap; keep serial to avoid pointless thread overhead.
        concurrency = 1

    done = 0
    if concurrency == 1:
        for idx in work_indices:
            _, out = _one(idx)
            records[idx] = out
            done += 1
            if done % 50 == 0:
                _write_jsonl(cfg.manifest_path, records)
                logger.info("download progress %s/%s", done, len(work_indices))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_one, idx): idx for idx in work_indices}
            for fut in as_completed(futures):
                idx, out = fut.result()
                records[idx] = out
                done += 1
                if done % 50 == 0:
                    _write_jsonl(cfg.manifest_path, records)
                    logger.info("download progress %s/%s", done, len(work_indices))

    downloaded = planned = skipped = failed = blocked = no_pdf = 0
    for rec in records:
        st = rec.download_status
        if st == "downloaded":
            downloaded += 1
        elif st == "planned":
            planned += 1
        elif st == "skipped":
            skipped += 1
        elif st == "blocked":
            blocked += 1
        elif st == "no_pdf":
            no_pdf += 1
        elif st == "failed":
            failed += 1
        # pending left as-is when outside budget

    _write_jsonl(cfg.manifest_path, records)
    summary = {
        "stage": "download",
        "execute_download": cfg.execute_download,
        "records": len(records),
        "attempted": len(work_indices),
        "budget": budget,
        "downloaded": downloaded,
        "planned": planned,
        "skipped": skipped,
        "failed": failed,
        "blocked": blocked,
        "no_pdf": no_pdf,
        "download_concurrency": concurrency,
        "cf_bypass_enabled": cfg.allow_cf_bypass,
        "proxy": proxy.http_proxy,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "manifest_path": str(cfg.manifest_path),
        "years": list(cfg.years),
        "venues": [v.value for v in cfg.resolved_venues()],
    }
    write_json_atomic(cfg.summary_path, summary)
    return summary


def prepare(cfg: PrepareConfig) -> dict[str, Any]:
    """Collect full catalog then download pass (dry-run unless execute_download)."""
    collect_summary = collect_manifest(cfg)
    download_summary = run_download_pass(cfg)
    return {"collect": collect_summary, "download": download_summary}
