"""Collect ACM conference papers (KDD/SIGMOD/SIGIR/WWW) via DBLP + download PDFs via nodriver.

Cloudflare on dl.acm.org is passed ONCE in a real (off-screen) browser, then each
PDF is fetched from inside that cleared page with fetch(credentials:'include') —
the browser re-attaches the hidden cf_clearance cookie with the real Chrome JA3,
so no per-paper navigation and no cookie extraction are needed.

Usage:
    python scripts/collect/collect_and_download_acm.py --stage collect --conferences KDD SIGMOD SIGIR
    python scripts/collect/collect_and_download_acm.py --stage download --conferences WWW
    python scripts/collect/collect_and_download_acm.py --stage download --conferences WWW --limit 4  # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import core.wmi_fix  # noqa: F401 — Python 3.13 WMI hang bypass

from pipeline.download_control import ChainedProxyDownloadError, ChainedProxyPool, interleave_by_key
from pipeline.medical_corpus.cf_bypass import (
    CfBypassResult,
    solve_cloudflare_for_url,
)
from pipeline.paper_crawler.ccf_a_main_corpus import (
    CorpusPaper,
    DblpAcmProceedingsSource,
    KddOfficialProceedingsSource,
    MainConferenceCorpusBuilder,
    OfficialProceedingsClient,
    SigirOfficialProgramSource,
    build_default_corpus_root,
    configure_logging,
    filter_main_track_papers,
    sanitize_file_component,
)
from pipeline.paper_crawler.runtime_files import (
    build_write_in_progress_path,
    is_valid_pdf_artifact,
    write_json_atomic,
)


def _build_existing_pdf_set(pdf_root: Path, conferences: list[str], years: tuple) -> set[str]:
    """Batch scan: one listdir per directory instead of N stat() calls.
    Returns set of full lowercase paths for O(1) membership test."""
    existing = set()
    for conf in conferences:
        for year in years:
            d = pdf_root / conf.lower() / str(year)
            if not d.is_dir():
                continue
            for name in os.listdir(d):
                if name.endswith(".pdf"):
                    existing.add(str(d / name))
    return existing

LOGGER = logging.getLogger(__name__)

DEFAULT_PROXY = "http://127.0.0.1:10808"
DEFAULT_FIRST_HOP_PROXY = "socks5://127.0.0.1:10808"
CHAIN_PROBE_PDF_URL = "https://aclanthology.org/2025.acl-long.1.pdf"
YEARS = (2022, 2023, 2024, 2025, 2026)

CONFERENCE_CONFIG = {
    "KDD": {
        "source_class": KddOfficialProceedingsSource,
        "dblp_venue_key": "kdd",
        "dblp_toc_overrides": {
            2025: ["conf/kdd/kdd2025-1", "conf/kdd/kdd2025-2"],
        },
    },
    "SIGMOD": {
        "dblp_venue_key": "sigmod",
        "dblp_toc_overrides": {
            2023: "journals/pacmmod/pacmmod1",
            2024: "journals/pacmmod/pacmmod2",
            2025: "journals/pacmmod/pacmmod3",
        },
    },
    "SIGIR": {
        "source_class": SigirOfficialProgramSource,
        "dblp_venue_key": "sigir",
    },
    # WWW (The Web Conference). Default key conf/www/www{year} is the main
    # research track only — the companion volume (posters/demos/workshops) is
    # a separate key conf/www/www{year}c, so no override/denylist is needed.
    "WWW": {"dblp_venue_key": "www"},
    # ACM Computing Surveys (CSUR) — CCF-A journal on ACM DL. As a journal, DBLP
    # organizes it by VOLUME not conf-year: volume N = year 1968+N (vol54=2022 …
    # vol58=2026). Override every year to its volume TOC key; PDFs come from the
    # same dl.acm.org/doi/pdf/{doi} path (probed open-access, 9/9 sample PDFs).
    "CSUR": {
        "dblp_venue_key": "csur",
        "dblp_toc_overrides": {
            2022: "journals/csur/csur54",
            2023: "journals/csur/csur55",
            2024: "journals/csur/csur56",
            2025: "journals/csur/csur57",
            2026: "journals/csur/csur58",
        },
    },
    # ── DS computer-architecture / parallel-computing venues ──
    # These DBLP TOCs mix the main research track with workshop/demo/poster/
    # tutorial entries under the SAME 10.1145 prefix (section-blind, W90). So
    # main_only=True runs the STRICT HTML-TOC filter (filter_main_track_papers):
    # keep ONLY research/short/industry, drop everything non-main AND anything
    # we cannot classify — the user gate "必须是主会… 如果无法分清这个，也不要下".
    # `years` overrides the global YEARS where only some years are ACM-hosted.
    "PPoPP": {
        "dblp_venue_key": "ppopp",
        "main_only": True,
        "years": (2022, 2023, 2024, 2025),
    },
    # ISCA 2022-2024 is on IEEE Xplore (10.1109, paywalled) → only 2025 moved to
    # ACM (10.1145) and is fetchable (conf/conferences.yaml ISCA years:[2025]).
    "ISCA": {
        "dblp_venue_key": "isca",
        "main_only": True,
        "years": (2025,),
    },
}
# Collected through the registry (scripts/collect/collect_2026_to_manifest.py,
# Crossref listing), downloaded here because ACM DL needs the browser session.
DOWNLOAD_ONLY_CONFERENCES = ("CCS", "SOSP")


def load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def select_acm_manifest_records(
    records: list[dict],
    conferences: set[str],
    *,
    year: int | None,
) -> list[dict]:
    return [
        record
        for record in records
        if record.get("conference") in conferences
        and (year is None or record.get("year") == year)
    ]


def backup_manifest(manifest_path: Path) -> Path:
    bak = manifest_path.with_suffix(f".json.bak.acm-{int(time.time())}")
    import shutil
    shutil.copy2(manifest_path, bak)
    LOGGER.info("Backed up manifest to %s (%d bytes)", bak.name, bak.stat().st_size)
    return bak


def collect_acm(
    conferences: list[str],
    proxy_url: str,
    limit: int | None = None,
    *,
    resolve_open_access: bool = False,
) -> list[CorpusPaper]:
    client = OfficialProceedingsClient(proxy_url=proxy_url)
    all_papers: list[CorpusPaper] = []
    try:
        for conf_name in conferences:
            cfg = CONFERENCE_CONFIG[conf_name]
            source_class = cfg.get("source_class", DblpAcmProceedingsSource)
            source = source_class(
                conference_name=conf_name,
                dblp_venue_key=cfg["dblp_venue_key"],
                dblp_toc_overrides=cfg.get("dblp_toc_overrides"),
            )
            main_only = cfg.get("main_only", False)
            for year in cfg.get("years", YEARS):
                LOGGER.info("Collecting %s %d from DBLP...", conf_name, year)
                try:
                    papers = source.collect_papers(year, client)
                    papers.sort(key=lambda p: p.paper_id)
                    resolve_oa = (
                        getattr(source, "resolve_open_access_pdf_urls", None)
                        if resolve_open_access else None
                    )
                    if resolve_oa is not None:
                        resolution = resolve_oa(papers, client)
                        LOGGER.info("  %s %d OA resolution: %s", conf_name, year, resolution)
                    if main_only:
                        # Section-blind JSON pull → strict main-only via HTML-TOC
                        # sections. Drop non-main + unclassifiable BEFORE download.
                        doi_to_track = source.collect_track_map(year, client)
                        papers, dropped = filter_main_track_papers(papers, doi_to_track)
                        LOGGER.info(
                            "  %s %d main-only filter: kept %d main, dropped %s",
                            conf_name, year, len(papers),
                            dict(sorted(dropped.items(), key=lambda kv: -kv[1])) or {},
                        )
                    if limit:
                        papers = papers[:limit]
                    all_papers.extend(papers)
                    LOGGER.info("  %d papers", len(papers))
                except Exception as e:
                    LOGGER.warning("Failed to collect %s %d: %s (skipping)", conf_name, year, e)
                time.sleep(5)  # DBLP rate limit: be polite
    finally:
        client.close()
    LOGGER.info("Total ACM collected: %d", len(all_papers))
    return all_papers


def merge_into_manifest(manifest_path: Path, new_papers: list[CorpusPaper]) -> int:
    existing = load_manifest(manifest_path)
    existing_ids = {r["paper_id"] for r in existing}
    added = 0
    for paper in new_papers:
        if paper.paper_id not in existing_ids:
            existing.append(paper.to_manifest_record())
            existing_ids.add(paper.paper_id)
            added += 1
    write_json_atomic(manifest_path, existing)
    LOGGER.info("Merged %d new ACM papers (total manifest: %d)", added, len(existing))
    return added


def select_pending_downloads(papers: list[CorpusPaper], pdf_root: Path) -> tuple[list[tuple], int]:
    """(paper, pdf_path) still to download, venues alternating, and the front-matter count.

    Papers whose PDF is already on disk get their pdf_path set and are left out.
    """
    conferences_in_batch = list({p.conference for p in papers})
    years_in_batch = tuple(sorted({p.year for p in papers}))
    existing_pdfs = _build_existing_pdf_set(
        pdf_root,
        conferences_in_batch,
        years_in_batch,
    )
    LOGGER.info("Batch scan found %d existing PDFs across %s", len(existing_pdfs), conferences_in_batch)

    # Identify proceedings front-matter: a DOI that is the PARENT of other article
    # DOIs in this set (e.g. 10.1145/3485447 is parent of 10.1145/3485447.3511297).
    # Journals like CSUR have unique single-number DOIs with no children, so none
    # are skipped — only conference volume entries are. Venue-agnostic, not a
    # naive "no dot" test (which wrongly skipped every journal article).
    def _doi_of(p) -> str:
        return (p.pdf_url or "").replace("https://dl.acm.org/doi/pdf/", "")

    parent_dois: set[str] = set()
    for paper in papers:
        base, _, seg = _doi_of(paper).rpartition("/")
        if base and "." in seg:
            parent_dois.add(f"{base}/{seg.split('.', 1)[0]}")

    needs_download: list[tuple] = []
    skipped_frontmatter = 0
    for paper in papers:
        doi = _doi_of(paper)
        if not doi or doi in parent_dois:
            skipped_frontmatter += 1
            continue
        pdf_path = pdf_root / paper.conference.lower() / str(paper.year) / f"{sanitize_file_component(paper.paper_id)}.pdf"
        if str(pdf_path) in existing_pdfs:
            paper.pdf_path = str(pdf_path)
            paper.pdf_resolution_status = paper.pdf_resolution_status or "downloaded"
            continue
        needs_download.append((paper, pdf_path))
    # Two consecutive PDFs come from different venues while more than one has work.
    needs_download = interleave_by_key(needs_download, key=lambda item: item[0].conference)
    return needs_download, skipped_frontmatter


async def download_acm_pdfs(
    papers: list[CorpusPaper],
    pdf_root: Path,
    proxy_url: str,
    batch_size: int = 3,
    delay_between: float = 2.0,
    limit: int | None = None,
    challenge_retries: int = 12,
    require_solver: bool = True,
    chained_proxy_pool: ChainedProxyPool | None = None,
    xray_executable: Path | None = None,
    max_pdfs_per_session: int = 30,
) -> list[CorpusPaper]:
    """Download PDFs from ACM DL inside a solver-cleared browser session.

    Why this design (and what it replaces):
      The proxied exit IP draws Cloudflare's non-interactive managed challenge
      ("verifying you are human", served in Chinese, with a Ray ID) — there is no
      clickable Turnstile and it never clears by waiting; only a FRESH browser
      gets a new CF evaluation (~1 in 6 launches pass). The previous approach
      opened a NEW TAB + CDP Fetch-intercept + page.navigate PER PAPER: the
      new_tab CDP call has no timeout and deadlocks once CF re-challenges, and
      cf_clearance is httpOnly/partitioned so it can't be lifted out to a plain
      httpx/curl_cffi client (both 403).

      Instead: use the vendored Ironboxplus TurnstileBypass solver to mint and
      verify a headed Chrome session, inject only its ACM-domain cookies into a
      fresh nodriver Chrome with the same UA, then keep that dl.acm.org page and
      fetch() PDFs inside it with credentials:'include'.  No per-paper navigation
      and no plain-client replay are involved.  When clearance expires, the same
      solver-assisted launch is repeated for the blocked batch.
    """
    needs_download, skipped_frontmatter = select_pending_downloads(papers, pdf_root)

    LOGGER.info(
        "Need to download: %d / %d (skipped %d existing, %d front-matter)",
        len(needs_download), len(papers),
        len(papers) - len(needs_download) - skipped_frontmatter, skipped_frontmatter,
    )

    # Smoke cap: applied AFTER front-matter/existing skips so it selects N real
    # to-download papers (not the front-matter that sorts first in the manifest).
    if limit:
        needs_download = needs_download[:limit]
        LOGGER.info("Smoke limit: capped to %d papers", len(needs_download))

    if not needs_download:
        return papers
    if max_pdfs_per_session <= 0:
        raise ValueError("max_pdfs_per_session must be positive")

    if (chained_proxy_pool is None) != (xray_executable is None):
        raise ValueError(
            "chained_proxy_pool and xray_executable must be provided together"
        )
    if chained_proxy_pool is not None:
        required_healthy = min(
            chained_proxy_pool.configured_count,
            max(1, min(6, int(challenge_retries))),
        )
        probe = await chained_proxy_pool.probe_pdf(
            CHAIN_PROBE_PDF_URL,
            required_healthy=required_healthy,
            max_candidates=min(96, chained_proxy_pool.configured_count),
            concurrency=min(6, chained_proxy_pool.configured_count),
            timeout_seconds=8.0,
        )
        if probe.healthy_count < required_healthy:
            raise RuntimeError(
                f"only {probe.healthy_count}/{required_healthy} landing exits passed "
                "the real two-hop PDF probe; "
                "refusing to fall back to the first hop"
            )
        LOGGER.info(
            "Two-hop browser pool ready: healthy=%d attempted=%d configured=%d",
            probe.healthy_count,
            probe.attempted_count,
            probe.configured_count,
        )

    WARM_MAX_ATTEMPTS = max(1, int(challenge_retries))
    WARM_POLL_ROUNDS = 4     # ~12s budget per browser instance before relaunching
    FETCH_CONCURRENCY = max(1, batch_size)   # parallel in-page fetches per batch (keep low: 429)
    FETCH_TIMEOUT = 120.0    # seconds per batch evaluate (covers a 4-5 MB PDF over CDP)
    SOLVER_TIMEOUT = 45.0    # dead/blocked exits should rotate quickly
    SOLVER_SESSION_TIMEOUT = 600.0
    CHUNK_MAX_RETRIES = max(1, int(challenge_retries))
    REMINT_ABORT = max(CHUNK_MAX_RETRIES + 1, 16)
    browser_exit_exclusions: set[int] = set()

    def _landing_url(pdf_url: str) -> str:
        return pdf_url.replace("https://dl.acm.org/doi/pdf/", "https://dl.acm.org/doi/")

    def _kill_chrome_tree(br):
        """Kill browser process tree (Windows: taskkill /T kills children)."""
        pid = getattr(br, '_process_pid', None) or (br._process.pid if br and br._process else None)
        if pid:
            try:
                import subprocess
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=10,
                )
                LOGGER.info("Killed Chrome tree PID %d", pid)
            except Exception:
                pass
        try:
            if br:
                br.stop()
        except Exception:
            pass

    # Runs inside the exact driverless Chrome session that solved Cloudflare.
    # Moving only cookies/UA to a second process is not sufficient for ACM's
    # PDF edge, so TurnstileBypass invokes this through its verified callback.
    FETCH_JS = r"""mw:async (urlsJson) => {
  const urls = JSON.parse(urlsJson);
  async function one(u) {
    try {
      const r = await fetch(u, {credentials: 'include'});
      const ct = (r.headers.get('content-type') || '').toLowerCase();
      if (r.status === 200 && ct.indexOf('pdf') !== -1) {
        const bytes = new Uint8Array(await r.arrayBuffer());
        let bin = ''; const CH = 0x8000;
        for (let i = 0; i < bytes.length; i += CH)
          bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
        return {ok: true, b64: btoa(bin), len: bytes.length};
      }
      let snip = '';
      try { snip = (await r.text()).slice(0, 120); } catch (e) {}
      return {ok: false, status: r.status, snip: snip};
    } catch (e) { return {ok: false, error: String(e)}; }
  }
  return JSON.stringify(await Promise.all(urls.map(one)));
}
"""

    class _SolvedPage:
        def __init__(self, results_by_url: dict[str, dict]):
            self._results_by_url = results_by_url

        def results_for(self, urls: list[str]) -> list[dict]:
            if any(url not in self._results_by_url for url in urls):
                return []
            return [
                self._results_by_url[url]
                for url in urls
            ]

    async def _launch_browser(challenge_url: str):
        """Solve one PDF edge and consume pending PDFs before that browser exits."""
        if not require_solver:
            raise RuntimeError("ACM DL downloads require TurnstileBypass")

        for attempt in range(1, WARM_MAX_ATTEMPTS + 1):
            session_results: dict[str, dict] = {}

            async def _download_in_verified_session(page, initial_download=None):
                if initial_download is not None:
                    initial = next(
                        (
                            item
                            for item in needs_download
                            if item[0].pdf_url == challenge_url
                        ),
                        None,
                    )
                    if initial is not None:
                        paper, pdf_path = initial
                        pdf_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary_path = build_write_in_progress_path(pdf_path)
                        try:
                            await asyncio.wait_for(
                                initial_download.save_as(str(temporary_path)),
                                timeout=FETCH_TIMEOUT,
                            )
                            if not is_valid_pdf_artifact(temporary_path):
                                raise RuntimeError("browser download failed PDF parser validation")
                            temporary_path.replace(pdf_path)
                            paper.pdf_path = str(pdf_path)
                            paper.downloaded_from_url = paper.pdf_url
                            paper.pdf_resolution_status = "downloaded"
                            paper.download_error = None
                            session_results[paper.pdf_url] = {
                                "ok": True,
                                "saved": True,
                                "len": pdf_path.stat().st_size,
                            }
                        except Exception as exc:
                            temporary_path.unlink(missing_ok=True)
                            session_results[paper.pdf_url] = {
                                "ok": False,
                                "error": f"download event save failed: {type(exc).__name__}: {exc}",
                            }

                pending = [
                    item
                    for item in needs_download
                    if not item[0].pdf_path and not item[0].download_error
                ][:max_pdfs_per_session]
                downloaded_here = sum(
                    1 for result in session_results.values() if result.get("saved")
                )
                challenged = False
                active_page = page
                if pending and initial_download is not None:
                    active_page = await page.context.new_page()
                    await active_page.goto(
                        _landing_url(pending[0][0].pdf_url),
                        wait_until="domcontentloaded",
                        timeout=30_000,
                    )
                for offset in range(0, len(pending), FETCH_CONCURRENCY):
                    chunk = pending[offset:offset + FETCH_CONCURRENCY]
                    urls = [paper.pdf_url for paper, _ in chunk]
                    try:
                        evaluate = getattr(active_page, "evaluate", None)
                        if not callable(evaluate):
                            raise RuntimeError("Camoufox page does not expose evaluate()")
                        raw = await asyncio.wait_for(
                            evaluate(FETCH_JS, json.dumps(urls)),
                            timeout=FETCH_TIMEOUT,
                        )
                        results = json.loads(raw) if raw else []
                    except Exception as exc:
                        results = [
                            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                            for _ in urls
                        ]

                    for (paper, pdf_path), result in zip(chunk, results):
                        if result.get("ok") and result.get("b64"):
                            try:
                                pdf_bytes = base64.b64decode(result["b64"])
                            except Exception:
                                pdf_bytes = b""
                            pdf_path.parent.mkdir(parents=True, exist_ok=True)
                            temporary_path = build_write_in_progress_path(pdf_path)
                            temporary_path.write_bytes(pdf_bytes)
                            if pdf_bytes[:5] != b"%PDF-" or not is_valid_pdf_artifact(temporary_path):
                                temporary_path.unlink(missing_ok=True)
                                session_results[paper.pdf_url] = {
                                    "ok": False,
                                    "error": f"PDF parser validation failed (len={len(pdf_bytes)})",
                                }
                                continue
                            temporary_path.replace(pdf_path)
                            paper.pdf_path = str(pdf_path)
                            paper.downloaded_from_url = paper.pdf_url
                            paper.pdf_resolution_status = "downloaded"
                            paper.download_error = None
                            downloaded_here += 1
                            session_results[paper.pdf_url] = {
                                "ok": True,
                                "saved": True,
                                "len": len(pdf_bytes),
                            }
                        else:
                            session_results[paper.pdf_url] = dict(result)
                            challenged = challenged or _looks_challenged(result)

                    LOGGER.info(
                        "Solver session progress: %d/%d pending PDFs saved",
                        downloaded_here,
                        len(pending),
                    )
                    if challenged:
                        break
                    if delay_between:
                        await asyncio.sleep(delay_between)
                return {
                    "downloaded": downloaded_here,
                    "attempted": len(session_results),
                    "challenged": challenged,
                }

            async def _solve_with_proxy(active_proxy: str):
                return await solve_cloudflare_for_url(
                    _landing_url(challenge_url),
                    proxy=active_proxy,
                    path="managed",
                    require_clearance=False,
                    headless=False,
                    verify=True,
                    timeout_s=SOLVER_TIMEOUT,
                    overall_timeout_s=SOLVER_SESSION_TIMEOUT,
                    backend="camoufox",
                    verified_session_callback=_download_in_verified_session,
                )

            if chained_proxy_pool is not None:
                lease = None
                try:
                    async with chained_proxy_pool.open_browser_proxy(
                        xray_executable=Path(xray_executable),
                        excluded_ordinals=browser_exit_exclusions,
                    ) as leased_proxy:
                        lease = leased_proxy
                        solver_result = await _solve_with_proxy(leased_proxy.proxy_url)
                        usable = (
                            solver_result.ok
                            and solver_result.verified
                            and any(result.get("saved") for result in session_results.values())
                        )
                        if not usable:
                            failure_summary = [
                                {
                                    "status": result.get("status"),
                                    "error": str(result.get("error") or "")[:160],
                                }
                                for result in session_results.values()
                            ]
                            raise RuntimeError(
                                solver_result.error
                                or f"solver produced no saved PDF: {failure_summary}"
                            )
                except Exception as exc:
                    if lease is not None:
                        browser_exit_exclusions.add(lease.exit_ordinal)
                    LOGGER.warning(
                        "ACM two-hop solver attempt %d failed on landing exit #%s: %s",
                        attempt,
                        lease.exit_ordinal if lease is not None else "unavailable",
                        f"{type(exc).__name__}: {exc}",
                    )
                    if isinstance(exc, ChainedProxyDownloadError) and lease is None:
                        # Every verified exit has been challenged once this run
                        # (2026-11-10: 6 verified exits ran out after 120 PDFs).
                        # Verify more of the pool; if none passes, the old
                        # exclusions are stale by now, so retry them after a pause.
                        verified_before = chained_proxy_pool.healthy_count
                        await chained_proxy_pool.probe_pdf(
                            CHAIN_PROBE_PDF_URL,
                            required_healthy=verified_before + 6,
                            max_candidates=chained_proxy_pool.configured_count,
                            concurrency=min(6, chained_proxy_pool.configured_count),
                            timeout_seconds=8.0,
                        )
                        LOGGER.info(
                            "Verified landing exits: %d -> %d",
                            verified_before,
                            chained_proxy_pool.healthy_count,
                        )
                        if chained_proxy_pool.healthy_count <= verified_before:
                            browser_exit_exclusions.clear()
                            await asyncio.sleep(60)
                    continue
            else:
                solver_result = await _solve_with_proxy(proxy_url)
                if not (
                    solver_result.ok
                    and solver_result.verified
                    and any(result.get("saved") for result in session_results.values())
                ):
                    LOGGER.warning(
                        "ACM solver attempt %d made no usable session: backend=%s verified=%s error=%s",
                        attempt,
                        solver_result.backend or "unknown",
                        solver_result.verified,
                        solver_result.error or "no saved PDF",
                    )
                    continue

            LOGGER.info(
                "ACM solver session completed (attempt %d, backend=%s, results=%d)",
                attempt,
                solver_result.backend or "unknown",
                len(session_results),
            )
            return None, _SolvedPage(session_results)

        raise RuntimeError(
            f"ACM DL solver produced no usable PDF session after "
            f"{WARM_MAX_ATTEMPTS} attempt(s)"
        )

    def _looks_challenged(res: dict) -> bool:
        if res.get("status") == 403:
            return True
        blob = ((res.get("snip") or "") + " " + (res.get("error") or "")).lower()
        return any(m in blob for m in (
            "just a moment", "verifying", "稍候", "安全验证", "attention required",
            "cloudflare", "ray id", "cf-chl", "challenge"))

    first_pdf_url = needs_download[0][0].pdf_url
    browser, page = await _launch_browser(first_pdf_url)
    ok_count = 0
    fail_count = 0
    remints_no_progress = 0
    chunk_retries = 0
    total = len(needs_download)

    try:
        i = 0
        while i < total:
            chunk = needs_download[i:i + FETCH_CONCURRENCY]
            urls = [p.pdf_url for p, _ in chunk]
            challenge_url = urls[0]

            if (i // FETCH_CONCURRENCY) % 10 == 0:
                LOGGER.info("Progress: %d/%d (ok=%d fail=%d)", i, total, ok_count, fail_count)

            results = page.results_for(urls)

            # session dead / hung → re-mint and retry the SAME chunk
            if not results:
                _kill_chrome_tree(browser)
                await asyncio.sleep(2)
                browser, page = await _launch_browser(challenge_url)
                remints_no_progress += 1
                chunk_retries += 1
                if remints_no_progress >= REMINT_ABORT:
                    LOGGER.error("%d consecutive re-mints with no progress, aborting", remints_no_progress)
                    break
                if chunk_retries >= CHUNK_MAX_RETRIES:
                    LOGGER.error("Batch at [%d] stuck after %d re-mints, skipping it", i, chunk_retries)
                    for paper, _ in chunk:
                        paper.download_error = "stuck batch (eval kept failing)"
                        fail_count += 1
                    i += FETCH_CONCURRENCY
                    chunk_retries = 0
                continue

            chunk_ok = 0
            chunk_challenged = 0
            for (paper, pdf_path), res in zip(chunk, results):
                if res.get("ok") and res.get("saved") and is_valid_pdf_artifact(pdf_path):
                    paper.pdf_path = str(pdf_path)
                    paper.downloaded_from_url = paper.pdf_url
                    paper.pdf_resolution_status = "downloaded"
                    paper.download_error = None
                    ok_count += 1
                    chunk_ok += 1
                    LOGGER.info(
                        "ok [%d/%d] %s (%d KB)",
                        ok_count,
                        total,
                        paper.paper_id,
                        int(res.get("len") or pdf_path.stat().st_size) // 1024,
                    )
                elif res.get("ok") and res.get("b64"):
                    try:
                        pdf_bytes = base64.b64decode(res["b64"])
                    except Exception:
                        pdf_bytes = b""
                    if pdf_bytes[:5] == b"%PDF-":
                        pdf_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary_path = build_write_in_progress_path(pdf_path)
                        temporary_path.write_bytes(pdf_bytes)
                        if not is_valid_pdf_artifact(temporary_path):
                            temporary_path.unlink(missing_ok=True)
                            paper.download_error = (
                                f"downloaded bytes failed PDF parser validation "
                                f"(len={len(pdf_bytes)})"
                            )
                            fail_count += 1
                            continue
                        temporary_path.replace(pdf_path)
                        paper.pdf_path = str(pdf_path)
                        paper.downloaded_from_url = paper.pdf_url
                        paper.pdf_resolution_status = "downloaded"
                        paper.download_error = None
                        ok_count += 1
                        chunk_ok += 1
                        LOGGER.info("ok [%d/%d] %s (%d KB)", ok_count, total, paper.paper_id, len(pdf_bytes) // 1024)
                    else:
                        paper.download_error = f"decoded not PDF (len={len(pdf_bytes)})"
                        fail_count += 1
                else:
                    if _looks_challenged(res):
                        chunk_challenged += 1
                    paper.download_error = f"fetch fail: {json.dumps(res, ensure_ascii=False)[:140]}"
                    fail_count += 1

            if chunk_ok > 0:
                # progress made — clear the stuck counters and advance
                remints_no_progress = 0
                chunk_retries = 0
                i += FETCH_CONCURRENCY
                await asyncio.sleep(delay_between)
            elif chunk_challenged > 0:
                # batch blocked by CF → cf_clearance expired. Roll back this
                # chunk's failures and re-mint, retrying the same chunk.
                fail_count -= len(chunk)
                for paper, _ in chunk:
                    paper.download_error = None
                LOGGER.warning("Batch at [%d] challenged (%d/%d) — cf_clearance expired, re-minting",
                               i, chunk_challenged, len(chunk))
                _kill_chrome_tree(browser)
                await asyncio.sleep(2)
                browser, page = await _launch_browser(challenge_url)
                remints_no_progress += 1
                chunk_retries += 1
                if remints_no_progress >= REMINT_ABORT:
                    LOGGER.error("%d consecutive re-mints with no progress, aborting", remints_no_progress)
                    break
                if chunk_retries >= CHUNK_MAX_RETRIES:
                    LOGGER.error("Batch at [%d] still challenged after %d re-mints, skipping it", i, chunk_retries)
                    for paper, _ in chunk:
                        paper.download_error = "stuck batch (challenged)"
                        fail_count += 1
                    i += FETCH_CONCURRENCY
                    chunk_retries = 0
            else:
                # all real failures (404/dead links), no CF challenge → record + advance
                chunk_retries = 0
                i += FETCH_CONCURRENCY
                await asyncio.sleep(delay_between)

        LOGGER.info("Download complete: %d OK, %d failed out of %d", ok_count, fail_count, total)
    finally:
        _kill_chrome_tree(browser)

    return papers


def main():
    parser = argparse.ArgumentParser(description="Collect + download ACM conference papers (DBLP + nodriver)")
    parser.add_argument("--stage", choices=("collect", "download", "all"), default="all")
    parser.add_argument("--conferences", nargs="+", default=list(CONFERENCE_CONFIG.keys()),
                        choices=[*CONFERENCE_CONFIG, *DOWNLOAD_ONLY_CONFERENCES],
                        help=f"{', '.join(DOWNLOAD_ONLY_CONFERENCES)} take --stage download only.")
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Restrict the download stage to one publication year.",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=None,
        help="Local PDF artifact root override; manifest remains canonical.",
    )
    parser.add_argument(
        "--proxy-pool-file",
        type=Path,
        default=None,
        help="Credentialed SOCKS landing pool used after --first-hop-proxy.",
    )
    parser.add_argument(
        "--first-hop-proxy",
        default=DEFAULT_FIRST_HOP_PROXY,
        help="Mandatory first SOCKS hop for landing-pool mode.",
    )
    parser.add_argument(
        "--xray-executable",
        type=Path,
        default=None,
        help="Xray binary used for the browser-facing two-hop SOCKS bridge.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit papers (collect: per conf/year; download: first N still-pending, for smoke)")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="In-page fetch concurrency per batch (keep low to avoid 429)")
    parser.add_argument(
        "--session-size",
        type=int,
        default=30,
        help="Maximum PDFs fetched under one browser clearance (default: 30).",
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between batches (seconds)")
    parser.add_argument(
        "--challenge-retries",
        type=int,
        default=12,
        help="Fresh-browser retries for one PDF batch after ACM challenges it (default: 12)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resolve-open-access",
        action="store_true",
        help="Explicitly replace ACM DL PDF URLs with resolved OA copies during collection.",
    )
    args = parser.parse_args()

    if (args.proxy_pool_file is None) != (args.xray_executable is None):
        parser.error("--proxy-pool-file and --xray-executable must be supplied together")
    download_only = sorted(set(args.conferences) & set(DOWNLOAD_ONLY_CONFERENCES))
    if download_only and args.stage != "download":
        parser.error(f"{', '.join(download_only)} are collected by collect_2026_to_manifest.py; use --stage download")

    configure_logging(verbose=args.verbose)
    # Quiet the per-launch browser/arg dumps during the long relaunch gauntlet so
    # the unattended log stays greppable (our own LOGGER stays at INFO).
    for _noisy in ("nodriver", "uc", "websockets", "httpx", "httpcore", "asyncio"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    corpus_root = build_default_corpus_root()
    manifest_path = corpus_root / "manifests" / "paper_manifest.json"
    pdf_root = args.pdf_root or (corpus_root / "pdf")

    if args.stage in ("collect", "all"):
        # Collect merges into the full corpus manifest; a short one means a broken copy.
        # Download only rewrites the records it loaded, so it runs on working slices too.
        existing = load_manifest(manifest_path)
        assert len(existing) >= 60000, f"ABORT: manifest too small ({len(existing)}), refusing to proceed"
        backup_manifest(manifest_path)
        acm_papers = collect_acm(
            args.conferences,
            args.proxy,
            limit=args.limit,
            resolve_open_access=args.resolve_open_access,
        )
        added = merge_into_manifest(manifest_path, acm_papers)
        if added == 0:
            LOGGER.info("No new papers to add")

    if args.stage in ("download", "all"):
        all_records = load_manifest(manifest_path)
        target_confs = set(args.conferences)
        acm_records = select_acm_manifest_records(
            all_records,
            target_confs,
            year=args.year,
        )
        acm_papers = [CorpusPaper.from_manifest_record(r) for r in acm_records]
        LOGGER.info("ACM papers to process: %d", len(acm_papers))

        chained_pool = (
            ChainedProxyPool.from_file(
                args.proxy_pool_file,
                first_hop_proxy=args.first_hop_proxy,
            )
            if args.proxy_pool_file is not None else None
        )
        downloaded = asyncio.run(download_acm_pdfs(
            acm_papers, pdf_root, args.proxy,
            batch_size=args.batch_size, delay_between=args.delay,
            limit=args.limit,
            challenge_retries=args.challenge_retries,
            chained_proxy_pool=chained_pool,
            xray_executable=args.xray_executable,
            max_pdfs_per_session=args.session_size,
        ))

        ok = sum(1 for p in downloaded if p.pdf_path)
        fail = sum(1 for p in downloaded if p.download_error)
        LOGGER.info("Final: %d OK, %d failed out of %d", ok, fail, len(downloaded))

        # Update manifest
        record_map = {r["paper_id"]: r for r in all_records}
        for p in downloaded:
            rec = record_map.get(p.paper_id)
            if rec:
                rec["pdf_path"] = p.pdf_path
                rec["downloaded_from_url"] = p.downloaded_from_url
                rec["pdf_resolution_status"] = p.pdf_resolution_status
                rec["pdf_resolution_note"] = p.pdf_resolution_note
                rec["download_error"] = p.download_error
        write_json_atomic(manifest_path, list(record_map.values()))
        LOGGER.info("Manifest updated")


if __name__ == "__main__":
    main()
