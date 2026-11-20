"""Back-fill / refresh ``papers.track_type`` from authoritative DBLP TOC sections.

Why
---
ACM venues were ingested via the section-blind ``DblpAcmProceedingsSource`` (whole
DBLP TOC, DOI-prefix filter only), so workshop / tutorial / demo / poster /
doctoral-consortium / keynote / late-breaking entries landed tagged ``track_type=
'main'`` alongside real research papers (project memory:
``project_dblp_acm_section_blind_contamination``). Per the user's decision we do
**not** delete those rows -- we *relabel* them so retrieval can flag "not a
main-conference paper (low value)".

What it does
------------
For each target ACM venue present in the DB, for each year, it fetches the DBLP
**HTML** TOC (which -- unlike the JSON API -- exposes section headings), classifies
every entry via ``classify_track``, and UPDATEs ``papers.track_type`` by DOI match.
Non-destructive (only the label changes). **Dry-run by default**; pass ``--apply``
to write. Reusable as the standard post-crawl tagging step for any ACM venue.

Usage
-----
    python scripts/tag_paper_tracks.py                 # dry-run, default venues
    python scripts/tag_paper_tracks.py --apply         # write to DB
    python scripts/tag_paper_tracks.py SIGIR KDD --apply
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db.database import DB_PATH  # noqa: E402
from pipeline.paper_crawler.ccf_a_main_corpus import (  # noqa: E402
    build_sources,
    parse_dblp_toc_tracks,
)
from pipeline.paper_crawler.track_classifier import MAIN_TRACKS  # noqa: E402

DEFAULT_VENUES = ["SIGIR", "KDD", "SIGMOD", "WWW"]
CACHE_DIR = Path(tempfile.gettempdir()) / "dblp_toc_cache"


def fetch_toc_html(toc_key: str, proxy: str | None, *, retries: int = 4) -> str | None:
    """Fetch (and cache) a DBLP HTML TOC. Cached to temp so reruns don't re-hit DBLP."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (toc_key.replace("/", "_") + ".html")
    if cache.exists() and cache.stat().st_size > 1000:
        return cache.read_text(encoding="utf-8", errors="ignore")
    url = f"https://dblp.org/db/{toc_key}.html"
    headers = {"User-Agent": "Mozilla/5.0 (track-tagger)"}
    delay = 4.0
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(proxy=proxy, timeout=45.0, follow_redirects=True, headers=headers) as c:
                r = c.get(url)
                if r.status_code == 404:
                    print(f"    [{toc_key}] 404 (no such TOC) -> skip")
                    return None
                r.raise_for_status()
                cache.write_text(r.text, encoding="utf-8")
                return r.text
        except Exception as exc:  # noqa: BLE001
            print(f"    [{toc_key}] fetch {attempt}/{retries} failed: {type(exc).__name__} {str(exc)[:50]}")
            time.sleep(delay)
            delay *= 2
    return None


def resolve_doi(paper_id: str, doi: str | None) -> str | None:
    """Lower-cased ACM DOI for a paper row: prefer the doi column, else derive from
    the paper_id (``{slug}-{year}-{doi-tail}`` -> ``10.1145/{doi-tail}``)."""
    if doi and doi.lower().startswith("10."):
        return doi.lower()
    parts = (paper_id or "").split("-", 2)
    if len(parts) == 3 and "." in parts[2] and parts[2][0].isdigit():
        return f"10.1145/{parts[2]}".lower()
    return None


def process_venue(conn, src, venue: str, proxy: str | None, apply: bool) -> dict:
    cur = conn.cursor()
    years = [r[0] for r in cur.execute(
        "SELECT DISTINCT year FROM papers WHERE venue=? AND year IS NOT NULL ORDER BY year", (venue,)
    ).fetchall()]
    if not years:
        print(f"\n## {venue}: (no papers in DB)")
        return {}
    venue_tally: dict[str, int] = {}
    for year in years:
        try:
            toc_keys = src._toc_keys_for_year(year)
        except Exception as exc:  # noqa: BLE001
            print(f"\n## {venue} {year}: cannot resolve TOC keys ({exc}) -> skip")
            continue
        doi_track: dict[str, str] = {}
        manifest: dict[str, dict[str, int]] = {}
        for key in toc_keys:
            html = fetch_toc_html(key, proxy)
            if not html:
                continue
            dt, mf = parse_dblp_toc_tracks(html)
            doi_track.update(dt)
            for sec, counts in mf.items():
                manifest.setdefault(sec, {})
                for t, n in counts.items():
                    manifest[sec][t] = manifest[sec].get(t, 0) + n
            time.sleep(3)

        rows = cur.execute(
            "SELECT paper_id, doi, track_type FROM papers WHERE venue=? AND year=?", (venue, year)
        ).fetchall()
        changes: list[tuple[str, str]] = []
        by_track: dict[str, int] = {}
        matched = unmatched = 0
        for paper_id, doi, current in rows:
            d = resolve_doi(paper_id, doi)
            new_track = doi_track.get(d) if d else None
            if new_track is None:
                unmatched += 1
                continue
            matched += 1
            by_track[new_track] = by_track.get(new_track, 0) + 1
            venue_tally[new_track] = venue_tally.get(new_track, 0) + 1
            # Leave main-conference papers tagged 'main' (untouched); only the genuinely
            # non-main entries get relabelled, so the write set == the contamination found.
            # (desired==current reverts any earlier mis-flag back to 'main' -> idempotent.)
            desired = new_track if new_track not in MAIN_TRACKS else "main"
            if desired != current:
                changes.append((desired, paper_id))

        non_main = sum(n for t, n in by_track.items() if t not in MAIN_TRACKS)
        main_n = matched - non_main
        print(f"\n## {venue} {year}: {len(rows)} papers | main {main_n} (kept 'main') "
              f"| NON-MAIN {non_main} -> relabel | unmatched {unmatched} | db-writes {len(changes)}")
        print(f"   classified: {dict(sorted(by_track.items(), key=lambda kv: -kv[1]))}")
        # show only the non-main / unusual sections in the manifest (research sessions are noise)
        flagged_secs = {s: c for s, c in manifest.items()
                        if any(t not in MAIN_TRACKS for t in c)}
        for sec, counts in flagged_secs.items():
            print(f"      [section] {sec[:54]:54} -> {counts}")

        if apply and changes:
            cur.executemany("UPDATE papers SET track_type=? WHERE paper_id=?", changes)
            conn.commit()
            print(f"   APPLIED {len(changes)} updates.")
    return venue_tally


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("venues", nargs="*", default=None, help="venue titles (default: SIGIR KDD SIGMOD WWW)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--proxy", default="http://127.0.0.1:10808", help="HTTP(S) proxy for DBLP; '' to disable")
    args = ap.parse_args()
    proxy = args.proxy or None
    venues = args.venues or DEFAULT_VENUES

    print(f"DB = {DB_PATH}")
    print(f"mode = {'APPLY (writing)' if args.apply else 'DRY-RUN (no writes)'} | proxy = {proxy}")
    sources = build_sources()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        print(f"papers in DB = {total}")
        grand: dict[str, int] = {}
        for venue in venues:
            src = sources.get(venue)
            if src is None:
                print(f"\n## {venue}: not in source registry -> skip")
                continue
            vt = process_venue(conn, src, venue, proxy, args.apply)
            for t, n in vt.items():
                grand[t] = grand.get(t, 0) + n
        print("\n================ GRAND TOTAL (matched papers by track) ================")
        for t, n in sorted(grand.items(), key=lambda kv: -kv[1]):
            tag = "MAIN" if t in MAIN_TRACKS else "non-main"
            print(f"   {t:14} {n:6}  [{tag}]")
        print(f"   non-main total = {sum(n for t, n in grand.items() if t not in MAIN_TRACKS)}")
        if not args.apply:
            print("\n(DRY-RUN -- nothing written. Re-run with --apply to relabel.)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
