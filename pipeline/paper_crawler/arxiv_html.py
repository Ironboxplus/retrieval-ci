"""arXiv CS native-HTML harvest: OAI-PMH enumeration + two-hop HTML download.

arXiv renders an HTML version (LaTeXML) of every TeX submission made on or
after 2023-12-01 when the conversion succeeds. This module:

1. enumerates the ``cs`` OAI-PMH set with the ``arXivRaw`` format (per-version
   dates and categories) through the single local first-hop proxy — the
   sanctioned metadata channel, one connection, honouring 503 Retry-After;
2. keeps one SQLite row per paper with its latest version and download state;
3. downloads ``https://export.arxiv.org/html/<id>`` through the credentialed
   two-hop ``ChainedProxyPool`` and stores it gzip-compressed.

Only papers whose latest version is dated on/after the HTML cutoff are
queued; a 404 means arXiv has no HTML for that paper and is terminal.
"""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import os
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import quote

import aiohttp
from aiohttp_socks import ProxyConnector

from pipeline.download_control import (
    ChainedProxyDownloadError,
    ChainedProxyPool,
    ChainedTargetHTTPError,
)
from pipeline.http_retry import retry_after_seconds
from pipeline.paper_crawler.runtime_files import remove_file_if_present

OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"
HTML_BASE = "https://export.arxiv.org/html/"
PDF_BASE = "https://export.arxiv.org/pdf/"
HTML_CUTOFF = datetime(2023, 12, 1, tzinfo=timezone.utc)

_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    datestamp TEXT NOT NULL,
    categories TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    abstract TEXT NOT NULL,
    comments TEXT,
    license TEXT,
    latest_version TEXT NOT NULL,
    latest_version_date TEXT NOT NULL,
    first_version_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    html_version TEXT,
    raw_bytes INTEGER,
    gz_bytes INTEGER,
    fetched_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS papers_status ON papers(status);
CREATE TABLE IF NOT EXISTS harvest_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Columns added after the table existed; HarvestStore adds any missing one on open.
# pdf_*: PDF fallback for papers without HTML (status no_html).
#   pdf_status: NULL = not tried, ok, missing (404: withdrawn), error.
# latest_size/withdrawn/content_version(_date): see ArxivRawRecord. Rows from
#   before them have content_version NULL, read as latest_version via
#   CONTENT_VERSION_SQL, so there is no backfill.
# convert_*: the last HTML -> Markdown conversion of the row (HarvestStore.mark_convert).
#   convert_status: ok, rejected (the page deterministically has no usable article:
#   converting that version again cannot succeed) or error (crash, IO: worth a retry).
#   convert_version: the content version it was converted from; an outcome only
#   counts while it equals the row's CONTENT_VERSION_SQL, so a new version starts clean.
ADDED_COLUMNS = {
    "pdf_status": "TEXT",
    "pdf_attempts": "INTEGER NOT NULL DEFAULT 0",
    "pdf_bytes": "INTEGER",
    "pdf_fetched_at": "TEXT",
    "pdf_error": "TEXT",
    "latest_size": "TEXT",
    "withdrawn": "INTEGER NOT NULL DEFAULT 0",
    "content_version": "TEXT",
    "content_version_date": "TEXT",
    "convert_status": "TEXT",
    "convert_version": "TEXT",
    "convert_error": "TEXT",
}

# The version whose HTML a row should have: the last one with content.
CONTENT_VERSION_SQL = "COALESCE(content_version, latest_version)"
# The HTML of the row's current content version was converted and found unusable
# (see ADDED_COLUMNS convert_*): converting or selecting it again cannot help.
CONVERT_REJECTED_SQL = f"(convert_status='rejected' AND convert_version = {CONTENT_VERSION_SQL})"
CONTENT_VERSION_DATE_SQL = "COALESCE(content_version_date, latest_version_date)"

# A fresh submission's HTML can lag its OAI record; no_html is retried this long.
NO_HTML_RETRY_DAYS = 3

# Total time for one HTML request. Measured on .88 through the exit-less route
# (80 papers, concurrency 3): p50 1.35 s, p90 3.5 s, max 12 s; but about 1 in
# 30-40 requests trickles its body, every read landing inside sock_read, so only
# this total ends it (2610.00004v1: 183 s = the old 180 s + a 3 s retry, where
# curl took 0.8 s). A body still trickling after a minute is cheaper to retry on
# a fresh connection than to wait out.
HTML_TIMEOUT_S = 60.0

# Rows eligible for download: latest version inside the HTML era.
ELIGIBLE_SQL = "latest_version_date >= ?"


@dataclass(frozen=True)
class ArxivRawRecord:
    arxiv_id: str
    datestamp: str
    categories: str
    title: str
    authors: str
    abstract: str
    comments: str | None
    license: str | None
    latest_version: str
    latest_version_date: str
    first_version_date: str
    # arXivRaw.xsd: every <version> has <date> and <size>; a withdrawal is a new
    # version whose size is "0kb" (the TeX source is removed). ``latest_size`` is
    # the newest version's size (None when the record omits it).
    latest_size: str | None
    # Last version that still has content: the one whose HTML/PDF to fetch.
    # A withdrawn paper keeps it (the unversioned URL 404s once the newest
    # version is empty); a revived paper moves it forward again.
    content_version: str
    content_version_date: str
    withdrawn: bool

    @property
    def primary_category(self) -> str:
        return self.categories.split()[0] if self.categories else ""


def _text(node: ET.Element | None) -> str:
    return " ".join((node.text or "").split()) if node is not None else ""


def _iso(rfc2822: str) -> str:
    return parsedate_to_datetime(rfc2822).astimezone(timezone.utc).isoformat()


def _is_empty_size(size: str | None) -> bool:
    return (size or "").strip().lower() == "0kb"


def parse_list_records(xml_bytes: bytes) -> tuple[list[ArxivRawRecord], str | None, str | None]:
    """Parse one ListRecords page -> (records, resumption token, OAI error code)."""
    root = ET.fromstring(xml_bytes)
    error = root.find("oai:error", _NS)
    if error is not None:
        return [], None, error.get("code") or "unknown"
    records: list[ArxivRawRecord] = []
    for rec in root.iterfind(".//oai:record", _NS):
        header = rec.find("oai:header", _NS)
        if header is None or header.get("status") == "deleted":
            continue
        raw = rec.find("oai:metadata/raw:arXivRaw", _NS)
        if raw is None:
            continue
        versions = [
            (v.get("version") or "", _iso(_text(v.find("raw:date", _NS))),
             _text(v.find("raw:size", _NS)) or None)
            for v in raw.iterfind("raw:version", _NS)
        ]
        if not versions:
            continue
        # Withdrawal is read from the version sizes, not <comments>: comments are
        # free text and unreliable (2609.10457 carries a "withdrawn" comment
        # although its v3 is a full revival with 27108kb).
        with_content = [v for v in versions if not _is_empty_size(v[2])]
        content = with_content[-1] if with_content else versions[0]
        records.append(ArxivRawRecord(
            arxiv_id=_text(raw.find("raw:id", _NS)),
            datestamp=_text(header.find("oai:datestamp", _NS)),
            categories=_text(raw.find("raw:categories", _NS)),
            title=_text(raw.find("raw:title", _NS)),
            authors=_text(raw.find("raw:authors", _NS)),
            abstract=_text(raw.find("raw:abstract", _NS)),
            comments=_text(raw.find("raw:comments", _NS)) or None,
            license=_text(raw.find("raw:license", _NS)) or None,
            latest_version=versions[-1][0],
            latest_version_date=versions[-1][1],
            first_version_date=versions[0][1],
            latest_size=versions[-1][2],
            content_version=content[0],
            content_version_date=content[1],
            withdrawn=_is_empty_size(versions[-1][2]),
        ))
    token_node = root.find(".//oai:resumptionToken", _NS)
    token = (token_node.text or "").strip() if token_node is not None else ""
    return records, token or None, None


class HarvestStore:
    """SQLite state for the harvest; one writer (the asyncio loop)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=120)  # enumerate and download run as two processes
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        have = {row[1] for row in self.db.execute("PRAGMA table_info(papers)")}
        for name, decl in ADDED_COLUMNS.items():
            if name not in have:
                self.db.execute(f"ALTER TABLE papers ADD COLUMN {name} {decl}")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def get_state(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM harvest_state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str | None) -> None:
        if value is None:
            self.db.execute("DELETE FROM harvest_state WHERE key=?", (key,))
        else:
            self.db.execute(
                "INSERT INTO harvest_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
        self.db.commit()

    def upsert(self, records: list[ArxivRawRecord]) -> None:
        """Insert or refresh metadata. A new *content* version re-queues a
        paper that was already fetched, so the stored HTML tracks arXiv; a
        withdrawal (newest version empty, content version unchanged) only
        updates the withdrawn flag, and a revival is a new content version.
        In an UPDATE every right-hand side sees the pre-update row."""
        changed = "COALESCE(papers.content_version, papers.latest_version) != excluded.content_version"
        self.db.executemany(
            f"""
            INSERT INTO papers(arxiv_id, datestamp, categories, primary_category, title, authors,
                               abstract, comments, license, latest_version, latest_version_date,
                               first_version_date, latest_size, withdrawn, content_version,
                               content_version_date)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                datestamp=excluded.datestamp,
                categories=excluded.categories,
                primary_category=excluded.primary_category,
                title=excluded.title,
                authors=excluded.authors,
                abstract=excluded.abstract,
                comments=excluded.comments,
                license=excluded.license,
                status=CASE WHEN {changed} THEN 'pending' ELSE papers.status END,
                attempts=CASE WHEN {changed} THEN 0 ELSE papers.attempts END,
                pdf_status=CASE WHEN {changed} THEN NULL ELSE papers.pdf_status END,
                pdf_attempts=CASE WHEN {changed} THEN 0 ELSE papers.pdf_attempts END,
                latest_version=excluded.latest_version,
                latest_version_date=excluded.latest_version_date,
                first_version_date=excluded.first_version_date,
                latest_size=excluded.latest_size,
                withdrawn=excluded.withdrawn,
                content_version=excluded.content_version,
                content_version_date=excluded.content_version_date
            """,
            [
                (r.arxiv_id, r.datestamp, r.categories, r.primary_category, r.title, r.authors,
                 r.abstract, r.comments, r.license, r.latest_version, r.latest_version_date,
                 r.first_version_date, r.latest_size, int(r.withdrawn), r.content_version,
                 r.content_version_date)
                for r in records
            ],
        )
        self.db.commit()

    def pending(self, *, max_attempts: int, limit: int | None = None,
                scope: tuple[str, list[object]] | None = None,
                no_html_retry_days: int = NO_HTML_RETRY_DAYS) -> list[tuple[str, str]]:
        """Rows to download as (arxiv_id, content version), newest first.
        ``scope`` is an extra (WHERE fragment, params) pair, e.g. one month
        slice from arxiv_corpus.slice_sql.

        Besides pending/retryable-error rows: ``no_html`` whose content version
        is younger than ``no_html_retry_days`` (arXiv converts to HTML some time
        after submission), and ``ok`` rows holding an older version's HTML."""
        extra, extra_params = scope if scope else ("1=1", [])
        retry_since = (datetime.now(timezone.utc) - timedelta(days=no_html_retry_days)).isoformat()
        sql = (
            f"SELECT arxiv_id, {CONTENT_VERSION_SQL} FROM papers WHERE {ELIGIBLE_SQL} AND {extra} "
            "AND (status='pending' OR (status='error' AND attempts < ?) "
            f"OR (status='no_html' AND {CONTENT_VERSION_DATE_SQL} >= ?) "
            f"OR (status='ok' AND html_version IS NOT NULL AND html_version != {CONTENT_VERSION_SQL})) "
            "ORDER BY latest_version_date DESC"
        )
        params: list[object] = [HTML_CUTOFF.isoformat(), *extra_params, max_attempts, retry_since]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [(row[0], row[1]) for row in self.db.execute(sql, params)]

    def pdf_pending(self, *, max_attempts: int, limit: int | None = None,
                    scope: tuple[str, list[object]] | None = None) -> list[str]:
        """Papers without HTML whose PDF is still to fetch, newest first."""
        extra, extra_params = scope if scope else ("1=1", [])
        sql = (
            f"SELECT arxiv_id FROM papers WHERE {ELIGIBLE_SQL} AND {extra} AND status='no_html' "
            "AND (pdf_status IS NULL OR (pdf_status='error' AND pdf_attempts < ?)) "
            "ORDER BY latest_version_date DESC"
        )
        params: list[object] = [HTML_CUTOFF.isoformat(), *extra_params, max_attempts]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [row[0] for row in self.db.execute(sql, params)]

    def mark_pdf(self, arxiv_id: str, status: str, *, pdf_bytes: int | None = None,
                 error: str | None = None) -> None:
        self.db.execute(
            """
            UPDATE papers SET pdf_status=?, pdf_attempts=pdf_attempts+1, pdf_bytes=?,
                              pdf_fetched_at=?, pdf_error=?
            WHERE arxiv_id=?
            """,
            (status, pdf_bytes, datetime.now(timezone.utc).isoformat(), error, arxiv_id),
        )
        self.db.commit()

    def mark_convert(self, outcomes: list[tuple[str, str, str, str | None]]) -> None:
        """Record conversion outcomes as (arxiv_id, status, content version, error)."""
        self.db.executemany(
            "UPDATE papers SET convert_status=?, convert_version=?, convert_error=? WHERE arxiv_id=?",
            [(status, version, error, arxiv_id) for arxiv_id, status, version, error in outcomes],
        )
        self.db.commit()

    def mark(self, arxiv_id: str, status: str, *, html_version: str | None = None,
             raw_bytes: int | None = None, gz_bytes: int | None = None,
             error: str | None = None) -> None:
        self.db.execute(
            """
            UPDATE papers SET status=?, attempts=attempts+1, html_version=?, raw_bytes=?,
                              gz_bytes=?, fetched_at=?, error=?
            WHERE arxiv_id=?
            """,
            (status, html_version, raw_bytes, gz_bytes,
             datetime.now(timezone.utc).isoformat(), error, arxiv_id),
        )
        self.db.commit()

    def summary(self, *, no_html_retry_days: int = NO_HTML_RETRY_DAYS) -> dict[str, object]:
        cutoff = HTML_CUTOFF.isoformat()
        retry_since = (datetime.now(timezone.utc) - timedelta(days=no_html_retry_days)).isoformat()
        total = self.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        eligible = self.db.execute(f"SELECT COUNT(*) FROM papers WHERE {ELIGIBLE_SQL}", (cutoff,)).fetchone()[0]
        by_status = dict(self.db.execute(
            f"SELECT status, COUNT(*) FROM papers WHERE {ELIGIBLE_SQL} GROUP BY status", (cutoff,),
        ).fetchall())
        raw, gz = self.db.execute(
            "SELECT COALESCE(SUM(raw_bytes),0), COALESCE(SUM(gz_bytes),0) FROM papers WHERE status='ok'",
        ).fetchone()
        return {
            "records": total,
            "eligible": eligible,
            "by_status": by_status,
            "raw_bytes": raw,
            "gz_bytes": gz,
            "oai_from": self.get_state("oai_from"),
            "oai_token": bool(self.get_state("oai_token")),
            "oai_done": self.get_state("oai_done"),
            "withdrawn": self.db.execute(
                f"SELECT COUNT(*) FROM papers WHERE {ELIGIBLE_SQL} AND withdrawn=1", (cutoff,),
            ).fetchone()[0],
            "no_html_retryable": self.db.execute(
                f"SELECT COUNT(*) FROM papers WHERE {ELIGIBLE_SQL} AND status='no_html' "
                f"AND {CONTENT_VERSION_DATE_SQL} >= ?", (cutoff, retry_since),
            ).fetchone()[0],
            "pdf_by_status": dict(self.db.execute(
                f"SELECT COALESCE(pdf_status, 'pending'), COUNT(*) FROM papers "
                f"WHERE {ELIGIBLE_SQL} AND status='no_html' GROUP BY 1", (cutoff,),
            ).fetchall()),
            "pdf_bytes": self.db.execute(
                "SELECT COALESCE(SUM(pdf_bytes),0) FROM papers WHERE pdf_status='ok'").fetchone()[0],
        }


async def harvest_oai(
    store: HarvestStore,
    *,
    first_hop_proxy: str | None,
    set_spec: str = "cs",
    from_date: str = "2023-12-01",
    log: Callable[[str], None] = print,
    max_pages: int | None = None,
    endpoint: str = OAI_ENDPOINT,
    page_delay_s: float = 3.0,
    retry_floor_s: float = 5.0,
) -> dict[str, int]:
    """Walk ListRecords to the end, resumable across process restarts.
    Returns ``{"pages", "records"}`` fetched by this call.

    Incremental watermark: a walk that reaches the end stores the newest
    datestamp seen as ``oai_from``, and the next walk (no resumption token)
    starts there; failing that, from the newest stored datestamp, failing that,
    ``from_date``. OAI-PMH 2.0 section 2.7.1: ``from`` is inclusive and day
    granular, so the boundary day is fetched again and the upsert absorbs it.
    A walk cut short by ``max_pages`` leaves the watermark alone.

    Resumption tokens expire (arXiv issues them until the next UTC midnight);
    on ``badResumptionToken`` the walk restarts from the newest datestamp
    already stored. ``first_hop_proxy=None`` connects directly (tests only).
    """
    def newest_datestamp() -> str | None:
        return store.db.execute("SELECT MAX(datestamp) FROM papers").fetchone()[0]

    store.set_state("oai_done", None)
    token = store.get_state("oai_token")
    start_from = store.get_state("oai_from") or newest_datestamp() or from_date
    pages = 0
    total_records = 0
    connector = ProxyConnector.from_url(first_hop_proxy) if first_hop_proxy else None
    timeout = aiohttp.ClientTimeout(total=300, sock_connect=20, sock_read=180)
    log(f"oai walk: {'resuming token' if token else f'from {start_from}'} (set={set_spec})")
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
        while True:
            if token:
                url = f"{endpoint}?verb=ListRecords&resumptionToken={quote(token, safe='')}"
            else:
                url = f"{endpoint}?verb=ListRecords&metadataPrefix=arXivRaw&set={set_spec}&from={start_from}"
            body: bytes | None = None
            for attempt in range(8):
                try:
                    async with session.get(url) as resp:
                        if resp.status == 503:
                            wait = retry_after_seconds(resp.headers.get("Retry-After"))
                            log(f"oai 503, retry after {wait:.0f}s")
                            await asyncio.sleep(min(max(wait, retry_floor_s), 600))
                            continue
                        resp.raise_for_status()
                        body = await resp.read()
                        break
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    backoff = min(600, 15 * 2 ** attempt)
                    log(f"oai request failed ({type(exc).__name__}), retry in {backoff}s")
                    await asyncio.sleep(backoff)
            if body is None:
                raise RuntimeError("OAI-PMH request kept failing; state saved, rerun to resume")
            records, next_token, error = parse_list_records(body)
            if error == "badResumptionToken":
                start_from = newest_datestamp() or start_from
                store.set_state("oai_token", None)
                token = None
                log(f"resumption token expired; restarting from {start_from}")
                continue
            if error == "noRecordsMatch":  # OAI-PMH 2.0 section 3.5: nothing newer, a finished walk
                next_token = None
            elif error:
                raise RuntimeError(f"OAI-PMH error {error}")
            store.upsert(records)
            pages += 1
            total_records += len(records)
            token = next_token
            store.set_state("oai_token", token)
            summary = store.summary()
            log(f"oai page {pages}: +{len(records)} records, total {summary['records']:,}, "
                f"eligible {summary['eligible']:,}")
            if not token:
                newest = newest_datestamp()
                if newest:
                    store.set_state("oai_from", newest)
                store.set_state("oai_done", datetime.now(timezone.utc).isoformat())
                return {"pages": pages, "records": total_records}
            if max_pages and pages >= max_pages:
                return {"pages": pages, "records": total_records}
            await asyncio.sleep(page_delay_s)


def html_path(root: Path, arxiv_id: str) -> Path:
    """``html/<yymm>/<id>.html.gz``; old-style ids (``cs/0101001``) keep their
    archive as the folder and swap the slash."""
    if "/" in arxiv_id:
        archive, number = arxiv_id.split("/", 1)
        return root / "html" / archive / f"{number}.html.gz"
    return root / "html" / arxiv_id.split(".", 1)[0] / f"{arxiv_id}.html.gz"


def pdf_path(root: Path, arxiv_id: str) -> Path:
    """``pdf/<yymm>/<id>.pdf``; old-style ids under their archive name.
    Stored as-is: PDFs are already compressed."""
    if "/" in arxiv_id:
        archive, number = arxiv_id.split("/", 1)
        return root / "pdf" / archive / f"{number}.pdf"
    return root / "pdf" / arxiv_id.split(".", 1)[0] / f"{arxiv_id}.pdf"


def is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def is_arxiv_html(path: Path) -> bool:
    try:
        head = path.read_bytes()[:400_000]
    except OSError:
        return False
    return b"ltx_document" in head or b"ltx_page_main" in head


def _html_version(raw: bytes, arxiv_id: str) -> str | None:
    marker = arxiv_id.encode() + b"v"
    pos = raw.find(marker)
    if pos < 0:
        return None
    digits = b""
    for byte in raw[pos + len(marker): pos + len(marker) + 4]:
        if 48 <= byte <= 57:
            digits += bytes([byte])
        else:
            break
    return f"v{digits.decode()}" if digits else None


async def fetch_one(pool: ChainedProxyPool, root: Path, arxiv_id: str, version: str,
                    *, max_exits: int,
                    timeout_seconds: float = HTML_TIMEOUT_S) -> tuple[str, dict[str, object]]:
    """Download one paper's HTML at ``version`` (its content version; the
    unversioned URL would 404 for a paper whose newest version is a withdrawal).
    Returns (status, fields for HarvestStore.mark)."""
    dest = html_path(root, arxiv_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.part")
    try:
        await pool.download_to_path(
            HTML_BASE + arxiv_id + version,
            tmp,
            max_attempts=max_exits,
            validator=is_arxiv_html,
            # export.arxiv.org serves gzip (~1/7.6 of the bytes on the wire, which
            # the metered exit pool pays for); aiohttp decodes it transparently.
            headers={"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                     "Accept-Encoding": "gzip"},
            timeout_seconds=timeout_seconds,
        )
    except ChainedTargetHTTPError as exc:
        remove_file_if_present(tmp)
        if exc.status_code in {404, 410}:
            return "no_html", {"error": f"HTTP {exc.status_code}"}
        return "error", {"error": str(exc)[:300]}
    except ChainedProxyDownloadError as exc:
        remove_file_if_present(tmp)
        return "error", {"error": str(exc)[:300]}
    raw = tmp.read_bytes()
    gz = gzip.compress(raw, compresslevel=6)
    gz_tmp = dest.with_name(dest.name + f".{os.getpid()}.gz.part")
    gz_tmp.write_bytes(gz)
    os.replace(gz_tmp, dest)
    remove_file_if_present(tmp)
    return "ok", {"html_version": _html_version(raw, arxiv_id) or version,
                  "raw_bytes": len(raw), "gz_bytes": len(gz)}


async def fetch_pdf(pool: ChainedProxyPool, root: Path, arxiv_id: str,
                    *, max_exits: int) -> tuple[str, dict[str, object]]:
    """Download the latest PDF of a paper without HTML. Returns (pdf_status,
    fields for HarvestStore.mark_pdf); 404/410 (withdrawn) -> missing."""
    dest = pdf_path(root, arxiv_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.part")
    try:
        await pool.download_to_path(
            PDF_BASE + arxiv_id, tmp, max_attempts=max_exits, validator=is_pdf,
            headers={"Accept": "application/pdf,*/*;q=0.8"}, timeout_seconds=600.0,
        )
    except ChainedTargetHTTPError as exc:
        remove_file_if_present(tmp)
        if exc.status_code in {404, 410}:
            return "missing", {"error": f"HTTP {exc.status_code}"}
        return "error", {"error": str(exc)[:300]}
    except ChainedProxyDownloadError as exc:
        remove_file_if_present(tmp)
        return "error", {"error": str(exc)[:300]}
    size = tmp.stat().st_size
    os.replace(tmp, dest)
    return "ok", {"pdf_bytes": size}


async def download_pending(
    store: HarvestStore,
    pool: ChainedProxyPool,
    root: Path,
    *,
    concurrency: int = 6,
    max_attempts: int = 3,
    max_exits: int = 12,
    limit: int | None = None,
    log: Callable[[str], None] = print,
    report_every: int = 200,
    follow: bool = False,
    follow_poll_seconds: float = 60.0,
    scope: tuple[str, list[object]] | None = None,
    probe_url: str | None = None,
    reprobe_check_seconds: float = 60.0,
    pdf_fallback: bool = False,
    pusher: "BatchPusherLike | None" = None,
    push_every_seconds: float = 120.0,
    no_html_retry_days: int = NO_HTML_RETRY_DAYS,
    html_timeout_s: float = HTML_TIMEOUT_S,
) -> dict[str, int]:
    """Download every pending paper. With ``follow`` the queue is refilled
    from SQLite while an enumeration process is still adding rows, and the
    run ends once the OAI walk is marked done and nothing new is pending.
    A paper is tried at most once per run; failures wait for the next run.

    With ``probe_url`` the pool is re-probed whenever fewer than
    ``concurrency`` exits are verified. The pool ranks a verified exit that
    failed once behind never-tried exits, so without this a long run drifts
    to one good exit plus timeouts on dead ones (measured: 0.85/s after two
    hours vs 2.0/s on a freshly probed pool).

    With ``pdf_fallback`` a paper whose HTML is missing gets its PDF in the
    same worker (so the request cap still holds), and papers already marked
    ``no_html`` whose PDF is not fetched are queued too.

    With ``pusher`` (arxiv_batches.BatchPusher) every batch whose papers are
    all resolved is handed to the ingest host while downloading continues,
    and once more after the last paper."""
    # (kind, arxiv_id, version): html items are pinned to a content version, pdf items are not.
    queue: asyncio.Queue[tuple[str, str, str | None]] = asyncio.Queue()
    seen: set[tuple[str, str, str | None]] = set()

    def refill() -> int:
        added = 0
        items = [("html", arxiv_id, version) for arxiv_id, version in
                 store.pending(max_attempts=max_attempts, limit=limit, scope=scope,
                               no_html_retry_days=no_html_retry_days)]
        if pdf_fallback:
            items += [("pdf", arxiv_id, None) for arxiv_id in
                      store.pdf_pending(max_attempts=max_attempts, limit=limit, scope=scope)]
        for item in items:
            if item not in seen:
                seen.add(item)
                queue.put_nowait(item)
                added += 1
        return added

    total = refill()
    counts = {"ok": 0, "no_html": 0, "error": 0}
    pdf_counts = {"ok": 0, "missing": 0, "error": 0}
    items_done = 0
    started = time.monotonic()
    log(f"download start: {total:,} queued, concurrency {concurrency}, follow={follow}")
    refill_lock = asyncio.Lock()

    async def next_id() -> tuple[str, str, str | None] | None:
        nonlocal total
        while True:
            try:
                return queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if not follow or (limit and len(seen) >= limit):
                return None
            async with refill_lock:
                if not queue.empty():
                    continue
                enumeration_done = store.get_state("oai_done") is not None
                added = refill()
                total += added
                if added:
                    log(f"refill: +{added:,} newly enumerated papers queued")
                    continue
                if enumeration_done:
                    return None
                await asyncio.sleep(follow_poll_seconds)

    async def get_pdf(arxiv_id: str) -> None:
        try:
            status, fields = await fetch_pdf(pool, root, arxiv_id, max_exits=max_exits)
        except Exception as exc:  # one bad paper must not kill the worker
            status, fields = "error", {"error": f"{type(exc).__name__}: {exc}"[:300]}
        store.mark_pdf(arxiv_id, status, **fields)  # type: ignore[arg-type]
        pdf_counts[status] += 1

    async def worker() -> None:
        nonlocal items_done
        while True:
            item = await next_id()
            if item is None:
                return
            kind, arxiv_id, version = item
            if kind == "pdf":
                await get_pdf(arxiv_id)
            else:
                try:
                    status, fields = await fetch_one(pool, root, arxiv_id, version, max_exits=max_exits,
                                                    timeout_seconds=html_timeout_s)
                except Exception as exc:  # one bad paper must not kill the worker
                    status, fields = "error", {"error": f"{type(exc).__name__}: {exc}"[:300]}
                store.mark(arxiv_id, status, **fields)  # type: ignore[arg-type]
                counts[status] += 1
                if status == "no_html" and pdf_fallback:
                    await get_pdf(arxiv_id)
            items_done += 1
            done = items_done
            if done % report_every == 0 or done == total:
                rate = done / max(time.monotonic() - started, 1e-6)
                eta_h = (total - done) / rate / 3600 if rate else float("inf")
                pdf = (f" pdf ok={pdf_counts['ok']:,} missing={pdf_counts['missing']:,} "
                       f"error={pdf_counts['error']:,}") if pdf_fallback else ""
                log(f"{done:,}/{total:,} ok={counts['ok']:,} no_html={counts['no_html']:,} "
                    f"error={counts['error']:,}{pdf} {rate:.2f}/s eta {eta_h:.1f}h "
                    f"healthy_exits={pool.healthy_count}")

    async def keep_pool_healthy() -> None:
        while True:
            await asyncio.sleep(reprobe_check_seconds)
            if pool.healthy_count >= concurrency:
                continue
            before = pool.healthy_count
            try:
                report = await pool.probe_pdf(
                    probe_url, required_healthy=2 * concurrency,
                    max_candidates=pool.configured_count, concurrency=concurrency,
                )
            except Exception as exc:  # the first hop may blink; try again next round
                log(f"pool re-probe failed: {type(exc).__name__}: {exc}"[:300])
                continue
            log(f"pool re-probe: healthy {before} -> {report.healthy_count} "
                f"(attempted {report.attempted_count}, {report.elapsed_seconds:.0f}s)")

    async def keep_pushing() -> None:
        while True:
            await pusher.round()
            await asyncio.sleep(push_every_seconds)

    workers = asyncio.gather(*(worker() for _ in range(concurrency)))
    background = []
    if probe_url:
        background.append(asyncio.ensure_future(keep_pool_healthy()))
    if pusher is not None:
        background.append(asyncio.ensure_future(keep_pushing()))
    try:
        await workers
    finally:
        for task in background:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    if pusher is not None:
        await pusher.round()  # batches the last papers completed
    if pdf_fallback:
        return {**counts, **{f"pdf_{k}": v for k, v in pdf_counts.items()}}
    return counts


class BatchPusherLike:  # the one method download_pending needs (arxiv_batches.BatchPusher)
    async def round(self) -> int: ...


def iter_status_lines(summary: dict[str, object]) -> Iterator[str]:
    yield f"OAI records: {summary['records']:,}  eligible (latest version >= 2023-12-01): {summary['eligible']:,}"
    yield f"OAI walk: done={summary['oai_done']} resumable_token={summary['oai_token']} from={summary['oai_from']}"
    by_status = summary["by_status"]
    assert isinstance(by_status, dict)
    yield "status: " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_status.items()))
    yield f"withdrawn: {summary['withdrawn']:,}  no_html still retryable: {summary['no_html_retryable']:,}"
    yield f"stored: raw {int(summary['raw_bytes']) / 1e9:.2f} GB, gzip {int(summary['gz_bytes']) / 1e9:.2f} GB"
    pdf = summary.get("pdf_by_status") or {}
    assert isinstance(pdf, dict)
    yield ("pdf fallback (no_html papers): " + ", ".join(f"{k}={v:,}" for k, v in sorted(pdf.items()))
           + f"; stored {int(summary.get('pdf_bytes') or 0) / 1e9:.2f} GB")
