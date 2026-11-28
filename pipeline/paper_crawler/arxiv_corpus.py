"""arXiv CS corpus: month slices, HTML -> JSON/Markdown conversion, slice manifests.

Layout (both under ``settings.data_dir``):

    arxiv/cs_html/                    harvest root (see arxiv_html.py)
        arxiv_cs.sqlite3              OAI metadata + download state
        html/<yymm>/<id>.html.gz
    papers/arxiv_cs_corpus/           corpus root (what the ingest pipeline reads)
        doc/<month>/<id>.json.gz      structured document (arxiv_html_convert)
        markdown/arxiv/<month>/<id>.md
        manifests/arxiv_<slice>.json  CorpusPaper records for one slice
        reports/convert_<slice>.json

A slice is one arXiv submission month keyed ``YYYYMM`` (from the id prefix),
e.g. 202409. A batch is a fixed id-number range inside a month keyed
``YYYYMMPP``: batch PP holds ids ``YYMM.NNNNN`` with NNNNN in
``[PP*BATCH_SPAN, (PP+1)*BATCH_SPAN)`` (~2,000 CS papers). Ranges, not counts,
so the downloader that pushes a finished batch and the ingest task that
receives it agree on membership without talking to each other. Markdown and
doc files live in the month of the paper's own id, whatever key converted it;
manifests and reports per key (slice or batch). Papers first submitted before the HTML era (id prefix < 2312,
or old-style ``cs/0101001`` ids) but revised after it share one legacy slice
``LEGACY_SLICE`` so ~160 near-empty months do not each become a task.

``DELTA_KEY`` (0) is the daily-update scope: every 2026+ paper, narrowed per run
to an explicit id list. Its manifest and report are named by the run
(``arxiv_delta_<run_id>.json``, ``convert_delta_<run_id>.json``) so runs do not
overwrite each other.

One manifest per slice keeps every probe's JSON parse small (a year of
arXiv CS with abstracts is ~200 MB; a month is ~30 MB).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from config import settings as retrieval_settings
from pipeline.paper_crawler.arxiv_html import (
    CONTENT_VERSION_SQL,
    CONVERT_REJECTED_SQL,
    HTML_CUTOFF,
    HarvestStore,
    html_path,
)
from pipeline.paper_crawler.arxiv_html_convert import (
    NotLatexmlDocument,
    parse_arxiv_html,
    to_markdown,
)
from pipeline.paper_crawler.ccf_a_main_corpus import CorpusPaper
from pipeline.paper_crawler.runtime_files import write_json_atomic, write_text_atomic

ARXIV_VENUE = "arXiv"
PREPRINT = "preprint"
SOURCE_NAME = "arxiv_html"
LEGACY_SLICE = 202311  # "2023-11 and earlier": pre-HTML-era ids revised after 2023-12
FIRST_MONTH_SLICE = 202312
BATCH_SPAN = 4000  # id numbers per batch; CS is ~half of all arXiv ids -> ~2,000 papers
MAX_BATCH_PART = 99999 // BATCH_SPAN
DELTA_KEY = 0  # the daily update: ids chosen per run, not a month or id range
DELTA_MIN_ID_PREFIX = "2601"  # first YYMM of the delta scope (2026+ submissions)


def harvest_root() -> Path:
    return Path(os.environ.get("ARXIV_HARVEST_ROOT")
                or Path(retrieval_settings.data_dir) / "arxiv" / "cs_html").resolve()


def corpus_root() -> Path:
    # Absolute: manifest markdown paths must resolve from any working directory.
    return Path(os.environ.get("ARXIV_CORPUS_ROOT")
                or Path(retrieval_settings.data_dir) / "papers" / "arxiv_cs_corpus").resolve()


def slice_of(arxiv_id: str) -> int:
    """Month slice key for an arXiv id (``YYYYMM``)."""
    if "/" in arxiv_id:
        return LEGACY_SLICE
    yymm = arxiv_id.split(".", 1)[0]
    if len(yymm) != 4 or not yymm.isdigit():
        raise ValueError(f"not an arXiv id: {arxiv_id!r}")
    key = 200000 + int(yymm)  # 2409 -> 202409
    return key if key >= FIRST_MONTH_SLICE else LEGACY_SLICE


def is_batch_key(key: int) -> bool:
    return key >= 10_000_000


def month_of(key: int) -> int:
    """The month slice a slice-or-batch key belongs to."""
    return key // 100 if is_batch_key(key) else key


def batch_range(key: int) -> tuple[int, int]:
    """Inclusive id-number range of a batch key."""
    part = key % 100
    return part * BATCH_SPAN, (part + 1) * BATCH_SPAN - 1


def batch_of(arxiv_id: str) -> Optional[int]:
    """Batch key of a new-style id in the HTML era; None for legacy ids."""
    month = slice_of(arxiv_id)
    if month == LEGACY_SLICE:
        return None
    return month * 100 + int(arxiv_id.split(".", 1)[1]) // BATCH_SPAN


def slice_label(key: int) -> str:
    if key == DELTA_KEY:
        return "delta"
    if is_batch_key(key):
        return f"{slice_label(month_of(key))} #{key % 100}"
    return "≤2023-11" if key == LEGACY_SLICE else f"{key // 100}-{key % 100:02d}"


def slice_sql(key: int) -> tuple[str, list[object]]:
    """WHERE fragment selecting a slice's or batch's rows in the harvest ``papers`` table."""
    if key == DELTA_KEY:
        # New-style ids sort as strings in submission order ("2601.00001" >= "2601"),
        # and old-style "cs/0101001" ids are excluded by the slash.
        return "(arxiv_id NOT LIKE '%/%' AND arxiv_id >= ?)", [DELTA_MIN_ID_PREFIX]
    if is_batch_key(key):
        month, _ = slice_sql(month_of(key))
        lo, hi = batch_range(key)
        return (f"({month[1:-1]} AND CAST(substr(arxiv_id, 6) AS INTEGER) BETWEEN ? AND ?)",
                [f"{month_of(key) % 10000:04d}.", lo, hi])
    if key == LEGACY_SLICE:
        return "(arxiv_id LIKE '%/%' OR substr(arxiv_id, 1, 4) < '2312')", []
    if key < FIRST_MONTH_SLICE:
        raise ValueError(f"unknown slice {key}")
    return "(arxiv_id NOT LIKE '%/%' AND substr(arxiv_id, 1, 5) = ?)", [f"{key % 10000:04d}."]


def paper_id_for(arxiv_id: str) -> str:
    return "arxiv-" + arxiv_id.replace("/", "-")


def split_authors(raw: str) -> list[str]:
    """arXivRaw author strings: ``A, B, and C`` / ``A and B``."""
    text = re.sub(r"\s+", " ", raw or "").strip()
    if not text:
        return []
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class SlicePaths:
    """Where one key's outputs live. Markdown/doc paths depend only on the paper
    id; the manifest and report are per key, and per run for ``DELTA_KEY``."""

    root: Path
    key: int
    run_id: int | None = None  # required for DELTA_KEY, forbidden otherwise

    def __post_init__(self) -> None:
        if (self.key == DELTA_KEY) != (self.run_id is not None):
            raise ValueError("run_id goes with DELTA_KEY and only with it")

    @property
    def _stem(self) -> str:
        return f"delta_{self.run_id}" if self.key == DELTA_KEY else str(self.key)

    @property
    def markdown_dir(self) -> Path:
        """The month directory of a month/batch key; a delta run spans months,
        so there it is ``markdown(arxiv_id).parent`` per paper instead."""
        if self.key == DELTA_KEY:
            raise ValueError("a delta run has no single markdown directory")
        return self.root / "markdown" / "arxiv" / str(month_of(self.key))

    @property
    def doc_dir(self) -> Path:
        if self.key == DELTA_KEY:
            raise ValueError("a delta run has no single doc directory")
        return self.root / "doc" / str(month_of(self.key))

    @property
    def manifest(self) -> Path:
        return self.root / "manifests" / f"arxiv_{self._stem}.json"

    @property
    def report(self) -> Path:
        return self.root / "reports" / f"convert_{self._stem}.json"

    def markdown(self, arxiv_id: str) -> Path:
        month = month_of(slice_of(arxiv_id))
        return self.root / "markdown" / "arxiv" / str(month) / f"{arxiv_id.replace('/', '-')}.md"

    def doc(self, arxiv_id: str) -> Path:
        month = month_of(slice_of(arxiv_id))
        return self.root / "doc" / str(month) / f"{arxiv_id.replace('/', '-')}.json.gz"


@dataclass(frozen=True)
class HarvestRow:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    first_version_date: str
    latest_version: str
    status: str
    content_version: str  # the version the stored HTML is of (see ArxivRawRecord)
    withdrawn: bool
    convert_status: Optional[str] = None   # last conversion outcome (ADDED_COLUMNS convert_*)
    convert_version: Optional[str] = None
    convert_error: Optional[str] = None

    @property
    def rejected(self) -> bool:
        """The current version's HTML was converted and has no usable article."""
        return self.convert_status == CONVERT_REJECTED and self.convert_version == self.content_version


_ROW_COLUMNS = (
    "arxiv_id, title, authors, abstract, categories, first_version_date, latest_version, status, "
    f"{CONTENT_VERSION_SQL}, withdrawn, convert_status, convert_version, convert_error"
)
CONVERT_OK, CONVERT_REJECTED, CONVERT_ERROR = "ok", "rejected", "error"
CONVERT_MARK_BATCH = 500  # outcomes written to the harvest database at a time
SQLITE_MAX_IN = 900  # SQLite's default variable limit is 999 (<3.32); stay under it


def _select_rows(harvest_db: Path, where: str, params: list[object],
                 statuses: Iterable[str] | None) -> list[HarvestRow]:
    sql = f"SELECT {_ROW_COLUMNS} FROM papers WHERE {where}"
    args = list(params)
    if statuses:
        statuses = list(statuses)
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        args += statuses
    sql += " ORDER BY arxiv_id"
    db = sqlite3.connect(f"file:{harvest_db}?mode=ro", uri=True)
    try:
        return [HarvestRow(*row[:9], bool(row[9]), *row[10:]) for row in db.execute(sql, args)]
    finally:
        db.close()


def slice_rows(harvest_db: Path, key: int, *, statuses: Iterable[str] | None = None) -> list[HarvestRow]:
    """Eligible harvest rows of one slice (latest version inside the HTML era)."""
    where, params = slice_sql(key)
    return _select_rows(harvest_db, f"latest_version_date >= ? AND {where}",
                        [HTML_CUTOFF.isoformat(), *params], statuses)


def rows_by_ids(harvest_db: Path, ids: Iterable[str], *,
                statuses: Iterable[str] | None = None) -> list[HarvestRow]:
    """Harvest rows of exactly these ids (no era filter: the caller chose them),
    ordered by id. Unknown ids are skipped."""
    wanted = sorted(set(ids))
    rows: list[HarvestRow] = []
    for start in range(0, len(wanted), SQLITE_MAX_IN):
        chunk = wanted[start:start + SQLITE_MAX_IN]
        rows += _select_rows(harvest_db, f"arxiv_id IN ({','.join('?' * len(chunk))})", chunk, statuses)
    return sorted(rows, key=lambda r: r.arxiv_id)


def _convert_one(args: tuple[str, str, str, str]) -> tuple[str, str, Optional[str]]:
    """(arxiv_id, CONVERT_* outcome, error). A page with no usable article is
    ``rejected`` (converting it again gives the same answer); any other exception
    is an ``error`` (crash, IO), worth another try."""
    arxiv_id, html_file, doc_file, md_file = args
    try:
        raw = gzip.decompress(Path(html_file).read_bytes())
        doc = parse_arxiv_html(raw, arxiv_id)
        markdown = to_markdown(doc)
        if not any(b.kind == "paragraph" for b in doc.blocks):
            return arxiv_id, CONVERT_REJECTED, "no body paragraphs"
        Path(doc_file).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(doc_file + ".part")
        tmp.write_bytes(gzip.compress(json.dumps(doc.to_json(), ensure_ascii=False).encode("utf-8"), 6))
        os.replace(tmp, doc_file)
        Path(md_file).parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(Path(md_file), markdown)
        return arxiv_id, CONVERT_OK, None
    except NotLatexmlDocument as exc:
        return arxiv_id, CONVERT_REJECTED, str(exc)
    except Exception as exc:  # one bad page must not stop the slice
        return arxiv_id, CONVERT_ERROR, f"{type(exc).__name__}: {exc}"[:300]


def _needs_convert(html_file: Path, md_file: Path, doc_file: Path) -> bool:
    if not (md_file.exists() and doc_file.exists()):
        return True
    # A re-download (new version) replaces the HTML; convert again.
    return html_file.stat().st_mtime > min(md_file.stat().st_mtime, doc_file.stat().st_mtime)


def convert_rows(rows: list[HarvestRow], paths: SlicePaths, *, hroot: Path, workers: int = 4,
                 limit: int = 0, log=print) -> dict[str, object]:
    """Convert the downloaded HTML of ``rows``; idempotent, re-converts only
    papers whose HTML is newer than their outputs.

    Every outcome is recorded per paper in the harvest database (HarvestStore.mark_convert),
    a batch at a time. A paper whose current version was ``rejected`` is not converted
    again; it is reported under ``rejected`` and left out of ``convertible``, the
    denominator of the markdown ratio. ``errors`` are the other papers without Markdown."""
    label = slice_label(paths.key)
    if limit:
        rows = rows[:limit]
    jobs = []
    rejected: dict[str, str] = {}
    for row in rows:
        if row.rejected:
            rejected[row.arxiv_id] = row.convert_error or "rejected"
            continue
        html_file = html_path(hroot, row.arxiv_id)
        if not html_file.exists():
            continue
        md_file, doc_file = paths.markdown(row.arxiv_id), paths.doc(row.arxiv_id)
        if _needs_convert(html_file, md_file, doc_file):
            jobs.append((row.arxiv_id, str(html_file), str(doc_file), str(md_file)))
    version_of = {row.arxiv_id: row.content_version for row in rows}
    failures: dict[str, str] = {}
    previous = convert_failures(paths)
    done = 0
    log(f"convert {label}: {len(rows):,} downloaded, {len(jobs):,} to convert, "
        f"{len(rejected):,} rejected earlier, workers={workers}")
    if jobs:
        store = HarvestStore(hroot / "arxiv_cs.sqlite3")
        outcomes: list[tuple[str, str, str, Optional[str]]] = []
        try:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
                for arxiv_id, outcome, error in pool.map(_convert_one, jobs, chunksize=16):
                    done += 1
                    if error:
                        failures[arxiv_id] = error
                    if outcome == CONVERT_REJECTED:
                        rejected[arxiv_id] = error or "rejected"
                    outcomes.append((arxiv_id, outcome, version_of[arxiv_id], error))
                    if len(outcomes) >= CONVERT_MARK_BATCH:
                        store.mark_convert(outcomes)
                        outcomes = []
                    if done % 2000 == 0:
                        log(f"convert {label}: {done:,}/{len(jobs):,} failures={len(failures)}")
        finally:
            if outcomes:
                store.mark_convert(outcomes)  # what finished before a crash is not lost
            store.close()
    converted_ids = {j[0] for j in jobs}
    merged = {k: v for k, v in previous.items() if k not in converted_ids}
    merged.update(failures)
    merged.update(rejected)
    report = {
        "slice": paths.key, "downloaded": len(rows), "converted_now": len(jobs) - len(failures),
        "failures": merged,
        "rejected": sorted(rejected),
        "errors": sorted(set(merged) - set(rejected)),
        "convertible": len(rows) - len(rejected),
        "markdown": sum(1 for r in rows if paths.markdown(r.arxiv_id).exists()),
    }
    write_json_atomic(paths.report, report)
    log(f"convert {label}: markdown {report['markdown']:,}/{report['convertible']:,} convertible "
        f"({len(rows):,} downloaded), rejected {len(rejected):,}, errors {len(report['errors'])}")
    return report


def convert_slice(key: int, *, hroot: Path | None = None, croot: Path | None = None,
                  workers: int = 4, limit: int = 0, log=print) -> dict[str, object]:
    """Convert every downloaded HTML of a month slice or batch."""
    hroot = hroot or harvest_root()
    return convert_rows(slice_rows(hroot / "arxiv_cs.sqlite3", key, statuses=["ok"]),
                        SlicePaths((croot or corpus_root()).resolve(), key),
                        hroot=hroot, workers=workers, limit=limit, log=log)


def rejected_ids(harvest_db: Path, ids: Iterable[str]) -> set[str]:
    """Which of these papers were found unusable at their current content version."""
    wanted = sorted(set(ids))
    found: set[str] = set()
    db = sqlite3.connect(f"file:{harvest_db}?mode=ro", uri=True)
    try:
        for start in range(0, len(wanted), SQLITE_MAX_IN):
            chunk = wanted[start:start + SQLITE_MAX_IN]
            found.update(r[0] for r in db.execute(
                f"SELECT arxiv_id FROM papers WHERE {CONVERT_REJECTED_SQL} "
                f"AND arxiv_id IN ({','.join('?' * len(chunk))})", chunk))
    finally:
        db.close()
    return found


def _read_report(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def convert_failures(paths: SlicePaths) -> dict[str, str]:
    """{arxiv_id: error} of the conversions that failed, from the key's report."""
    return _read_report(paths.report).get("failures", {})


def corpus_paper(row: HarvestRow, markdown_file: Optional[Path], extract_error: Optional[str]) -> CorpusPaper:
    return CorpusPaper(
        paper_id=paper_id_for(row.arxiv_id),
        title=row.title,
        authors=split_authors(row.authors),
        conference=ARXIV_VENUE,
        year=int(row.first_version_date[:4]),
        pdf_url=f"https://arxiv.org/pdf/{row.arxiv_id}",
        paper_page_url=f"https://arxiv.org/abs/{row.arxiv_id}",
        source_name=SOURCE_NAME,
        abstract=row.abstract,
        markdown_path=str(markdown_file) if markdown_file else None,
        extract_error=extract_error,
        arxiv_id=row.arxiv_id,
        venue_tier=PREPRINT,
        track_type=PREPRINT,
        source_version=row.content_version,
        withdrawn=row.withdrawn,
    )


def write_manifest(rows: list[HarvestRow], paths: SlicePaths, failures: dict[str, str],
                   *, log=print) -> dict[str, int]:
    """Write ``paths.manifest`` for ``rows`` from the converted files.

    Every downloaded paper gets a record; ``markdown_path`` is set only when
    the Markdown exists, and a conversion failure is carried as
    ``extract_error`` so preindex skips it (``_eligible`` in preindex). A
    withdrawn paper is listed like any other (its last content version's
    Markdown), flagged ``withdrawn`` so preindex can update the flag."""
    records = []
    with_md = 0
    for row in rows:
        md = paths.markdown(row.arxiv_id)
        has_md = md.exists()
        with_md += has_md
        records.append(corpus_paper(row, md if has_md else None,
                                    None if has_md else failures.get(row.arxiv_id)).to_manifest_record())
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.manifest, records)
    log(f"manifest {slice_label(paths.key)}: {len(records):,} records, {with_md:,} with markdown -> {paths.manifest}")
    return {"records": len(records), "with_markdown": with_md}


def write_slice_manifest(key: int, *, hroot: Path | None = None, croot: Path | None = None,
                         log=print) -> dict[str, int]:
    """Rebuild a month/batch manifest from harvest state + converted files."""
    hroot = hroot or harvest_root()
    paths = SlicePaths((croot or corpus_root()).resolve(), key)
    rows = slice_rows(hroot / "arxiv_cs.sqlite3", key, statuses=["ok"])
    return write_manifest(rows, paths, convert_failures(paths), log=log)


def all_slices(harvest_db: Path) -> list[int]:
    db = sqlite3.connect(f"file:{harvest_db}?mode=ro", uri=True)
    try:
        ids = db.execute(
            "SELECT DISTINCT CASE WHEN arxiv_id LIKE '%/%' THEN '0000' ELSE substr(arxiv_id, 1, 4) END "
            "FROM papers WHERE latest_version_date >= ?",
            (HTML_CUTOFF.isoformat(),),
        ).fetchall()
    finally:
        db.close()
    return sorted({LEGACY_SLICE if p == "0000" else slice_of(f"{p}.00001") for (p,) in ids})


def iter_slice_counts(harvest_db: Path) -> Iterator[tuple[int, dict[str, int]]]:
    for key in all_slices(harvest_db):
        rows = slice_rows(harvest_db, key)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        yield key, counts
