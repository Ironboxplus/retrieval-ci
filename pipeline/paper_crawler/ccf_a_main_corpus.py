"""
Build a clean main-conference-only CCF-A AI corpus from official proceedings.

The default scope is the latest three fully archived proceedings years that are
consistently available from official sources across the target venues:
2022, 2023, and 2024.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import settings as retrieval_settings
from core.db.database import DB_PATH, init_db
from pipeline.download_control import (
    AggregateDownloadRateLimiter,
    ChainedProxyPool,
    ChainedTargetHTTPError,
    DEFAULT_DOWNLOAD_BYTES_PER_SECOND,
    DEFAULT_STREAM_CHUNK_BYTES,
    require_explicit_proxy,
    require_reachable_proxy,
)
from pipeline.paper_crawler.models.paper import Conference, Paper, PaperMetadata, PaperType
from pipeline.paper_crawler.track_classifier import MAIN_TRACKS, classify_track
from pipeline.paper_crawler.runtime_files import (
    build_write_in_progress_path,
    is_valid_markdown_artifact,
    is_valid_pdf_artifact,
    load_json_with_recovery,
    remove_file_if_present,
    write_json_atomic,
    write_text_atomic,
)


LOGGER = logging.getLogger(__name__)

# Manifest keys that are deliberately not `CorpusPaper` fields. Dropping these is
# expected, so they must not produce the unknown-field warning on every single row.
_IGNORED_MANIFEST_KEYS = frozenset({"selection_rule"})

DEFAULT_PROXY_URL = "http://127.0.0.1:10808"
DEFAULT_TARGET_YEARS = (2022, 2023, 2024, 2025)
DEFAULT_TARGET_CONFERENCES = (
    "AAAI",
    "ACL",
    "CVPR",
    "EMNLP",
    "ICCV",
    "ICLR",
    "ICML",
    "IJCAI",
    "KDD",
    "NeurIPS",
    "SIGIR",
    "SIGMOD",
)
DEFAULT_MANIFEST_CHECKPOINT_INTERVAL = 200
DEFAULT_DOWNLOAD_RETRY_ATTEMPTS = 3
DEFAULT_DOWNLOAD_CONCURRENCY = 4
OPENREVIEW_COOKIE_FILE_ENV = "OPENREVIEW_COOKIE_FILE"
OPENREVIEW_PAGE_DELAY_SECONDS_ENV = "OPENREVIEW_PAGE_DELAY_SECONDS"

MAIN_CONFERENCE_RULE = "accepted_main_conference_only"
CORPUS_NAME = "ccf_a_main_conference_corpus"
KNOWN_RETRACTED_PAPERS = {
    "cvpr-2022-kim_e2v-sde_from_asynchronous_events_to_fast_and_continuous_video_reconstruction_cvpr_2022_paper": {
        "notice": "IEEE Xplore lists this CVPR 2022 item as retracted.",
        "notice_url": "https://ieeexplore.ieee.org/document/9880442/",
    }
}

ICML_VOLUME_BY_YEAR = {
    2022: "v162",
    2023: "v202",
    2024: "v235",
    2025: "v267",
}


def build_default_corpus_root() -> Path:
    return Path(retrieval_settings.data_dir) / "papers" / CORPUS_NAME


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_file_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-._" else "_" for character in value)


def extract_inline_text(node: Any) -> str:
    """Preserve inline word boundaries while collapsing layout whitespace.

    ACL Anthology wraps protected capitals such as the ``O`` and ``T`` in
    ``OctoTools`` in separate spans. Supplying a separator to ``get_text``
    invents spaces inside the word; raw descendant text retains the HTML's real
    word spacing, after which layout whitespace can be normalized safely.
    """
    return " ".join(node.get_text("", strip=False).split())


def normalize_link_label(anchor: Any) -> str:
    return anchor.get_text(" ", strip=True).lower()


def resolve_labeled_link(container: Any, base_url: str, expected_label: str) -> Optional[str]:
    normalized_expected_label = expected_label.lower()
    for anchor in container.find_all("a", href=True):
        if normalize_link_label(anchor) == normalized_expected_label:
            return urljoin(base_url, anchor["href"])
    return None


def normalize_aaai_pdf_url(url: str) -> str:
    return url.replace("/article/view/", "/article/download/")


def load_openreview_cookies_from_env() -> Optional[httpx.Cookies]:
    """Load a temporary Chrome-CDP cookie export for OpenReview API challenges.

    The file is intentionally opt-in via env var so normal corpus runs do not
    persist or discover browser cookies. Expected input is the list returned by
    CDP Network.getAllCookies, filtered to OpenReview domains by the caller.
    """
    cookie_file = os.environ.get(OPENREVIEW_COOKIE_FILE_ENV)
    if not cookie_file:
        return None

    path = Path(cookie_file)
    if not path.exists():
        raise RuntimeError(f"{OPENREVIEW_COOKIE_FILE_ENV} points to a missing file: {path}")

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load {OPENREVIEW_COOKIE_FILE_ENV}={path}: {exc}") from exc

    cookies = httpx.Cookies()
    loaded = 0
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        domain = record.get("domain")
        name = record.get("name")
        value = record.get("value")
        if not domain or not name or value is None or "openreview.net" not in domain:
            continue
        cookies.set(name, value, domain=domain, path=record.get("path") or "/")
        loaded += 1

    if loaded == 0:
        raise RuntimeError(f"{OPENREVIEW_COOKIE_FILE_ENV}={path} contained no OpenReview cookies")
    return cookies


def http_retry_delay_seconds(exc: Exception, attempt: int) -> float:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return max(60.0, float(retry_after))
            except ValueError:
                pass
        return 60.0
    return float(min(2 * attempt, 5))


def build_arxiv_pdf_url(url: str) -> Optional[str]:
    normalized_url = url.strip()
    if not normalized_url:
        return None
    if "arxiv.org/abs/" in normalized_url:
        arxiv_identifier = normalized_url.rsplit("/abs/", maxsplit=1)[-1]
        return f"https://arxiv.org/pdf/{arxiv_identifier}.pdf"
    if "arxiv.org/pdf/" in normalized_url:
        return normalized_url if normalized_url.endswith(".pdf") else f"{normalized_url}.pdf"
    return None


def normalize_title_key(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip()).casefold()


@dataclass(slots=True)
class CorpusPaper:
    paper_id: str
    title: str
    authors: list[str]
    conference: str
    year: int
    pdf_url: str
    paper_page_url: str
    source_name: str
    abstract: str = ""
    venue_track: Optional[str] = None
    source_cycle: Optional[str] = None
    alternate_pdf_urls: list[str] = field(default_factory=list)
    downloaded_from_url: Optional[str] = None
    pdf_resolution_status: Optional[str] = None
    pdf_resolution_note: Optional[str] = None
    retraction_notice_url: Optional[str] = None
    pdf_path: Optional[str] = None
    markdown_path: Optional[str] = None
    download_error: Optional[str] = None
    extract_error: Optional[str] = None
    index_error: Optional[str] = None
    indexed_doc_id: Optional[str] = None
    indexed_doc_id: Optional[str] = None

    def to_manifest_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["selection_rule"] = MAIN_CONFERENCE_RULE
        return record

    @classmethod
    def from_manifest_record(cls, record: dict[str, Any]) -> "CorpusPaper":
        """Build a paper from a manifest row, tolerating fields this build lacks.

        The manifest is shared state between hosts that upgrade at different times,
        so a row carrying a field this build does not know is a NORMAL condition.
        Expanding it wholesale made that fatal: on 2026-09-14 the KDD collection
        added ``venue_track`` and `.88`'s older `CorpusPaper` killed the whole SIGIR
        preindex stage with ``TypeError: unexpected keyword argument 'venue_track'``.
        Unknown keys are dropped, but always NAMED in the log — a silently swallowed
        typo would be a worse failure than the crash it replaces.
        """
        known = {f.name for f in fields(cls)}
        filtered_record = {k: v for k, v in record.items() if k in known}
        unknown = set(record) - known - _IGNORED_MANIFEST_KEYS
        if unknown:
            LOGGER.warning(
                "Ignoring unknown manifest field(s) %s on paper %s; this build is "
                "older than the manifest writer",
                ", ".join(sorted(unknown)),
                record.get("paper_id", "<no paper_id>"),
            )
        return cls(**filtered_record)

    def to_runtime_paper(self) -> Paper:
        try:
            conference = Conference(self.conference)
        except ValueError:
            conference = Conference.OTHER

        return Paper(
            id=self.paper_id,
            title=self.title,
            authors=self.authors,
            abstract=self.abstract or None,
            year=self.year,
            conference=conference,
            paper_type=PaperType.UNKNOWN,
            pdf_url=self.downloaded_from_url or self.pdf_url,
            paper_page_url=self.paper_page_url,
            source=self.source_name,
            venue_track=self.venue_track,
            pdf_path=self.pdf_path,
            markdown_path=self.markdown_path,
            metadata=PaperMetadata(
                venue=self.conference,
                year=self.year,
                track="main_conference",
            ),
        )


@dataclass(frozen=True, slots=True)
class ConferencePaperCandidate:
    """Announced paper without a stable proceedings identity or public PDF.

    Candidate snapshots are intentionally separate from ``CorpusPaper`` so a
    title-only conference announcement can never enter the canonical manifest,
    SQL, or Qdrant before the publisher assigns stable paper identifiers.
    """

    candidate_id: str
    title: str
    authors: tuple[str, ...]
    conference: str
    year: int
    source_url: str
    venue_track: str = "main"

    def to_snapshot_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "title": self.title,
            "authors": list(self.authors),
            "conference": self.conference,
            "year": self.year,
            "source_url": self.source_url,
            "venue_track": self.venue_track,
            "status": "announced_waiting_for_proceedings",
        }


class OfficialProceedingsClient:
    def __init__(self, proxy_url: str = DEFAULT_PROXY_URL, timeout_seconds: int = 60):
        self.proxy_url = proxy_url
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        self.client = self._build_client()

    def _build_client(self) -> httpx.Client:
        client_options: dict[str, Any] = {
            "proxy": self.proxy_url,
            "timeout": self.timeout_seconds,
            "follow_redirects": True,
            "headers": self.headers,
        }
        cookies = load_openreview_cookies_from_env()
        if cookies is not None:
            client_options["cookies"] = cookies
        return httpx.Client(**client_options)

    def reset_client(self) -> None:
        self.close()
        self.client = self._build_client()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def get_text(self, url: str) -> str:
        last_error: Optional[Exception] = None
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.get(url)
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_error = exc
                self.reset_client()
                if attempt == max_attempts:
                    break
                LOGGER.warning("Retrying %s after request failure (%s/%s): %s", url, attempt, max_attempts, exc)
                sleep_seconds = min(2 * attempt, 8)
                time.sleep(sleep_seconds)
        raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error

    def get_soup(self, url: str) -> BeautifulSoup:
        return BeautifulSoup(self.get_text(url), "html.parser")

    def post_json(
        self,
        url: str,
        *,
        body: Any,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """POST JSON through the same explicit proxy with bounded retries."""
        last_error: Optional[Exception] = None
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.post(url, params=params, json=body)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                retry_after = None
                status_code = None
                if isinstance(exc, httpx.HTTPStatusError):
                    status_code = exc.response.status_code
                    retry_after = exc.response.headers.get("retry-after")
                self.reset_client()
                if attempt == max_attempts:
                    break
                try:
                    if status_code == 429:
                        delay = max(60.0, float(retry_after or 0))
                    else:
                        delay = float(retry_after) if retry_after else min(2 * attempt, 8)
                except ValueError:
                    delay = 60.0 if status_code == 429 else min(2 * attempt, 8)
                time.sleep(max(0.0, min(delay, 120.0)))
        raise RuntimeError(f"Failed to POST {url}: {last_error}") from last_error

    def fetch_title(self, url: str) -> str:
        soup = self.get_soup(url)
        if soup.title:
            return soup.title.get_text(" ", strip=True)
        return ""


class OfficialProceedingsSource:
    conference_name: str

    def supports_year(self, year: int) -> bool:
        return True

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        raise NotImplementedError


class NeuripsProceedingsSource(OfficialProceedingsSource):
    conference_name = "NeurIPS"

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        list_url = f"https://proceedings.neurips.cc/paper/{year}"
        soup = client.get_soup(list_url)
        papers: dict[str, CorpusPaper] = {}

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if "-Abstract-Conference.html" not in href:
                continue

            title = anchor.get_text(" ", strip=True)
            if not title:
                continue

            hash_match = re.search(
                rf"/paper_files/paper/{year}/hash/([0-9a-f]+)-Abstract-Conference\.html",
                href,
            )
            if hash_match is None:
                continue

            paper_hash = hash_match.group(1)
            paper_page_url = urljoin(list_url, href)
            pdf_url = f"https://proceedings.neurips.cc/paper_files/paper/{year}/file/{paper_hash}-Paper-Conference.pdf"
            paper_id = f"neurips-{year}-{paper_hash}"
            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=[],
                conference=self.conference_name,
                year=year,
                pdf_url=pdf_url,
                paper_page_url=paper_page_url,
                source_name="neurips_proceedings",
            )

        return list(papers.values())


class IcmlProceedingsSource(OfficialProceedingsSource):
    conference_name = "ICML"
    OPENREVIEW_API_V2 = "https://api2.openreview.net"
    PAGE_SIZE = 1000
    PAGE_DELAY_SECONDS = 10.0
    VIRTUAL_SCHEDULE_URL_BY_YEAR = {
        2026: "https://icml.cc/Conferences/2026/Schedule",
    }
    VIRTUAL_PAPERS_URL_BY_YEAR = {
        2026: "https://icml.cc/virtual/2026/papers.html",
    }
    OPENREVIEW_YEAR_CONFIG: dict[int, dict[str, str]] = {
        2026: {
            "invitation": "ICML.cc/2026/Conference/-/Submission",
            "venueid": "ICML.cc/2026/Conference",
        }
    }

    def supports_year(self, year: int) -> bool:
        return (
            year in ICML_VOLUME_BY_YEAR
            or year in self.VIRTUAL_SCHEDULE_URL_BY_YEAR
            or year in self.OPENREVIEW_YEAR_CONFIG
        )

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        if year in self.VIRTUAL_SCHEDULE_URL_BY_YEAR:
            return self._collect_virtual_schedule_papers(year, client)
        if year in self.OPENREVIEW_YEAR_CONFIG:
            return self._collect_openreview_papers(year, client)
        return self._collect_pmlr_papers(year, client)

    def _collect_pmlr_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        volume_name = ICML_VOLUME_BY_YEAR[year]
        list_url = f"https://proceedings.mlr.press/{volume_name}/"
        soup = client.get_soup(list_url)
        papers: dict[str, CorpusPaper] = {}

        for title_node in soup.find_all("p", class_="title"):
            title = title_node.get_text(" ", strip=True)
            container = title_node.parent
            if not title or container is None:
                continue

            page_url = None
            pdf_url = None
            for anchor in container.find_all("a", href=True):
                href = anchor["href"]
                if href.startswith(f"https://proceedings.mlr.press/{volume_name}/") and href.endswith(".html"):
                    page_url = href
                if href.endswith(".pdf"):
                    pdf_url = href

            if page_url is None or pdf_url is None:
                continue

            slug = Path(page_url).stem
            authors_text = ""
            authors_node = container.find("span", class_="authors")
            if authors_node is not None:
                authors_text = authors_node.get_text(" ", strip=True)
            authors = [name.strip() for name in authors_text.split(",") if name.strip()]

            paper_id = f"icml-{year}-{slug}"
            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                pdf_url=pdf_url,
                paper_page_url=page_url,
                source_name="pmlr",
            )

        return list(papers.values())

    @staticmethod
    def _openreview_forum_id(url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc != "openreview.net" or parsed.path != "/forum":
            return ""
        values = parse_qs(parsed.query).get("id", [])
        return values[0].strip() if values else ""

    @staticmethod
    def _maincard_parent(anchor: Any) -> Optional[Any]:
        for parent in anchor.parents:
            classes = parent.get("class", []) if hasattr(parent, "get") else []
            if "maincard" in classes:
                return parent
        return None

    @staticmethod
    def _maincard_event_id(card: Any) -> str:
        for parent in card.parents:
            if hasattr(parent, "get"):
                event_id = parent.get("data-event-id")
                if event_id:
                    return str(event_id).strip()
        return ""

    @staticmethod
    def _virtual_papers_event_id(url: str) -> str:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[-2] == "poster" and parts[-1].isdigit():
            return parts[-1]
        return ""

    def _collect_virtual_papers_event_ids(self, year: int, client: OfficialProceedingsClient) -> set[str]:
        papers_url = self.VIRTUAL_PAPERS_URL_BY_YEAR[year]
        soup = client.get_soup(papers_url)
        event_ids = {
            event_id
            for anchor in soup.find_all("a", href=True)
            if (event_id := self._virtual_papers_event_id(anchor["href"]))
        }
        if not event_ids:
            raise RuntimeError(
                f"ICML {year} papers page yielded no poster event ids; "
                "refusing to fall back to the all-events schedule page."
            )
        return event_ids

    def _collect_virtual_schedule_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        schedule_url = self.VIRTUAL_SCHEDULE_URL_BY_YEAR[year]
        soup = client.get_soup(schedule_url)
        allowed_event_ids = self._collect_virtual_papers_event_ids(year, client)
        papers: dict[str, CorpusPaper] = {}

        for anchor in soup.find_all("a", href=True):
            forum_id = self._openreview_forum_id(anchor["href"])
            if not forum_id:
                continue

            card = self._maincard_parent(anchor)
            if card is None:
                continue
            event_id = self._maincard_event_id(card)
            if event_id not in allowed_event_ids:
                continue

            title_node = card.find("div", class_="maincardBody")
            title = title_node.get_text(" ", strip=True) if title_node is not None else ""
            if not title:
                continue

            authors_node = card.find("div", class_="maincardFooter")
            authors_text = authors_node.get_text(" ", strip=True) if authors_node is not None else ""
            authors = [name.strip() for name in authors_text.split("\u22c5") if name.strip()]

            paper_id = f"icml-{year}-{forum_id}"
            if paper_id in papers:
                continue

            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                pdf_url=f"https://openreview.net/pdf?id={forum_id}",
                paper_page_url=f"https://openreview.net/forum?id={forum_id}",
                source_name="icml_virtual_schedule",
            )

        return list(papers.values())

    def _paginate_openreview_notes(self, config: dict[str, str], client: OfficialProceedingsClient) -> list[dict]:
        all_notes: list[dict] = []
        offset = 0
        while True:
            page_params = {
                "invitation": config["invitation"],
                "content.venueid": config["venueid"],
                "limit": self.PAGE_SIZE,
                "offset": offset,
            }
            url = f"{self.OPENREVIEW_API_V2}/notes?{urlencode(page_params)}"
            data = json.loads(client.get_text(url))
            if data.get("name") == "ChallengeRequiredError":
                raise RuntimeError(
                    "OpenReview challenge required. Complete it in a CDP browser, export cookies with "
                    f"Network.getAllCookies, and set {OPENREVIEW_COOKIE_FILE_ENV} to that temporary JSON file."
                )

            notes = data.get("notes", [])
            all_notes.extend(notes)
            total = data.get("count", len(all_notes))
            offset += len(notes)
            LOGGER.info("ICML OpenReview: fetched %s/%s notes", offset, total)
            if not notes or offset >= total:
                break
            delay_seconds = float(os.environ.get(OPENREVIEW_PAGE_DELAY_SECONDS_ENV, self.PAGE_DELAY_SECONDS))
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        return all_notes

    @staticmethod
    def _extract_openreview_field(content: dict, key: str, default: Any = "") -> Any:
        field = content.get(key, default)
        if isinstance(field, dict):
            return field.get("value", default)
        return field if field is not None else default

    @classmethod
    def _openreview_pdf_url(cls, note: dict, content: dict) -> str:
        pdf_field = cls._extract_openreview_field(content, "pdf")
        if isinstance(pdf_field, str) and pdf_field.strip():
            pdf_value = pdf_field.strip()
            if pdf_value.startswith("http://") or pdf_value.startswith("https://"):
                return pdf_value
            if pdf_value.startswith("/"):
                return f"https://openreview.net{pdf_value}"
        return f"https://openreview.net/pdf?id={note.get('id', '')}"

    def _collect_openreview_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        config = self.OPENREVIEW_YEAR_CONFIG[year]
        notes = self._paginate_openreview_notes(config, client)

        papers: dict[str, CorpusPaper] = {}
        for note in notes:
            forum_id = note.get("id", "")
            content = note.get("content", {})

            title = self._extract_openreview_field(content, "title")
            authors = self._extract_openreview_field(content, "authors", [])
            abstract = self._extract_openreview_field(content, "abstract")

            if not forum_id or not title:
                continue

            paper_id = f"icml-{year}-{forum_id}"
            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors if isinstance(authors, list) else [],
                conference=self.conference_name,
                year=year,
                pdf_url=self._openreview_pdf_url(note, content),
                paper_page_url=f"https://openreview.net/forum?id={forum_id}",
                source_name="openreview",
                abstract=abstract,
            )

        return list(papers.values())


class CvfProceedingsSource(OfficialProceedingsSource):
    def __init__(self, conference_name: str):
        self.conference_name = conference_name

    def supports_year(self, year: int) -> bool:
        if self.conference_name == "ICCV":
            return year % 2 == 1
        return True

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        list_url = f"https://openaccess.thecvf.com/{self.conference_name}{year}?day=all"
        soup = client.get_soup(list_url)
        papers: dict[str, CorpusPaper] = {}

        for title_node in soup.find_all("dt", class_="ptitle"):
            title = title_node.get_text(" ", strip=True)
            if not title:
                continue

            title_anchor = title_node.find("a", href=True)
            authors_node = title_node.find_next_sibling("dd")
            links_node = authors_node.find_next_sibling("dd") if authors_node else None
            if title_anchor is None or links_node is None:
                continue

            detail_url = urljoin(list_url, title_anchor["href"])
            pdf_url = resolve_labeled_link(links_node, list_url, "pdf")
            alternate_pdf_urls: list[str] = []

            if pdf_url is None or detail_url is None:
                continue

            for anchor in links_node.find_all("a", href=True):
                if normalize_link_label(anchor) != "arxiv":
                    continue
                arxiv_pdf_url = build_arxiv_pdf_url(urljoin(list_url, anchor["href"]))
                if arxiv_pdf_url and arxiv_pdf_url not in alternate_pdf_urls:
                    alternate_pdf_urls.append(arxiv_pdf_url)

            file_stem = Path(detail_url).stem
            authors_text = authors_node.get_text(" ", strip=True) if authors_node else ""
            authors = [name.strip() for name in authors_text.split(",") if name.strip()]
            paper_id = f"{self.conference_name.lower()}-{year}-{sanitize_file_component(file_stem.lower())}"

            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                pdf_url=pdf_url,
                alternate_pdf_urls=alternate_pdf_urls,
                paper_page_url=detail_url,
                source_name="cvf_open_access",
            )

        return list(papers.values())


class AclAnthologySource(OfficialProceedingsSource):
    """Base for any conference hosted on aclanthology.org."""

    conference_name: str
    MAIN_VOLUME_SUFFIXES: tuple[str, ...] = ("main",)
    _VENUE_SLUG: str = ""  # e.g. "acl", "emnlp"

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        papers: dict[str, CorpusPaper] = {}

        for suffix in self.MAIN_VOLUME_SUFFIXES:
            volume_url = f"https://aclanthology.org/volumes/{year}.{self._VENUE_SLUG}-{suffix}/"
            soup = client.get_soup(volume_url)
            for entry_container in soup.select("div.d-sm-flex.align-items-stretch.mb-3"):
                title_anchor = entry_container.select_one("span.d-block strong a.align-middle[href]")
                if title_anchor is None:
                    continue

                href = title_anchor["href"]
                match = re.fullmatch(rf"/{year}\.{self._VENUE_SLUG}-{suffix}\.(\d+)/", href)
                if match is None:
                    continue

                article_number = int(match.group(1))
                title = extract_inline_text(title_anchor)
                if article_number == 0 or not title:
                    continue

                pdf_anchor = entry_container.find("a", href=True, attrs={"aria-label": "Open PDF"})
                if pdf_anchor is None:
                    continue

                pdf_url = urljoin(volume_url, pdf_anchor["href"])
                expected_pdf_url = f"https://aclanthology.org/{year}.{self._VENUE_SLUG}-{suffix}.{article_number}.pdf"
                if pdf_url != expected_pdf_url:
                    continue

                metadata_span = title_anchor.find_parent("span")
                author_anchors = metadata_span.find_all("a", href=True) if metadata_span is not None else []
                authors = [
                    author_anchor.get_text(" ", strip=True)
                    for author_anchor in author_anchors
                    if author_anchor is not title_anchor and author_anchor.get_text(" ", strip=True)
                ]

                paper_id = f"{self._VENUE_SLUG}-{year}-{suffix}-{article_number}"
                papers[paper_id] = CorpusPaper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    conference=self.conference_name,
                    year=year,
                    pdf_url=pdf_url,
                    paper_page_url=urljoin(volume_url, href),
                    source_name="acl_anthology",
                )

        return list(papers.values())


class AclMainConferenceSource(AclAnthologySource):
    conference_name = "ACL"
    MAIN_VOLUME_SUFFIXES = ("long", "short")
    _VENUE_SLUG = "acl"


class EmnlpMainConferenceSource(AclAnthologySource):
    conference_name = "EMNLP"
    MAIN_VOLUME_SUFFIXES = ("main",)
    _VENUE_SLUG = "emnlp"
    ANNOUNCED_PAGE_BY_YEAR = {
        2026: "https://2026.emnlp.org/program/main_papers/",
    }

    def collect_announced_papers(
        self,
        year: int,
        client: OfficialProceedingsClient,
    ) -> list[ConferencePaperCandidate]:
        """Collect the official accepted-main announcement without publishing it.

        EMNLP exposes titles and authors before ACL Anthology assigns canonical
        ``YYYY.emnlp-main.N`` identifiers.  Returning a candidate type keeps
        that provisional list out of the paper manifest and both production
        stores; a later normal ``collect_papers`` run remains the only promotion
        path once the Anthology volume exists.
        """
        source_url = self.ANNOUNCED_PAGE_BY_YEAR.get(year)
        if source_url is None:
            raise ValueError(f"No announced-paper source configured for EMNLP {year}")

        soup = client.get_soup(source_url)
        content = soup.select_one("section.page__content")
        if content is None:
            raise RuntimeError(f"EMNLP {year} accepted page has no page__content section")

        candidates: dict[str, ConferencePaperCandidate] = {}
        for item in content.select("li"):
            title_node = item.find("strong")
            authors_node = item.find("em")
            if title_node is None or authors_node is None:
                continue
            title = extract_inline_text(title_node)
            authors = tuple(
                author.strip()
                for author in extract_inline_text(authors_node).split(",")
                if author.strip()
            )
            if not title or not authors:
                continue
            identity = f"{normalize_title_key(title)}\0{'|'.join(author.casefold() for author in authors)}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            candidate_id = f"emnlp-{year}-announced-{digest}"
            candidates[candidate_id] = ConferencePaperCandidate(
                candidate_id=candidate_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                source_url=source_url,
            )

        if not candidates:
            raise RuntimeError(f"EMNLP {year} accepted page yielded no main papers")
        return list(candidates.values())


class IjcaiProceedingsSource(OfficialProceedingsSource):
    conference_name = "IJCAI"

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        list_url = f"https://www.ijcai.org/proceedings/{year}"
        soup = client.get_soup(list_url)
        papers: dict[str, CorpusPaper] = {}

        for wrapper in soup.find_all("div", class_="paper_wrapper"):
            title_node = wrapper.find("div", class_="title")
            details_node = wrapper.find("div", class_="details")
            if title_node is None or details_node is None:
                continue

            title = title_node.get_text(" ", strip=True)
            authors_node = wrapper.find("div", class_="authors")
            authors_text = authors_node.get_text(" ", strip=True) if authors_node else ""
            authors = [name.strip() for name in authors_text.split(",") if name.strip()]

            pdf_url = None
            detail_url = None
            for anchor in details_node.find_all("a", href=True):
                href = anchor["href"]
                label = anchor.get_text(" ", strip=True).lower()
                if label == "pdf":
                    pdf_url = urljoin(f"{list_url}/", href)
                if "detail" in label:
                    detail_url = urljoin(f"{list_url}/", href)

            if pdf_url is None or detail_url is None:
                continue

            detail_id_match = re.search(rf"/proceedings/{year}/(\d+)", detail_url)
            if detail_id_match is None:
                continue

            detail_id = int(detail_id_match.group(1))
            paper_id = f"ijcai-{year}-{detail_id:04d}"
            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                pdf_url=pdf_url,
                paper_page_url=detail_url,
                source_name="ijcai_proceedings",
            )

        return list(papers.values())


class AaaiTechnicalTracksSource(OfficialProceedingsSource):
    conference_name = "AAAI"
    archive_url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
    issue_scan_windows = {
        2022: range(500, 526),
        2023: range(545, 566),
        2024: range(575, 596),
        2025: range(624, 653),
        2026: range(683, 734),
    }

    def _discover_latest_issue_id(self, client: OfficialProceedingsClient) -> int:
        archive_html = client.get_text(self.archive_url)
        issue_ids = [
            int(match)
            for match in re.findall(r"/index\.php/AAAI/issue/view/(\d+)", archive_html)
        ]
        if not issue_ids:
            raise RuntimeError("Could not discover any AAAI issue ids from the archive page")
        return max(issue_ids)

    def _discover_track_issue_ids(
        self,
        year: int,
        client: OfficialProceedingsClient,
    ) -> list[int]:
        year_marker = f"AAAI-{str(year)[-2:]} Technical Tracks"
        matched_issue_ids: list[int] = []
        issue_candidates = self.issue_scan_windows.get(year)
        if issue_candidates is None:
            latest_issue_id = self._discover_latest_issue_id(client)
            issue_candidates = range(latest_issue_id, 480, -1)

        for issue_id in issue_candidates:
            issue_title = client.fetch_title(f"https://ojs.aaai.org/index.php/AAAI/issue/view/{issue_id}")
            if year_marker in issue_title:
                matched_issue_ids.append(issue_id)

        return sorted(matched_issue_ids)

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        papers: dict[str, CorpusPaper] = {}

        for issue_id in self._discover_track_issue_ids(year, client):
            issue_url = f"https://ojs.aaai.org/index.php/AAAI/issue/view/{issue_id}"
            soup = client.get_soup(issue_url)

            for summary in soup.select(".obj_article_summary"):
                title_link = None
                pdf_link = None
                for anchor in summary.find_all("a", href=True):
                    href = anchor["href"]
                    label = anchor.get_text(" ", strip=True)
                    if "/index.php/AAAI/article/view/" not in href:
                        continue
                    if label == "PDF":
                        pdf_link = href
                        continue
                    if label == "Video/Poster":
                        continue
                    if title_link is None:
                        title_link = anchor

                if title_link is None or pdf_link is None:
                    continue

                article_view_href = title_link["href"]
                article_id_match = re.search(r"/index\.php/AAAI/article/view/(\d+)", article_view_href)
                if article_id_match is None:
                    continue

                article_id = article_id_match.group(1)
                title = title_link.get_text(" ", strip=True)
                author_node = summary.select_one(".authors")
                authors_text = author_node.get_text(" ", strip=True) if author_node else ""
                authors = [name.strip() for name in authors_text.split(",") if name.strip()]

                paper_id = f"aaai-{year}-{article_id}"
                papers[paper_id] = CorpusPaper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    conference=self.conference_name,
                    year=year,
                    pdf_url=normalize_aaai_pdf_url(urljoin(issue_url, pdf_link)),
                    paper_page_url=urljoin(issue_url, article_view_href),
                    source_name="aaai_ojs",
                )

        return list(papers.values())


class IclrOpenReviewSource(OfficialProceedingsSource):
    """ICLR papers via OpenReview HTTP API (v1 for <=2023, v2 for >=2024)."""

    conference_name = "ICLR"
    OPENREVIEW_API_V1 = "https://api.openreview.net"
    OPENREVIEW_API_V2 = "https://api2.openreview.net"
    PAGE_SIZE = 1000

    YEAR_CONFIG: dict[int, dict[str, Any]] = {
        2022: {
            "api": "v1",
            "invitation": "ICLR.cc/2022/Conference/-/Blind_Submission",
            "accepted_venues": [
                "ICLR 2022 Oral",
                "ICLR 2022 Spotlight",
                "ICLR 2022 Poster",
            ],
        },
        2023: {
            "api": "v1",
            "invitation": "ICLR.cc/2023/Conference/-/Blind_Submission",
            "accepted_venues": [
                "ICLR 2023 poster",
                "ICLR 2023 notable top 5%",
                "ICLR 2023 notable top 25%",
            ],
        },
        2024: {
            "api": "v2",
            "invitation": "ICLR.cc/2024/Conference/-/Submission",
            "venueid": "ICLR.cc/2024/Conference",
        },
        2025: {
            "api": "v2",
            "invitation": "ICLR.cc/2025/Conference/-/Submission",
            "venueid": "ICLR.cc/2025/Conference",
        },
        2026: {
            "api": "v2",
            "invitation": "ICLR.cc/2026/Conference/-/Submission",
            "venueid": "ICLR.cc/2026/Conference",
        },
    }

    def supports_year(self, year: int) -> bool:
        return year in self.YEAR_CONFIG

    def _paginate_notes(self, base_url: str, params: dict, client: OfficialProceedingsClient) -> list[dict]:
        all_notes: list[dict] = []
        offset = 0
        while True:
            page_params = {**params, "limit": self.PAGE_SIZE, "offset": offset}
            url = f"{base_url}/notes?{urlencode(page_params)}"
            data = json.loads(client.get_text(url))
            notes = data.get("notes", [])
            all_notes.extend(notes)
            total = data.get("count", len(all_notes))
            offset += len(notes)
            LOGGER.info("ICLR: fetched %s/%s notes", offset, total)
            if not notes or offset >= total:
                break
        return all_notes

    def _fetch_v1(self, year: int, config: dict, client: OfficialProceedingsClient) -> list[dict]:
        all_notes: list[dict] = []
        for venue in config["accepted_venues"]:
            notes = self._paginate_notes(
                self.OPENREVIEW_API_V1,
                {"invitation": config["invitation"], "content.venue": venue},
                client,
            )
            all_notes.extend(notes)
            LOGGER.info("ICLR %s venue '%s': %s papers", year, venue, len(notes))
        return all_notes

    def _fetch_v2(self, config: dict, client: OfficialProceedingsClient) -> list[dict]:
        return self._paginate_notes(
            self.OPENREVIEW_API_V2,
            {"invitation": config["invitation"], "content.venueid": config["venueid"]},
            client,
        )

    @staticmethod
    def _extract_field(content: dict, key: str, default: Any = "") -> Any:
        field = content.get(key, {})
        if isinstance(field, dict):
            return field.get("value", default)
        return field if field is not None else default

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        config = self.YEAR_CONFIG[year]
        if config["api"] == "v1":
            notes = self._fetch_v1(year, config, client)
        else:
            notes = self._fetch_v2(config, client)

        papers: dict[str, CorpusPaper] = {}
        for note in notes:
            forum_id = note.get("id", "")
            content = note.get("content", {})

            title = self._extract_field(content, "title")
            authors = self._extract_field(content, "authors", [])
            abstract = self._extract_field(content, "abstract")

            if not forum_id or not title:
                continue

            paper_id = f"iclr-{year}-{forum_id}"
            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors if isinstance(authors, list) else [],
                conference=self.conference_name,
                year=year,
                pdf_url=f"https://openreview.net/pdf?id={forum_id}",
                paper_page_url=f"https://openreview.net/forum?id={forum_id}",
                source_name="openreview",
                abstract=abstract,
            )

        return list(papers.values())


# W112: full conference papers at the main_only venues (PPoPP/ISCA) are >=6 pages;
# posters / extended abstracts are 2-3 pages. <=4 is a safe cut (clean data gap at
# 5pp). Used only to down-classify an otherwise-main entry on a flat/unsectioned TOC.
_POSTER_MAX_PAGES = 4


def _dblp_toc_entry_page_count(entry) -> Optional[int]:
    """Printed page count for a DBLP TOC <li> entry, or None if unparseable.

    DBLP marks pagination as ``<span itemprop="pagination">1-16</span>``. Returns
    the inclusive span length (16 → ``1-16``); single-page or malformed spans → None.
    """
    span = entry.select_one('span[itemprop="pagination"]')
    if span is None:
        return None
    match = re.match(r"\s*(\d+)\s*-\s*(\d+)", span.get_text(strip=True))
    if match is None:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return end - start + 1 if end >= start else None


def parse_dblp_toc_tracks(html: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Parse a DBLP HTML table-of-contents into per-paper track labels.

    DBLP's JSON TOC API is section-blind, so workshop/tutorial/demo/poster/etc.
    entries that share the main proceedings' DOI prefix are indistinguishable there.
    The HTML TOC groups entries under section headers, which ``classify_track`` maps
    to a track label (research/short/industry = main; workshop/demo/poster/... = not).

    Returns ``(doi_to_track, manifest)``: ``doi_to_track`` maps a lower-cased DOI
    (e.g. ``10.1145/3626772.3661366``) to its ``track_type``; ``manifest`` maps each
    section heading to a ``{track_type: count}`` tally for audit/logging. Entries
    without a DOI link are skipped (they cannot be matched back to an ingested paper).
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("div", id="main") or soup
    doi_to_track: dict[str, str] = {}
    manifest: dict[str, dict[str, int]] = {}
    current = "(no section)"
    for el in main.find_all(["h2", "h3", "li"]):
        if el.name in ("h2", "h3"):
            current = el.get_text(" ", strip=True).replace("\n", " ").strip() or "(no section)"
            continue
        classes = el.get("class") or []
        if "entry" not in classes or "inproceedings" not in classes:
            continue
        title_node = el.select_one(".title")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        track = classify_track(current, title).track_type
        # W112 page-count guard: a main-classified entry of <=4 printed pages is
        # structurally a poster / extended abstract, not a full paper (full papers
        # at these venues are >=6pp; the data shows a clean gap with nothing at 5pp).
        # This catches poster contamination when the DBLP TOC is FLAT/unsectioned
        # (e.g. PPoPP 2022, whose entire TOC defaults to 'research' — section-blind
        # classify_track cannot see the posters). Only DOWN-classifies main rows;
        # never up-classifies an already non-main entry. Section-labelled posters
        # are unaffected (already non-main). Honors the user gate "无法分清就不下".
        if track in MAIN_TRACKS:
            pages = _dblp_toc_entry_page_count(el)
            if pages is not None and pages <= _POSTER_MAX_PAGES:
                track = "poster"
        manifest.setdefault(current, {})
        manifest[current][track] = manifest[current].get(track, 0) + 1
        link = el.select_one('a[href*="doi.org/"]')
        if link:
            m = re.search(r"doi\.org/(10\.\d{3,9}/[^\s\"'<>]+)", link.get("href", ""))
            if m:
                doi_to_track[m.group(1).lower()] = track
    return doi_to_track, manifest


def _acm_doi_of(paper: "CorpusPaper") -> str:
    """Lower-cased ACM DOI from a collected paper's ``pdf_url``
    (``https://dl.acm.org/doi/pdf/10.1145/x.y`` -> ``10.1145/x.y``)."""
    return (paper.pdf_url or "").replace("https://dl.acm.org/doi/pdf/", "").lower()


def filter_main_track_papers(
    papers: list["CorpusPaper"],
    doi_to_track: dict[str, str],
) -> tuple[list["CorpusPaper"], dict[str, int]]:
    """Strict main-only gate for section-blind ACM collection (architecture venues).

    Keep a paper IFF its DOI is present in the DBLP-HTML-TOC track map AND maps to a
    MAIN track (research/short/industry). Everything else is dropped:

    * non-main tracks (workshop/demo/poster/tutorial/keynote/...), and
    * DOIs ABSENT from the HTML TOC (unclassifiable) — "can't confirm it's main ->
      don't download", the user's gate for these venues.

    This is the OPPOSITE default from ``classify_track`` (which keeps-by-default as
    research): here we keep ONLY confirmed-main, because for a section-blind whole-
    TOC pull a missing classification means we never verified the entry is a main-
    conference research paper. Returns ``(kept, dropped_tally)`` where the tally maps
    a drop reason (the non-main track label, or ``"unclassified"``) to a count.
    """
    kept: list[CorpusPaper] = []
    dropped: dict[str, int] = {}
    for paper in papers:
        track = doi_to_track.get(_acm_doi_of(paper))
        if track in MAIN_TRACKS:
            kept.append(paper)
        else:
            reason = track or "unclassified"
            dropped[reason] = dropped.get(reason, 0) + 1
    return kept, dropped


class DblpAcmProceedingsSource(OfficialProceedingsSource):
    """ACM conference papers via DBLP TOC API. Works for KDD, SIGMOD, SIGIR, etc.

    Supports year-specific overrides for venues that changed DBLP layout (e.g. SIGMOD → PACMMOD).
    Pass ``dblp_toc_overrides={2023: "journals/pacmmod/pacmmod1", ...}`` for per-year keys.
    """

    DBLP_BATCH_SIZE = 1000
    PROCEEDINGS_DOI_PREFIX = "10.1145/"

    def __init__(
        self,
        conference_name: str,
        dblp_venue_key: str,
        dblp_toc_overrides: Optional[dict[int, str]] = None,
    ):
        self.conference_name = conference_name
        self.dblp_venue_key = dblp_venue_key
        self.dblp_toc_overrides = dblp_toc_overrides or {}

    def _toc_keys_for_year(self, year: int) -> list[str]:
        """Return DBLP TOC key(s) for a given year. Supports str or list overrides."""
        override = self.dblp_toc_overrides.get(year)
        if override:
            return override if isinstance(override, list) else [override]
        return [f"conf/{self.dblp_venue_key}/{self.dblp_venue_key}{year}"]

    def _dblp_toc_url(self, toc_key: str, offset: int = 0) -> str:
        q = f"toc:db/{toc_key}.bht:"
        return (
            f"https://dblp.org/search/publ/api?"
            f"q={q}&h={self.DBLP_BATCH_SIZE}&f={offset}&format=json"
        )

    @staticmethod
    def _parse_authors(authors_field: Any) -> list[str]:
        if authors_field is None:
            return []
        raw = authors_field.get("author", [])
        if isinstance(raw, dict):
            raw = [raw]
        return [a["text"].split(" 0")[0] for a in raw if a.get("text")]

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        papers: dict[str, CorpusPaper] = {}

        for toc_key in self._toc_keys_for_year(year):
            offset = 0
            while True:
                url = self._dblp_toc_url(toc_key, offset)
                data = json.loads(client.get_text(url))
                hits_obj = data.get("result", {}).get("hits", {})
                total = int(hits_obj.get("@total", 0))
                hit_list = hits_obj.get("hit", [])

                for hit in hit_list:
                    info = hit.get("info", {})
                    ee = info.get("ee", "")
                    if isinstance(ee, list):
                        ee = ee[0]
                    if not ee or self.PROCEEDINGS_DOI_PREFIX not in str(ee):
                        continue

                    doi = str(ee).replace("https://doi.org/", "")
                    title = info.get("title", "").rstrip(".")
                    if not title:
                        continue

                    authors = self._parse_authors(info.get("authors"))
                    doi_suffix = doi.replace(self.PROCEEDINGS_DOI_PREFIX, "")
                    paper_id = f"{self.dblp_venue_key}-{year}-{doi_suffix}"

                    papers[paper_id] = CorpusPaper(
                        paper_id=paper_id,
                        title=title,
                        authors=authors,
                        conference=self.conference_name,
                        year=year,
                        pdf_url=f"https://dl.acm.org/doi/pdf/{doi}",
                        paper_page_url=str(ee),
                        source_name="dblp",
                    )

                offset += len(hit_list)
                if not hit_list or offset >= total:
                    break

        return list(papers.values())

    def collect_track_map(self, year: int, client: OfficialProceedingsClient) -> dict[str, str]:
        """DOI -> track for a year, from the DBLP **HTML** TOC(s).

        ``collect_papers`` uses the section-blind JSON API; this fetches the HTML
        TOC for the same key(s), which DOES expose section headings, so the result
        distinguishes main-conference research papers from the workshop/demo/poster/
        tutorial entries sharing the ``10.1145`` DOI prefix. Feed it to
        :func:`filter_main_track_papers` to keep only main-track papers."""
        doi_to_track: dict[str, str] = {}
        for toc_key in self._toc_keys_for_year(year):
            url = f"https://dblp.org/db/{toc_key}.html"
            try:
                html = client.get_text(url)
            except Exception as exc:  # noqa: BLE001 — one bad TOC must not lose the year
                LOGGER.warning("Cannot fetch HTML TOC %s: %s", url, exc)
                continue
            dt, _ = parse_dblp_toc_tracks(html)
            doi_to_track.update(dt)
        return doi_to_track


class KddOfficialProceedingsSource(DblpAcmProceedingsSource):
    """KDD via its official program for 2026, with DBLP for older editions.

    The live 2026 program embeds two JSON arrays, one per submission cycle.
    DBLP can lag a current conference and its JSON TOC is section-blind, while
    the official rows already carry an explicit archival track and DOI.  Every
    allowed track below is a published main-conference track; unknown codes
    fail closed instead of being mislabeled as research papers.
    """

    OFFICIAL_PAGE_BY_YEAR = {
        2026: "https://kdd2026.kdd.org/papers/",
    }
    OFFICIAL_ARRAYS = (
        ("july", "cycle1Papers"),
        ("february", "cycle2Papers"),
    )
    OFFICIAL_TRACKS = {
        "rtp": "research",
        "ads": "applied_data_science",
        "dtb": "datasets_and_benchmarks",
        "ais": "ai_for_sciences",
        "bsi": "blue_sky_ideas",
    }

    def __init__(
        self,
        conference_name: str = "KDD",
        dblp_venue_key: str = "kdd",
        dblp_toc_overrides: Optional[dict[int, str]] = None,
    ) -> None:
        super().__init__(conference_name, dblp_venue_key, dblp_toc_overrides)

    @staticmethod
    def _decode_embedded_array(page_text: str, variable_name: str) -> list[dict[str, Any]]:
        assignment = re.search(rf"\bconst\s+{re.escape(variable_name)}\s*=", page_text)
        if assignment is None:
            raise RuntimeError(f"KDD official page is missing {variable_name}")
        payload = page_text[assignment.end():].lstrip()
        value, _ = json.JSONDecoder().raw_decode(payload)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise RuntimeError(f"KDD official page {variable_name} is not a paper array")
        return value

    @staticmethod
    def _parse_official_authors(value: Any) -> list[str]:
        names: list[str] = []
        for item in str(value or "").split(";"):
            name = item.strip().split(" (", 1)[0].strip()
            if name:
                names.append(name)
        return names

    def _collect_official_2026(
        self,
        year: int,
        client: OfficialProceedingsClient,
    ) -> list[CorpusPaper]:
        page_url = self.OFFICIAL_PAGE_BY_YEAR[year]
        page_text = client.get_text(page_url)
        papers: dict[str, CorpusPaper] = {}
        raw_rows = 0
        duplicate_rows = 0

        for cycle, variable_name in self.OFFICIAL_ARRAYS:
            for row in self._decode_embedded_array(page_text, variable_name):
                raw_rows += 1
                track_code = str(row.get("track") or "").strip().lower()
                if track_code not in self.OFFICIAL_TRACKS:
                    raise RuntimeError(
                        f"KDD {year} official page contains unknown track {track_code!r}"
                    )
                title = " ".join(str(row.get("title") or "").split())
                authors = self._parse_official_authors(row.get("authors"))
                doi_url = str(row.get("url") or "").strip()
                parsed = urlparse(doi_url)
                doi = parsed.path.lstrip("/")
                if (
                    not title
                    or not authors
                    or parsed.scheme != "https"
                    or parsed.hostname != "doi.org"
                    or not doi.startswith(self.PROCEEDINGS_DOI_PREFIX)
                ):
                    raise RuntimeError(
                        f"KDD {year} official row lacks title/authors/ACM DOI: {row!r}"
                    )

                doi_suffix = doi.removeprefix(self.PROCEEDINGS_DOI_PREFIX)
                paper_id = f"{self.dblp_venue_key}-{year}-{doi_suffix}"
                paper = CorpusPaper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    conference=self.conference_name,
                    year=year,
                    pdf_url=f"https://dl.acm.org/doi/pdf/{doi}",
                    paper_page_url=f"https://doi.org/{doi}",
                    source_name="kdd_official_program",
                    venue_track=self.OFFICIAL_TRACKS[track_code],
                    source_cycle=cycle,
                )
                previous = papers.get(paper_id)
                if previous is not None and previous != paper:
                    raise RuntimeError(f"KDD {year} conflicting duplicate DOI for {paper_id}")
                if previous is not None:
                    duplicate_rows += 1
                papers[paper_id] = paper

        if not papers:
            raise RuntimeError(f"KDD {year} official page yielded no papers")
        if duplicate_rows:
            LOGGER.warning(
                "KDD %s official program contained %s exact duplicate DOI rows; "
                "kept %s canonical papers from %s rows",
                year,
                duplicate_rows,
                len(papers),
                raw_rows,
            )
        return list(papers.values())

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        if year in self.OFFICIAL_PAGE_BY_YEAR:
            return self._collect_official_2026(year, client)
        return super().collect_papers(year, client)

    def resolve_open_access_pdf_urls(
        self,
        papers: Iterable[CorpusPaper],
        client: OfficialProceedingsClient,
    ) -> dict[str, int]:
        """Resolve DOI-identified KDD papers to non-ACM public PDF mirrors.

        ACM's edge currently challenges plain HTTP and browser fetches on the
        configured exits. Semantic Scholar's batch endpoint is used only as an
        identifier bridge: the returned DOI and normalized title must match the
        official KDD row before an arXiv or other direct OA PDF is preferred.
        The publisher PDF remains an alternate and the DOI landing page remains
        the paper page, preserving provenance.
        """
        selected = list(papers)
        by_doi = {
            _acm_doi_of(paper): paper
            for paper in selected
            if _acm_doi_of(paper)
        }
        stats = {
            "requested": len(by_doi),
            "matched": 0,
            "resolved": 0,
            "arxiv": 0,
            "other_oa": 0,
            "not_found": 0,
            "identity_mismatch": 0,
        }
        endpoint = "https://api.semanticscholar.org/graph/v1/paper/batch"
        fields = "paperId,title,authors,externalIds,openAccessPdf"

        items = list(by_doi.items())
        for start in range(0, len(items), 500):
            if start:
                time.sleep(2.0)
            batch = items[start:start + 500]
            response = client.post_json(
                endpoint,
                params={"fields": fields},
                body={"ids": [f"DOI:{doi}" for doi, _ in batch]},
            )
            if not isinstance(response, list) or len(response) != len(batch):
                raise RuntimeError(
                    "Semantic Scholar batch response did not preserve request cardinality"
                )

            for (doi, paper), record in zip(batch, response):
                if not isinstance(record, dict):
                    stats["not_found"] += 1
                    continue
                external_ids = record.get("externalIds") or {}
                returned_doi = str(external_ids.get("DOI") or "").lower()
                returned_title = str(record.get("title") or "")
                if returned_doi != doi or normalize_title_key(returned_title) != normalize_title_key(paper.title):
                    stats["identity_mismatch"] += 1
                    continue
                stats["matched"] += 1

                resolved_url = ""
                resolution_kind = ""
                arxiv_id = str(external_ids.get("ArXiv") or "").strip()
                if arxiv_id:
                    resolved_url = f"https://arxiv.org/pdf/{arxiv_id}"
                    resolution_kind = "arxiv"
                else:
                    oa_url = str((record.get("openAccessPdf") or {}).get("url") or "").strip()
                    host = (urlparse(oa_url).hostname or "").lower()
                    if oa_url.startswith("https://") and host not in {"doi.org", "dl.acm.org"}:
                        resolved_url = oa_url
                        resolution_kind = "other_oa"

                if not resolved_url:
                    continue
                publisher_pdf_url = paper.pdf_url
                paper.alternate_pdf_urls = [
                    url
                    for url in [publisher_pdf_url, *paper.alternate_pdf_urls]
                    if url and url != resolved_url
                ]
                paper.pdf_url = resolved_url
                paper.pdf_resolution_status = "resolved_open_access"
                paper.pdf_resolution_note = (
                    f"Resolved via Semantic Scholar {resolution_kind}; official DOI and title matched."
                )
                stats["resolved"] += 1
                stats[resolution_kind] += 1

        return stats


class SigirOfficialProgramSource(DblpAcmProceedingsSource):
    """SIGIR 2026 from the conference's final program, DBLP for older years.

    The final program exposes archival track badges and ACM DOI identities in
    static HTML. That is stricter than the preliminary accepted-papers page and
    avoids DBLP's current anti-bot response. Only explicitly enumerated main-
    conference tracks are retained; demos and journal-to-conference talks are
    known non-corpus entries, and every unknown badge fails closed.
    """

    OFFICIAL_PROGRAM_BY_YEAR = {
        2026: "https://sigir2026.org/program.html",
    }
    PROCEEDINGS_DOI_PREFIX_BY_YEAR = {
        2026: "10.1145/3805712.",
    }
    MAIN_TRACK_BY_BADGE = {
        "full": "research",
        "short": "short",
        "industry": "industry",
        "resource": "resource",
        "reproducibility": "reproducibility",
        "low-resource": "low_resource",
        "perspective": "perspective",
    }
    EXCLUDED_BADGES = {
        "demo",
        "tois",
    }

    def __init__(
        self,
        conference_name: str = "SIGIR",
        dblp_venue_key: str = "sigir",
        dblp_toc_overrides: Optional[dict[int, str]] = None,
    ) -> None:
        super().__init__(conference_name, dblp_venue_key, dblp_toc_overrides)

    def _collect_official_program(
        self,
        year: int,
        client: OfficialProceedingsClient,
    ) -> list[CorpusPaper]:
        page_url = self.OFFICIAL_PROGRAM_BY_YEAR[year]
        page_text = client.get_text(page_url)
        soup = BeautifulSoup(page_text, "html.parser")
        papers: dict[str, CorpusPaper] = {}
        raw_rows = 0
        duplicate_rows = 0
        excluded_counts: dict[str, int] = {}
        expected_doi_prefix = self.PROCEEDINGS_DOI_PREFIX_BY_YEAR[year]

        for row in soup.select("div.paper"):
            raw_rows += 1
            badge_node = row.select_one(".badge")
            badge = (
                " ".join(badge_node.get_text(" ", strip=True).split()).casefold()
                if badge_node is not None
                else ""
            )
            if badge in self.EXCLUDED_BADGES:
                excluded_counts[badge] = excluded_counts.get(badge, 0) + 1
                continue
            track = self.MAIN_TRACK_BY_BADGE.get(badge)
            if track is None:
                raise RuntimeError(
                    f"SIGIR {year} official program contains unknown badge {badge!r}"
                )

            title_node = row.select_one("a.ptitle[href]")
            authors_node = row.select_one(".pauth")
            abstract_node = row.select_one(".abstxt")
            title = (
                " ".join(title_node.get_text(" ", strip=True).split())
                if title_node is not None
                else ""
            )
            doi_url = str(title_node.get("href") or "").strip() if title_node else ""
            authors = (
                [
                    " ".join(author.get_text(" ", strip=True).split())
                    for author in authors_node.select("a.au")
                    if author.get_text(" ", strip=True)
                ]
                if authors_node is not None
                else []
            )
            abstract = (
                " ".join(abstract_node.get_text(" ", strip=True).split())
                if abstract_node is not None
                else ""
            )
            parsed = urlparse(doi_url)
            doi = parsed.path.lstrip("/")
            if (
                not title
                or not authors
                or parsed.scheme != "https"
                or parsed.hostname != "doi.org"
                or not doi.startswith(expected_doi_prefix)
            ):
                raise RuntimeError(
                    "SIGIR "
                    f"{year} official {badge!r} row lacks title/authors/ACM DOI: "
                    f"{title!r} {doi_url!r}"
                )

            doi_suffix = doi.removeprefix(self.PROCEEDINGS_DOI_PREFIX)
            paper_id = f"{self.dblp_venue_key}-{year}-{doi_suffix}"
            paper = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                pdf_url=f"https://dl.acm.org/doi/pdf/{doi}",
                paper_page_url=f"https://doi.org/{doi}",
                source_name="sigir_official_program",
                abstract=abstract,
                venue_track=track,
            )
            previous = papers.get(paper_id)
            if previous is not None and previous != paper:
                raise RuntimeError(f"SIGIR {year} conflicting duplicate DOI for {paper_id}")
            if previous is not None:
                duplicate_rows += 1
            papers[paper_id] = paper

        if not papers:
            raise RuntimeError(f"SIGIR {year} official program yielded no main papers")
        LOGGER.info(
            "SIGIR %s official program: %s canonical main papers from %s rows; "
            "%s duplicate rows; excluded=%s",
            year,
            len(papers),
            raw_rows,
            duplicate_rows,
            excluded_counts,
        )
        return list(papers.values())

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        if year in self.OFFICIAL_PROGRAM_BY_YEAR:
            return self._collect_official_program(year, client)
        return super().collect_papers(year, client)


class UsenixProceedingsSource(OfficialProceedingsSource):
    """Generic USENIX open-access proceedings (FAST, USENIX ATC, Security, ...).

    USENIX hosts every conference on the same site with a uniform layout:
        index: https://www.usenix.org/conference/{url_slug}{YY}/technical-sessions
               (one ``<article class="node-paper">`` per accepted paper)
        pdf:   https://www.usenix.org/system/files/{file_prefix}{YY}-{slug}.pdf
               where ``slug`` is the trailing path of the per-paper
               ``/conference/{url_slug}{YY}/presentation/{slug}`` URL.

    The deterministic PDF URL avoids fetching one detail page per paper.
    Non-paper entries (keynote / panel / award / ...) are dropped via the
    sub-type taxonomy. Verified 2026-06-03: ``fast24-{slug}.pdf`` and
    ``atc24-{slug}.pdf`` both resolve (200, application/pdf).
    """

    # Sub-type taxonomy values that are NOT research papers.
    _NON_PAPER_SUBTYPES = {
        "Keynote", "Keynote Address", "Plenary", "Panel", "Award Talk",
        "Test of Time Award", "Distinguished Lecture", "Invited Talk",
        "Townhall", "Town Hall", "Conference Reception", "Lightning Talks",
    }

    def __init__(
        self,
        conference_name: str,
        url_slug: str,
        paper_id_prefix: Optional[str] = None,
        file_prefix: Optional[str] = None,
        source_name: Optional[str] = None,
    ):
        self.conference_name = conference_name
        self.url_slug = url_slug                       # path slug, e.g. "fast", "atc", "usenixsecurity"
        self.paper_id_prefix = paper_id_prefix or url_slug   # paper_id namespace, e.g. "uss"
        self._file_prefix = file_prefix or url_slug    # PDF filename prefix; default == url_slug
        self.source_name = source_name or f"usenix_{url_slug}_techsessions"

    def supports_year(self, year: int) -> bool:
        return 2015 <= year <= 2030

    def pdf_file_prefix(self, year: int, yy: str) -> str:
        """PDF filename prefix for a year. Override for venues that changed it."""
        return f"{self._file_prefix}{yy}"

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        yy = f"{year % 100:02d}"
        index_url = f"https://www.usenix.org/conference/{self.url_slug}{yy}/technical-sessions"
        soup = client.get_soup(index_url)

        slug_re = re.compile(rf"^/conference/{self.url_slug}{yy}/presentation/(?P<slug>[^/?#]+)$")
        papers: dict[str, CorpusPaper] = {}

        for article in soup.select("article.node-paper"):
            h2_anchor = article.select_one("h2 a[href]")
            if h2_anchor is None:
                continue
            slug_match = slug_re.match(h2_anchor["href"])
            if slug_match is None:
                continue
            slug = slug_match.group("slug")
            title = h2_anchor.get_text(" ", strip=True)
            if not title:
                continue

            # Reject non-paper entries (keynote/panel/etc) by sub-type taxonomy.
            subtype_node = article.select_one(".field-name-field-paper-sub-type .field-item")
            if subtype_node is not None:
                subtype = subtype_node.get_text(" ", strip=True)
                if subtype in self._NON_PAPER_SUBTYPES:
                    continue

            author_node = article.select_one(".field-name-field-paper-people-text")
            if author_node is None:
                # Real papers always have an author block; missing one usually means
                # a panel / keynote / awards entry that escaped the subtype filter.
                continue
            authors: list[str] = []
            # Strip the affiliation <em> nodes; remaining text is the author list.
            for em in author_node.select("em"):
                em.extract()
            raw = author_node.get_text(" ", strip=True)
            for chunk in re.split(r"[;,]\s*|\s+and\s+", raw):
                chunk = chunk.strip(" ,;")
                if chunk:
                    authors.append(chunk)

            paper_id = f"{self.paper_id_prefix}-{year}-{slug}"
            pdf_url = f"https://www.usenix.org/system/files/{self.pdf_file_prefix(year, yy)}-{slug}.pdf"
            paper_page_url = urljoin(index_url, h2_anchor["href"])

            papers[paper_id] = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                pdf_url=pdf_url,
                paper_page_url=paper_page_url,
                source_name=self.source_name,
            )

        return list(papers.values())


class UsenixSecuritySource(UsenixProceedingsSource):
    """USENIX Security Symposium — fully open access at usenix.org.

    Thin specialization of :class:`UsenixProceedingsSource` that preserves the
    legacy ``uss-{year}-{slug}`` paper_id (existing indexed data depends on it)
    and the pre-2023 ``sec{YY}-`` PDF filename prefix (2023+ switched to the
    modern ``usenixsecurity{YY}-`` form).
    """

    conference_name = "USENIX Security"

    def __init__(self):
        super().__init__(
            conference_name="USENIX Security",
            url_slug="usenixsecurity",
            paper_id_prefix="uss",
            file_prefix="usenixsecurity",
            source_name="usenix_security_techsessions",
        )

    def supports_year(self, year: int) -> bool:
        return 2018 <= year <= 2030

    def pdf_file_prefix(self, year: int, yy: str) -> str:
        # 2018-2022 used "sec{YY}-", 2023+ uses "usenixsecurity{YY}-".
        return f"sec{yy}" if year <= 2022 else f"usenixsecurity{yy}"


class NdssSymposiumSource(OfficialProceedingsSource):
    """NDSS Symposium — fully open access at ndss-symposium.org (no Cloudflare).

    The authoritative MAIN-conference list is the per-year accepted-papers index:
        https://www.ndss-symposium.org/ndss{year}/accepted-papers/
    Co-located workshops (USEC / WOSOC / MADWeb / BAR / ...) and the poster track
    live on *separate* pages, so the main symposium is the only thing that appears
    here — non-main tracks are structurally excluded, not heuristically filtered.

    Each entry links to ``/ndss-paper/{slug}/`` whose detail page carries the
    camera-ready PDF at ``/wp-content/uploads/{year}-{cycleid}-paper.pdf`` (cycleid
    e.g. ``s1950`` for the Summer cycle, ``f0042`` for the Fall cycle). We fetch
    every detail page so each paper is verified one-by-one: we take ONLY the
    ``*-paper.pdf`` link and explicitly reject ``*-slides.pdf`` / ``*-poster.pdf``
    and other non-paper artifacts. A main paper whose camera-ready PDF is not yet
    posted is emitted but flagged ``missing_camera_ready`` rather than being given
    a slides URL — never silently contaminated (the W90 lesson, applied at source).

    Verified 2026-06-15: DBLP ``conf/ndss/ndss2026`` (265 inproceedings) and the
    accepted-papers index (265 ndss-paper links) agree exactly; 2025 = 211/211.
    """

    conference_name = "NDSS"

    # PDF filename markers that are NOT the camera-ready research paper.
    _NON_PAPER_PDF_MARKERS = (
        "-slides", "-poster", "-supplement", "-supplementary",
        "-appendix", "-talk", "-abstract",
    )

    def __init__(self, source_name: Optional[str] = None):
        self.source_name = source_name or "ndss_accepted_papers"

    def supports_year(self, year: int) -> bool:
        return 2016 <= year <= 2030

    def _accepted_papers_url(self, year: int) -> str:
        return f"https://www.ndss-symposium.org/ndss{year}/accepted-papers/"

    @staticmethod
    def _parse_authors(raw: str) -> list[str]:
        """Split an ``Name (Affiliation, ...), Name (Affiliation), ...`` block.

        Commas inside the parenthesised affiliation must NOT split authors, so we
        split on ``)`` followed by a comma and drop the parenthesised part.
        """
        authors: list[str] = []
        for chunk in re.split(r"\)\s*,\s*", raw or ""):
            name = chunk.split("(")[0].strip(" ,; ")
            if name:
                authors.append(name)
        return authors

    def _collect_paper_page_urls(self, index_soup: BeautifulSoup, index_url: str) -> list[tuple[str, str]]:
        """Return ordered, de-duplicated (slug, absolute paper-page url) pairs."""
        ordered: list[tuple[str, str]] = []
        seen: set[str] = set()
        for anchor in index_soup.select('a[href*="/ndss-paper/"]'):
            href = anchor.get("href", "")
            match = re.search(r"/ndss-paper/([^/?#]+)/?$", href)
            if match is None:
                continue
            slug = match.group(1)
            if slug in seen:
                continue
            seen.add(slug)
            ordered.append((slug, urljoin(index_url, href)))
        return ordered

    def collect_papers(self, year: int, client: OfficialProceedingsClient) -> list[CorpusPaper]:
        index_url = self._accepted_papers_url(year)
        index_soup = client.get_soup(index_url)
        page_urls = self._collect_paper_page_urls(index_soup, index_url)

        papers: dict[str, CorpusPaper] = {}
        for slug, page_url in page_urls:
            try:
                page = client.get_soup(page_url)
            except Exception as exc:  # one bad page must not abort the whole venue
                LOGGER.warning("NDSS: failed to fetch paper page %s: %s", page_url, exc)
                continue

            # camera-ready PDF: the *-paper.pdf only; slides/poster never leak.
            pdf_url = ""
            alternates: list[str] = []
            for anchor in page.select('a[href$=".pdf"]'):
                href = urljoin(page_url, anchor.get("href", ""))
                low = href.lower()
                if low.endswith("-paper.pdf"):
                    pdf_url = href
                elif any(marker in low for marker in self._NON_PAPER_PDF_MARKERS):
                    continue  # drop slides / poster / supplement entirely
                else:
                    alternates.append(href)

            heading = page.select_one("h1")
            title = heading.get_text(" ", strip=True) if heading else slug.replace("-", " ").title()
            if not title:
                continue

            # Authors: prefer the dedicated .paper-data block; fall back to the
            # first paragraph of .entry-content.
            data_node = page.select_one(".paper-data")
            content_node = page.select_one(".entry-content")
            author_text = ""
            if data_node is not None:
                author_text = data_node.get_text(" ", strip=True)
            elif content_node is not None:
                first_p = content_node.find("p")
                if first_p is not None:
                    author_text = first_p.get_text(" ", strip=True)
            authors = self._parse_authors(author_text)

            # Abstract: first substantial paragraph that is not the author line.
            abstract = ""
            if content_node is not None:
                for para in content_node.find_all("p"):
                    text = para.get_text(" ", strip=True)
                    if text and text != author_text and len(text) > 80:
                        abstract = text
                        break

            # Stable id from the cycle id embedded in the camera-ready filename
            # (s1950 / f0042); fall back to the URL slug when the PDF is absent.
            cycle_id = None
            if pdf_url:
                cid_match = re.search(rf"/{year}-([A-Za-z]?\d+[A-Za-z]?)-paper\.pdf$", pdf_url)
                if cid_match:
                    cycle_id = cid_match.group(1)
            paper_id = f"ndss-{year}-{cycle_id or slug}"

            paper = CorpusPaper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                conference=self.conference_name,
                year=year,
                pdf_url=pdf_url,
                paper_page_url=page_url,
                source_name=self.source_name,
                abstract=abstract,
                alternate_pdf_urls=alternates,
            )
            if not pdf_url:
                paper.pdf_resolution_status = "missing_camera_ready"
                paper.pdf_resolution_note = (
                    "Main accepted paper has no -paper.pdf on its NDSS page yet "
                    "(camera-ready not posted); not substituting slides/alternates."
                )
            papers[paper_id] = paper

        return list(papers.values())


_SOURCE_CLASS_MAP: dict[str, type] = {
    "AaaiTechnicalTracksSource": AaaiTechnicalTracksSource,
    "AclMainConferenceSource": AclMainConferenceSource,
    "CvfProceedingsSource": CvfProceedingsSource,
    "DblpAcmProceedingsSource": DblpAcmProceedingsSource,
    "EmnlpMainConferenceSource": EmnlpMainConferenceSource,
    "IclrOpenReviewSource": IclrOpenReviewSource,
    "IcmlProceedingsSource": IcmlProceedingsSource,
    "IjcaiProceedingsSource": IjcaiProceedingsSource,
    "KddOfficialProceedingsSource": KddOfficialProceedingsSource,
    "NdssSymposiumSource": NdssSymposiumSource,
    "NeuripsProceedingsSource": NeuripsProceedingsSource,
    "SigirOfficialProgramSource": SigirOfficialProgramSource,
    "UsenixProceedingsSource": UsenixProceedingsSource,
    "UsenixSecuritySource": UsenixSecuritySource,
}

_CONF_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "conf" / "conferences.yaml"


def load_conference_registry(yaml_path: Optional[Path] = None) -> list[dict[str, Any]]:
    path = yaml_path or _CONF_YAML_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        entries = raw.get("conferences", [])
        categories = raw.get("categories", {})
    else:
        entries = raw
        categories = {}
    registry = []
    for entry in entries:
        cls_name = entry["source_class"]
        cls = _SOURCE_CLASS_MAP.get(cls_name)
        title = entry.get("title", entry.get("name", ""))
        if cls is None:
            LOGGER.warning("Unknown source class %s for %s, skipping", cls_name, title)
            continue
        args = entry.get("source_args", {})
        source = cls(**args) if args else cls()
        rank = entry.get("rank", {})
        ccf_tier = rank.get("ccf", "")
        venue_tier = f"CCF-{ccf_tier}" if ccf_tier else entry.get("venue_tier", "")
        sub = entry.get("sub", "")
        cat = categories.get(sub, {})
        enriched = {
            **entry,
            "name": title,
            "venue_tier": venue_tier,
            "sub_name": cat.get("name", ""),
            "sub_name_en": cat.get("name_en", ""),
            "source": source,
        }
        registry.append(enriched)
    return registry


def load_categories(yaml_path: Optional[Path] = None) -> dict[str, dict[str, str]]:
    path = yaml_path or _CONF_YAML_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw.get("categories", {})
    return {}


def build_sources(yaml_path: Optional[Path] = None) -> dict[str, OfficialProceedingsSource]:
    return {
        entry["name"]: entry["source"]
        for entry in load_conference_registry(yaml_path)
        if entry.get("source") is not None
    }


class MainConferenceCorpusBuilder:
    def __init__(
        self,
        corpus_root: Optional[Path] = None,
        pdf_root: Optional[Path] = None,
        proxy_url: str = DEFAULT_PROXY_URL,
        years: Iterable[int] = DEFAULT_TARGET_YEARS,
        conferences: Iterable[str] = DEFAULT_TARGET_CONFERENCES,
        sample_limit_per_conference_year: Optional[int] = None,
        manifest_checkpoint_interval: int = DEFAULT_MANIFEST_CHECKPOINT_INTERVAL,
        download_retry_attempts: int = DEFAULT_DOWNLOAD_RETRY_ATTEMPTS,
        download_concurrency: int = DEFAULT_DOWNLOAD_CONCURRENCY,
    ):
        self.corpus_root = ensure_directory(corpus_root or build_default_corpus_root())
        self.manifest_dir = ensure_directory(self.corpus_root / "manifests")
        # Artifact roots may be Windows links to an authenticated SMB share.
        # Metadata collection must not touch that mount; download/extraction
        # create the concrete venue directory only when the stage really runs.
        self.pdf_root = Path(pdf_root) if pdf_root is not None else self.corpus_root / "pdf"
        self.markdown_root = self.corpus_root / "markdown"
        self.report_dir = ensure_directory(self.corpus_root / "reports")
        self.proxy_url = proxy_url
        self.years = tuple(sorted(years))
        self.conferences = tuple(conferences)
        self.sample_limit_per_conference_year = sample_limit_per_conference_year
        self.manifest_checkpoint_interval = max(1, manifest_checkpoint_interval)
        self.download_retry_attempts = max(1, download_retry_attempts)
        self.download_concurrency = max(1, download_concurrency)
        self.sources = build_sources()

    @property
    def manifest_path(self) -> Path:
        return self.manifest_dir / "paper_manifest.json"

    @property
    def scope_path(self) -> Path:
        return self.manifest_dir / "build_scope.json"

    @property
    def summary_path(self) -> Path:
        return self.report_dir / "build_summary.json"

    def reset_corpus_workspace(self) -> None:
        for path in (self.manifest_dir, self.pdf_root, self.markdown_root, self.report_dir):
            if path.exists():
                shutil.rmtree(path)
            ensure_directory(path)

    def build_markdown_document(self, paper: CorpusPaper, page_sections: Optional[list[str]] = None) -> str:
        header_lines = [
            "---",
            f"paper_id: {paper.paper_id}",
            f"title: {paper.title}",
            f"conference: {paper.conference}",
            f"year: {paper.year}",
            f"source_name: {paper.source_name}",
            f"pdf_url: {paper.pdf_url}",
            f"downloaded_from_url: {paper.downloaded_from_url or paper.pdf_url}",
            f"paper_page_url: {paper.paper_page_url}",
            f"selection_rule: {MAIN_CONFERENCE_RULE}",
        ]

        if paper.pdf_resolution_status:
            header_lines.append(f"pdf_resolution_status: {paper.pdf_resolution_status}")
        if paper.pdf_resolution_note:
            header_lines.append(f"pdf_resolution_note: {paper.pdf_resolution_note}")
        if paper.retraction_notice_url:
            header_lines.append(f"retraction_notice_url: {paper.retraction_notice_url}")

        header_lines.extend(["---", "", f"# {paper.title}"])

        if paper.authors:
            header_lines.append("")
            header_lines.append("Authors: " + ", ".join(paper.authors))

        if paper.abstract:
            header_lines.extend(["", "## Abstract", "", paper.abstract])

        if page_sections:
            return "\n".join(header_lines) + "\n\n" + "\n\n".join(page_sections).strip() + "\n"

        availability_lines = [
            "",
            "## Availability",
            "",
            "Full-text PDF is currently unavailable for this paper in the rebuild pipeline.",
        ]
        if paper.pdf_resolution_status:
            availability_lines.append(f"Resolution status: {paper.pdf_resolution_status}")
        if paper.pdf_resolution_note:
            availability_lines.append(f"Resolution note: {paper.pdf_resolution_note}")
        if paper.retraction_notice_url:
            availability_lines.append(f"Retraction notice: {paper.retraction_notice_url}")

        return "\n".join(header_lines + availability_lines) + "\n"

    def save_scope(self) -> None:
        scope = {
            "corpus_name": CORPUS_NAME,
            "selection_rule": MAIN_CONFERENCE_RULE,
            "years": list(self.years),
            "conferences": list(self.conferences),
            "proxy_url": self.proxy_url,
            "sample_limit_per_conference_year": self.sample_limit_per_conference_year,
            "manifest_checkpoint_interval": self.manifest_checkpoint_interval,
            "download_retry_attempts": self.download_retry_attempts,
            "download_concurrency": self.download_concurrency,
        }
        write_json_atomic(self.scope_path, scope)

    def load_manifest(self) -> list[CorpusPaper]:
        records = load_json_with_recovery(self.manifest_path, default_value=[])
        return [CorpusPaper.from_manifest_record(record) for record in records]

    def save_manifest(self, papers: Iterable[CorpusPaper]) -> None:
        records = [paper.to_manifest_record() for paper in papers]
        write_json_atomic(self.manifest_path, records)

    def load_summary(self) -> dict[str, Any]:
        return load_json_with_recovery(self.summary_path, default_value={})

    def repair_unresolved_pdf_urls(self, papers: Iterable[CorpusPaper]) -> list[CorpusPaper]:
        repaired_papers = list(papers)
        cvf_cache: dict[tuple[str, int], dict[str, CorpusPaper]] = {}

        with OfficialProceedingsClient(proxy_url=self.proxy_url) as client:
            for paper in repaired_papers:
                if paper.pdf_path:
                    continue

                if paper.conference in {"CVPR", "ICCV"} and paper.source_name == "cvf_open_access":
                    cache_key = (paper.conference, paper.year)
                    if cache_key not in cvf_cache:
                        source = CvfProceedingsSource(paper.conference)
                        refreshed_records = source.collect_papers(paper.year, client)
                        cvf_cache[cache_key] = {
                            normalize_title_key(record.title): record
                            for record in refreshed_records
                        }

                    refreshed_record = cvf_cache[cache_key].get(normalize_title_key(paper.title))
                    if refreshed_record is not None:
                        paper.pdf_url = refreshed_record.pdf_url
                        paper.paper_page_url = refreshed_record.paper_page_url
                        paper.alternate_pdf_urls = list(refreshed_record.alternate_pdf_urls)
                        if paper.pdf_resolution_status and paper.pdf_resolution_status.startswith("unresolved_"):
                            paper.pdf_resolution_status = None
                            paper.pdf_resolution_note = None
                            paper.download_error = None
                    continue

                if paper.conference == "IJCAI":
                    try:
                        soup = client.get_soup(paper.paper_page_url)
                        citation_pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
                        if citation_pdf_meta is not None and citation_pdf_meta.get("content"):
                            paper.pdf_url = citation_pdf_meta["content"].strip()
                            paper.download_error = None
                            if paper.pdf_resolution_status == "download_failed":
                                paper.pdf_resolution_status = None
                                paper.pdf_resolution_note = None
                    except Exception as exc:
                        LOGGER.warning("Failed to refresh IJCAI PDF URL for %s: %s", paper.paper_id, exc)

        self.save_manifest(repaired_papers)
        return repaired_papers

    def save_manifest_checkpoint(
        self,
        papers: Iterable[CorpusPaper],
        stage_name: str,
        processed_count: int,
        total_count: int,
    ) -> None:
        LOGGER.info(
            "Saving manifest checkpoint for %s (%s/%s)",
            stage_name,
            processed_count,
            total_count,
        )
        self.save_manifest(papers)

    def collect_manifest(self) -> list[CorpusPaper]:
        client = OfficialProceedingsClient(proxy_url=self.proxy_url)
        all_papers: dict[str, CorpusPaper] = {}
        try:
            for conference_name in self.conferences:
                source = self.sources[conference_name]
                for year in self.years:
                    if not source.supports_year(year):
                        LOGGER.info("Skipping %s %s because there was no main-conference edition", conference_name, year)
                        continue
                    LOGGER.info("Collecting official proceedings records for %s %s", conference_name, year)
                    collected_papers = sorted(
                        source.collect_papers(year, client),
                        key=lambda paper: paper.paper_id,
                    )
                    if self.sample_limit_per_conference_year is not None:
                        collected_papers = collected_papers[: self.sample_limit_per_conference_year]
                    for paper in collected_papers:
                        all_papers[paper.paper_id] = paper
                    self.save_manifest(all_papers.values())
        finally:
            client.close()

        self.save_scope()
        self.save_manifest(all_papers.values())
        return list(all_papers.values())

    def quarantine_active_indices(self) -> None:
        backup_root = ensure_directory(Path(retrieval_settings.data_dir) / "backups")
        backup_dir = backup_root / "pre_ccf_a_main_conference_corpus_reset"

        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        move_targets = [
            (DB_PATH, backup_dir / "retrieval.db"),
            (Path(retrieval_settings.chroma_dir), backup_dir / "indices" / "chroma"),
            (Path(retrieval_settings.whoosh_dir), backup_dir / "indices" / "whoosh"),
            (
                Path(retrieval_settings.index_dir) / "papers_paper_doc_mapping.json",
                backup_dir / "indices" / "papers_paper_doc_mapping.json",
            ),
        ]

        for source_path, backup_path in move_targets:
            if not source_path.exists():
                continue
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(backup_path))

        ensure_directory(DB_PATH.parent)
        ensure_directory(Path(retrieval_settings.chroma_dir))
        ensure_directory(Path(retrieval_settings.whoosh_dir))
        init_db()
        self.reset_corpus_workspace()

    async def download_papers(
        self,
        papers: Iterable[CorpusPaper],
        max_concurrent: Optional[int] = None,
        min_request_interval_seconds: float = 0.0,
        max_bytes_per_second: int | None = DEFAULT_DOWNLOAD_BYTES_PER_SECOND,
        chained_proxy_pool: ChainedProxyPool | None = None,
        proxy_attempts_per_url: int = 24,
    ) -> list[CorpusPaper]:
        downloaded_papers = list(papers)
        total_count = len(downloaded_papers)
        proxy_url = require_explicit_proxy(self.proxy_url)
        if max_bytes_per_second is not None and max_bytes_per_second < 0:
            raise ValueError("max_bytes_per_second must be non-negative")
        if proxy_attempts_per_url <= 0:
            raise ValueError("proxy_attempts_per_url must be positive")
        resolved_max_concurrent = max(
            1,
            min(int(max_concurrent or self.download_concurrency), total_count or 1),
        )
        min_request_interval_seconds = max(0.0, float(min_request_interval_seconds or 0.0))
        queue: asyncio.Queue[CorpusPaper] = asyncio.Queue()
        for paper in downloaded_papers:
            queue.put_nowait(paper)

        progress_lock = asyncio.Lock()
        request_lock = asyncio.Lock()
        next_request_at = 0.0
        processed_count = 0
        limiter = (
            AggregateDownloadRateLimiter(max_bytes_per_second)
            if max_bytes_per_second
            else None
        )
        await asyncio.to_thread(require_reachable_proxy, proxy_url)

        client_options = {
            "proxy": proxy_url,
            "timeout": 120,
            "follow_redirects": True,
            "trust_env": False,
            "headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,*/*;q=0.8",
                "Accept-Encoding": "identity",
            },
        }
        cookies = load_openreview_cookies_from_env()
        if cookies is not None:
            client_options["cookies"] = cookies

        async with httpx.AsyncClient(**client_options) as client:
            async def mark_progress() -> None:
                nonlocal processed_count
                async with progress_lock:
                    processed_count += 1
                    if (
                        processed_count % self.manifest_checkpoint_interval == 0
                        or processed_count == total_count
                    ):
                        self.save_manifest_checkpoint(
                            downloaded_papers,
                            stage_name="download",
                            processed_count=processed_count,
                            total_count=total_count,
                        )

            async def download_single(paper: CorpusPaper) -> None:
                nonlocal next_request_at
                pdf_path = self.pdf_root / paper.conference.lower() / str(paper.year) / f"{sanitize_file_component(paper.paper_id)}.pdf"
                pdf_path.parent.mkdir(parents=True, exist_ok=True)

                if is_valid_pdf_artifact(pdf_path):
                    paper.pdf_path = str(pdf_path)
                    paper.downloaded_from_url = paper.downloaded_from_url or paper.pdf_url
                    paper.pdf_resolution_status = paper.pdf_resolution_status or "downloaded"
                    paper.pdf_resolution_note = paper.pdf_resolution_note or "Primary or previously resolved PDF is available locally."
                    paper.download_error = None
                    await mark_progress()
                    return
                remove_file_if_present(pdf_path)

                retraction_metadata = KNOWN_RETRACTED_PAPERS.get(paper.paper_id)
                if retraction_metadata is not None:
                    paper.pdf_path = None
                    paper.downloaded_from_url = None
                    paper.pdf_resolution_status = "retracted"
                    paper.pdf_resolution_note = retraction_metadata["notice"]
                    paper.retraction_notice_url = retraction_metadata["notice_url"]
                    paper.download_error = None
                    await mark_progress()
                    return

                try:
                    temporary_pdf_path = build_write_in_progress_path(pdf_path)
                    last_error: Optional[Exception] = None
                    candidate_urls = [paper.pdf_url, *paper.alternate_pdf_urls]
                    unique_candidate_urls: list[str] = []
                    for candidate_url in candidate_urls:
                        normalized_candidate_url = candidate_url.strip()
                        if not normalized_candidate_url or normalized_candidate_url in unique_candidate_urls:
                            continue
                        unique_candidate_urls.append(normalized_candidate_url)

                    for candidate_url in unique_candidate_urls:
                        attempts = (
                            1 if chained_proxy_pool is not None
                            else self.download_retry_attempts
                        )
                        for attempt in range(1, attempts + 1):
                            try:
                                remove_file_if_present(temporary_pdf_path)
                                if min_request_interval_seconds > 0:
                                    async with request_lock:
                                        now = time.monotonic()
                                        if now < next_request_at:
                                            await asyncio.sleep(next_request_at - now)
                                        next_request_at = time.monotonic() + min_request_interval_seconds
                                if chained_proxy_pool is not None:
                                    await chained_proxy_pool.download_to_path(
                                        candidate_url,
                                        temporary_pdf_path,
                                        max_attempts=proxy_attempts_per_url,
                                        validator=is_valid_pdf_artifact,
                                        rate_limiter=limiter,
                                        cookies=cookies,
                                    )
                                else:
                                    async with client.stream("GET", candidate_url) as response:
                                        response.raise_for_status()
                                        iterator = response.aiter_bytes(DEFAULT_STREAM_CHUNK_BYTES).__aiter__()
                                        with temporary_pdf_path.open("wb") as output_file:
                                            while True:
                                                if limiter is not None:
                                                    await limiter.wait_for_next_read_async()
                                                try:
                                                    chunk = await iterator.__anext__()
                                                except StopAsyncIteration:
                                                    break
                                                if not chunk:
                                                    continue
                                                output_file.write(chunk)
                                if not is_valid_pdf_artifact(temporary_pdf_path):
                                    raise ValueError(f"Downloaded PDF validation failed for {paper.paper_id}")
                                temporary_pdf_path.replace(pdf_path)
                                paper.pdf_path = str(pdf_path)
                                paper.downloaded_from_url = candidate_url
                                paper.pdf_resolution_status = "downloaded"
                                paper.pdf_resolution_note = f"Resolved PDF from {candidate_url}"
                                paper.download_error = None
                                return
                            except Exception as exc:
                                last_error = exc
                                remove_file_if_present(temporary_pdf_path)
                                if attempt == attempts:
                                    break
                                LOGGER.warning(
                                    "Retrying download for %s from %s (%s/%s): %s",
                                    paper.paper_id,
                                    candidate_url,
                                    attempt,
                                    attempts,
                                    exc,
                                )
                                await asyncio.sleep(http_retry_delay_seconds(exc, attempt))
                    if last_error is not None:
                        raise last_error
                    raise ValueError(f"No usable PDF URL for {paper.paper_id}")
                except Exception as exc:
                    paper.download_error = str(exc)
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code == 404
                    ) or (
                        isinstance(exc, ChainedTargetHTTPError)
                        and exc.status_code in {404, 410}
                    ):
                        paper.pdf_resolution_status = "unresolved_404"
                        paper.pdf_resolution_note = f"All configured PDF sources returned 404 for {paper.paper_id}."
                    else:
                        paper.pdf_resolution_status = "download_failed"
                        paper.pdf_resolution_note = str(exc)
                    LOGGER.warning("Failed to download %s: %s", paper.paper_id, exc)
                    remove_file_if_present(build_write_in_progress_path(pdf_path))
                finally:
                    await mark_progress()

            async def worker() -> None:
                while True:
                    try:
                        paper = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                    try:
                        await download_single(paper)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # `download_single` records per-paper failures itself, so
                        # reaching here means its own bookkeeping broke. One paper
                        # must still not abort the gather: that would discard every
                        # download since the last 200-paper checkpoint.
                        LOGGER.exception(
                            "Unhandled error while downloading %s: %s",
                            paper.paper_id,
                            exc,
                        )
                        paper.download_error = str(exc)
                        paper.pdf_resolution_status = "download_failed"
                        paper.pdf_resolution_note = str(exc)
                    finally:
                        queue.task_done()

            await asyncio.gather(*(worker() for _ in range(min(resolved_max_concurrent, total_count or 1))))

        self.save_manifest(downloaded_papers)
        return downloaded_papers

    def extract_markdown(self, papers: Iterable[CorpusPaper]) -> list[CorpusPaper]:
        extracted_papers = list(papers)
        total_count = len(extracted_papers)
        processed_count = 0

        for paper in extracted_papers:
            markdown_path = self.markdown_root / paper.conference.lower() / str(paper.year) / f"{sanitize_file_component(paper.paper_id)}.md"
            markdown_path.parent.mkdir(parents=True, exist_ok=True)

            if paper.pdf_path is None:
                paper.markdown_path = None
                paper.extract_error = "PDF not downloaded yet"
                if markdown_path.exists():
                    markdown_path.unlink()
                processed_count += 1
                if (
                    processed_count % self.manifest_checkpoint_interval == 0
                    or processed_count == total_count
                ):
                    self.save_manifest_checkpoint(
                        extracted_papers,
                        stage_name="extract",
                        processed_count=processed_count,
                        total_count=total_count,
                    )
                continue

            if is_valid_markdown_artifact(markdown_path, paper.paper_id):
                paper.markdown_path = str(markdown_path)
                paper.extract_error = None
                processed_count += 1
                if (
                    processed_count % self.manifest_checkpoint_interval == 0
                    or processed_count == total_count
                ):
                    self.save_manifest_checkpoint(
                        extracted_papers,
                        stage_name="extract",
                        processed_count=processed_count,
                        total_count=total_count,
                    )
                continue
            if markdown_path.exists():
                markdown_path.unlink()

            try:
                pdf_path = Path(paper.pdf_path)
                reader = PdfReader(str(pdf_path))
                page_sections: list[str] = []
                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        page_sections.append(f"## Page {page_number}\n\n{page_text}")

                if not page_sections:
                    raise ValueError("No extractable text found in the PDF")

                markdown_text = self.build_markdown_document(paper, page_sections=page_sections)
                write_text_atomic(markdown_path, markdown_text)
                paper.markdown_path = str(markdown_path)
                paper.extract_error = None
            except Exception as exc:
                paper.markdown_path = None
                paper.extract_error = str(exc)
                LOGGER.warning("Failed to extract markdown for %s: %s", paper.paper_id, exc)

            processed_count += 1
            if (
                processed_count % self.manifest_checkpoint_interval == 0
                or processed_count == total_count
            ):
                self.save_manifest_checkpoint(
                    extracted_papers,
                    stage_name="extract",
                    processed_count=processed_count,
                    total_count=total_count,
                )

        self.save_manifest(extracted_papers)
        return extracted_papers

    async def index_papers(self, papers: Iterable[CorpusPaper]) -> list[CorpusPaper]:
        from pipeline.paper_crawler.integration.index_integration import PaperIndexConfig, PaperIndexer

        indexed_papers = list(papers)
        indexer = PaperIndexer(
            PaperIndexConfig(
                collection_name="papers",
                data_source="system",
                embedding_device="cuda",
                require_gpu=True,
                max_concurrent_indexing=1,
            )
        )

        total_count = len(indexed_papers)
        for processed_count, paper_record in enumerate(indexed_papers, start=1):
            existing_doc_id = indexer.get_indexed_doc_id(paper_record.paper_id)
            if existing_doc_id is not None:
                paper_record.indexed_doc_id = existing_doc_id
                paper_record.index_error = None
                if (
                    processed_count % self.manifest_checkpoint_interval == 0
                    or processed_count == total_count
                ):
                    self.save_manifest_checkpoint(
                        indexed_papers,
                        stage_name="index",
                        processed_count=processed_count,
                        total_count=total_count,
                    )
                continue

            if paper_record.indexed_doc_id and not paper_record.index_error:
                if (
                    processed_count % self.manifest_checkpoint_interval == 0
                    or processed_count == total_count
                ):
                    self.save_manifest_checkpoint(
                        indexed_papers,
                        stage_name="index",
                        processed_count=processed_count,
                        total_count=total_count,
                    )
                continue

            if paper_record.markdown_path is None:
                paper_record.indexed_doc_id = None
                if (
                    processed_count % self.manifest_checkpoint_interval == 0
                    or processed_count == total_count
                ):
                    self.save_manifest_checkpoint(
                        indexed_papers,
                        stage_name="index",
                        processed_count=processed_count,
                        total_count=total_count,
                    )
                continue

            try:
                paper = paper_record.to_runtime_paper()
                paper_record.indexed_doc_id = await indexer.index_paper(
                    paper=paper,
                    markdown_path=Path(paper_record.markdown_path),
                )
                paper_record.index_error = None
            except Exception as exc:
                paper_record.indexed_doc_id = None
                paper_record.index_error = str(exc)
                LOGGER.warning("Failed to index %s: %s", paper_record.paper_id, exc)

            if (
                processed_count % self.manifest_checkpoint_interval == 0
                or processed_count == total_count
            ):
                self.save_manifest_checkpoint(
                    indexed_papers,
                    stage_name="index",
                    processed_count=processed_count,
                    total_count=total_count,
                )

        self.save_manifest(indexed_papers)
        return indexed_papers

    def write_summary(self, papers: Iterable[CorpusPaper]) -> None:
        paper_list = list(papers)
        by_conference: dict[str, int] = {}
        by_year: dict[str, int] = {}
        by_conference_year: dict[str, int] = {}
        for paper in paper_list:
            by_conference[paper.conference] = by_conference.get(paper.conference, 0) + 1
            year_key = str(paper.year)
            conference_year_key = f"{paper.conference}-{paper.year}"
            by_year[year_key] = by_year.get(year_key, 0) + 1
            by_conference_year[conference_year_key] = by_conference_year.get(conference_year_key, 0) + 1

        summary = {
            "corpus_name": CORPUS_NAME,
            "selection_rule": MAIN_CONFERENCE_RULE,
            "years": list(self.years),
            "conferences": list(self.conferences),
            "sample_limit_per_conference_year": self.sample_limit_per_conference_year,
            "manifest_checkpoint_interval": self.manifest_checkpoint_interval,
            "download_retry_attempts": self.download_retry_attempts,
            "download_concurrency": self.download_concurrency,
            "paper_count": len(paper_list),
            "downloaded_count": sum(1 for paper in paper_list if paper.pdf_path),
            "extracted_count": sum(1 for paper in paper_list if paper.markdown_path and not paper.extract_error),
            "indexed_count": sum(1 for paper in paper_list if paper.indexed_doc_id),
            "download_failure_count": sum(1 for paper in paper_list if paper.download_error),
            "extract_failure_count": sum(1 for paper in paper_list if paper.extract_error),
            "index_failure_count": sum(1 for paper in paper_list if paper.index_error),
            "by_conference": by_conference,
            "by_year": by_year,
            "by_conference_year": by_conference_year,
        }
        write_json_atomic(self.summary_path, summary)

    def require_manifest_records(self, papers: list[CorpusPaper], stage: str) -> None:
        if papers:
            return

        raise RuntimeError(
            f"No manifest records are available for stage '{stage}'. "
            "Run the collect stage first or restore the formal manifest before continuing."
        )

    def require_completed_stage_snapshot(self, papers: list[CorpusPaper], stage: str) -> None:
        if stage not in {"download", "extract", "index"}:
            return

        if not self.summary_path.exists():
            raise RuntimeError(
                f"Stage '{stage}' requires a completed prior stage snapshot. "
                "Wait for collect to finish and write build_summary.json before continuing."
            )

        summary = self.load_summary()
        expected_count = summary.get("paper_count")
        if isinstance(expected_count, int) and expected_count > 0 and len(papers) != expected_count:
            raise RuntimeError(
                f"Stage '{stage}' refused to start because manifest has {len(papers)} records "
                f"but build_summary.json expects {expected_count}. "
                "Rerun collect to completion before continuing."
            )

    def run(self, stage: str) -> list[CorpusPaper]:
        if stage == "reset":
            self.quarantine_active_indices()
            return []

        if stage == "all":
            self.quarantine_active_indices()

        papers = self.load_manifest()
        if stage in {"collect", "all"}:
            papers = self.collect_manifest()
            self.write_summary(papers)

        if stage == "repair_urls":
            self.require_manifest_records(papers, stage="repair_urls")
            papers = self.repair_unresolved_pdf_urls(papers)

        if stage in {"download", "all"}:
            self.require_manifest_records(papers, stage="download")
            self.require_completed_stage_snapshot(papers, stage="download")
            papers = asyncio.run(self.download_papers(papers))

        if stage in {"extract", "all"}:
            self.require_manifest_records(papers, stage="extract")
            self.require_completed_stage_snapshot(papers, stage="extract")
            papers = self.extract_markdown(papers)

        if stage in {"index", "all"}:
            self.require_manifest_records(papers, stage="index")
            self.require_completed_stage_snapshot(papers, stage="index")
            papers = asyncio.run(self.index_papers(papers))

        self.write_summary(papers)
        return papers


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the CCF-A main-conference-only paper corpus from official proceedings.",
    )
    parser.add_argument(
        "--stage",
        choices=("reset", "collect", "repair_urls", "download", "extract", "index", "all"),
        default="all",
        help="Which stage of the rebuild pipeline to run.",
    )
    parser.add_argument(
        "--proxy",
        default=DEFAULT_PROXY_URL,
        help="HTTP proxy used for all remote downloads and proceedings access.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=list(DEFAULT_TARGET_YEARS),
        help="Fully archived proceedings years to include.",
    )
    parser.add_argument(
        "--conferences",
        nargs="*",
        default=list(DEFAULT_TARGET_CONFERENCES),
        help="Conference names to include.",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=build_default_corpus_root(),
        help="Semantic output root for manifests, PDFs, markdown, and reports.",
    )
    parser.add_argument(
        "--sample-limit-per-conference-year",
        type=int,
        default=None,
        help="Optional smoke-mode limit applied after each conference-year collect step.",
    )
    parser.add_argument(
        "--manifest-checkpoint-interval",
        type=int,
        default=DEFAULT_MANIFEST_CHECKPOINT_INTERVAL,
        help="How many records to process between manifest checkpoints in long-running stages.",
    )
    parser.add_argument(
        "--download-retry-attempts",
        type=int,
        default=DEFAULT_DOWNLOAD_RETRY_ATTEMPTS,
        help="How many times to retry a PDF download before marking it as failed.",
    )
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=DEFAULT_DOWNLOAD_CONCURRENCY,
        help="How many PDF downloads to run concurrently during the long-running download stage.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    builder = MainConferenceCorpusBuilder(
        corpus_root=args.corpus_root,
        proxy_url=args.proxy,
        years=args.years,
        conferences=args.conferences,
        sample_limit_per_conference_year=args.sample_limit_per_conference_year,
        manifest_checkpoint_interval=args.manifest_checkpoint_interval,
        download_retry_attempts=args.download_retry_attempts,
        download_concurrency=args.download_concurrency,
    )
    builder.run(stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
