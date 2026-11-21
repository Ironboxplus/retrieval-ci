"""Official proceedings collectors for ICRA / CoRL / RSS."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from pipeline.robotics_corpus.models import RoboticsPaperRecord, RoboticsVenue, sanitize_id_component
from pipeline.robotics_corpus.proxy import RoboticsProxyConfig, load_robotics_proxy_from_settings

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# CoRL is published in PMLR. Map calendar year -> volume slug.
# Probed 2026-07-26 via proceedings.mlr.press (2022–2025 present; 2026 TBD).
CORL_PMLR_VOLUME_BY_YEAR: dict[int, str] = {
    2018: "v87",
    2019: "v100",
    2020: "v155",
    2021: "v164",
    2022: "v205",
    2023: "v229",
    2024: "v270",
    2025: "v305",
    # 2026: fill when PMLR volume is published (collect will also try DBLP fallback).
}

# RSS roman edition index: rss01 = 2005 => year = 2004 + N.
# Confirmed 2022=rss18 … 2025=rss21; rss22 (2026) not published as of 2026-07-26.
RSS_BASE_YEAR = 2004
RSS_PROCEEDINGS_ROOT = "https://www.roboticsproceedings.org"

# User bulk window: all three venues, every calendar year 2022–2026 inclusive.
DEFAULT_TARGET_YEARS = (2022, 2023, 2024, 2025, 2026)
DEFAULT_VENUES = (RoboticsVenue.ICRA, RoboticsVenue.CORL, RoboticsVenue.RSS)


def rss_slug_for_year(year: int) -> str:
    """Map calendar year to roboticsproceedings.org directory (rss20 for 2024)."""
    n = year - RSS_BASE_YEAR
    if n < 1:
        raise ValueError(f"RSS year too early: {year}")
    return f"rss{n:02d}"


def ieee_stamp_pdf_url(arnumber: str) -> str:
    return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}&ref="


def ieee_document_url(arnumber: str) -> str:
    return f"https://ieeexplore.ieee.org/document/{arnumber}/"


def extract_ieee_arnumber(url: str) -> Optional[str]:
    """Extract IEEE Xplore article number from document URL, stamp URL, or DOI.

    IEEE conference DOIs commonly end with the arnumber:
    ``10.1109/ICRA57147.2024.10611355`` → ``10611355``.
    """
    text = (url or "").strip()
    m = re.search(r"/document/(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"arnumber=(\d+)", text)
    if m:
        return m.group(1)
    # DOI / bare doi: 10.1109/<conf>.<year>.<arnumber>
    m = re.search(r"10\.1109/[^/\s]+\.(\d{6,})$", text.rstrip("/"))
    if m:
        return m.group(1)
    m = re.search(r"10\.1109/[^/\s]+\.(\d{6,})(?:[/?#]|$)", text)
    if m:
        return m.group(1)
    return None


def build_arxiv_pdf_url(url: str) -> Optional[str]:
    normalized = (url or "").strip()
    if not normalized:
        return None
    if "arxiv.org/abs/" in normalized:
        ident = normalized.rsplit("/abs/", maxsplit=1)[-1]
        return f"https://arxiv.org/pdf/{ident}.pdf"
    if "arxiv.org/pdf/" in normalized:
        return normalized if normalized.endswith(".pdf") else f"{normalized}.pdf"
    return None


class RoboticsHttpClient:
    """Thin httpx wrapper with proxy + bounded retries."""

    def __init__(
        self,
        *,
        proxy: RoboticsProxyConfig | None = None,
        timeout_s: float = 90.0,
        min_interval_s: float = 0.35,
        max_retries: int = 4,
    ):
        self.proxy = proxy or load_robotics_proxy_from_settings()
        self.timeout_s = timeout_s
        self.min_interval_s = max(0.0, min_interval_s)
        self.max_retries = max(1, max_retries)
        self._next_at = 0.0
        self._client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                proxy=self.proxy.as_httpx_proxy(),
                timeout=self.timeout_s,
                follow_redirects=True,
                headers={"User-Agent": DEFAULT_UA, "Accept": "*/*"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "RoboticsHttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = time.monotonic()
        if now < self._next_at:
            time.sleep(self._next_at - now)
        self._next_at = time.monotonic() + self.min_interval_s

    def get_text(self, url: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self._http().get(url)
                # Do not retry permanent client errors (except rate limit).
                if response.status_code in {404, 410}:
                    response.raise_for_status()
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    try:
                        sleep_s = max(60.0, float(retry_after)) if retry_after else 60.0 * attempt
                    except ValueError:
                        sleep_s = 60.0 * attempt
                    last_error = httpx.HTTPStatusError(
                        f"429 Too Many Requests for {url}",
                        request=response.request,
                        response=response,
                    )
                    logger.warning(
                        "DBLP/source rate-limited %s (%s/%s); sleep %.0fs",
                        url, attempt, self.max_retries, sleep_s,
                    )
                    time.sleep(sleep_s)
                    self.close()
                    continue
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as exc:
                last_error = exc
                code = exc.response.status_code if exc.response is not None else None
                if code in {404, 410}:
                    break  # permanent
                if attempt == self.max_retries:
                    break
                sleep_s = min(2 * attempt, 8)
                logger.warning("retry %s (%s/%s): %s", url, attempt, self.max_retries, exc)
                time.sleep(sleep_s)
                self.close()
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                sleep_s = min(2 ** attempt, 30)
                logger.warning("retry %s (%s/%s): %s", url, attempt, self.max_retries, exc)
                time.sleep(sleep_s)
                self.close()
        raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error

    def get_soup(self, url: str) -> BeautifulSoup:
        # lxml is ~10× faster than html.parser on multi‑MB DBLP TOC pages.
        text = self.get_text(url)
        try:
            return BeautifulSoup(text, "lxml")
        except Exception:
            return BeautifulSoup(text, "html.parser")

    def get_json(self, url: str) -> dict[str, Any]:
        return json.loads(self.get_text(url))


def collect_rss_year(year: int, client: RoboticsHttpClient) -> list[RoboticsPaperRecord]:
    """Collect RSS main proceedings from roboticsproceedings.org (fully OA).

    Raises if the year edition page is missing (e.g. 2026 before publication)
    so the prepare loop can record a skipped cell without aborting other years.
    """
    slug = rss_slug_for_year(year)
    list_url = f"{RSS_PROCEEDINGS_ROOT}/{slug}/index.html"
    try:
        soup = client.get_soup(list_url)
    except Exception as exc:
        raise RuntimeError(
            f"RSS {year} proceedings not available at {list_url}: {exc}"
        ) from exc
    papers: dict[str, RoboticsPaperRecord] = {}

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not re.fullmatch(r"p\d+\.html", href, flags=re.I):
            continue
        title = anchor.get_text(" ", strip=True)
        if not title:
            continue
        stem = Path(href).stem.lower()  # p001
        paper_page = urljoin(list_url, href)
        pdf_url = urljoin(list_url, f"{stem}.pdf")
        paper_id = f"rss-{year}-{stem}"
        # DOI pattern when present: 10.15607/RSS.YYYY.ROMAN.NNN
        n = int(stem[1:]) if stem[1:].isdigit() else 0
        roman = _year_to_roman(year)
        doi = f"10.15607/RSS.{year}.{roman}.{n:03d}" if n else None
        papers[paper_id] = RoboticsPaperRecord(
            paper_id=paper_id,
            title=title,
            venue=RoboticsVenue.RSS,
            year=year,
            pdf_url=pdf_url,
            paper_page_url=paper_page,
            doi=doi,
            source_name="roboticsproceedings",
            download_status="pending",
            raw={"list_url": list_url, "slug": slug},
        )
    if not papers:
        raise RuntimeError(f"RSS {year}: no paper entries parsed from {list_url}")
    return list(papers.values())


def _year_to_roman(year: int) -> str:
    """RSS DOI uses Roman numerals for the edition (XX for 2024)."""
    n = year - RSS_BASE_YEAR
    if n <= 0:
        return str(n)
    # Standard roman up to ~40 editions is enough for RSS.
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    remaining = n
    for value, numeral in vals:
        while remaining >= value:
            out.append(numeral)
            remaining -= value
    return "".join(out)


def _collect_corl_from_pmlr_volume(
    year: int, volume: str, client: RoboticsHttpClient,
) -> list[RoboticsPaperRecord]:
    list_url = f"https://proceedings.mlr.press/{volume}/"
    soup = client.get_soup(list_url)
    papers: dict[str, RoboticsPaperRecord] = {}

    for title_node in soup.find_all("p", class_="title"):
        title = title_node.get_text(" ", strip=True)
        container = title_node.parent
        if not title or container is None:
            continue

        page_url = None
        pdf_url = None
        openreview_url = None
        for anchor in container.find_all("a", href=True):
            href = str(anchor["href"])
            if href.endswith(".html") and f"/{volume}/" in href:
                page_url = href
            if href.endswith(".pdf"):
                pdf_url = href
            if "openreview.net" in href:
                openreview_url = href

        if page_url is None or pdf_url is None:
            continue

        slug = Path(page_url).stem
        authors_text = ""
        authors_node = container.find("span", class_="authors")
        if authors_node is not None:
            authors_text = authors_node.get_text(" ", strip=True)
        authors = [name.strip() for name in authors_text.split(",") if name.strip()]

        paper_id = f"corl-{year}-{sanitize_id_component(slug)}"
        alternate: list[str] = []
        if openreview_url and "forum?id=" in openreview_url:
            forum_id = openreview_url.rsplit("id=", 1)[-1]
            alternate.append(f"https://openreview.net/pdf?id={forum_id}")
            alternate.append(f"https://api.openreview.net/attachment?id={forum_id}&name=pdf")

        papers[paper_id] = RoboticsPaperRecord(
            paper_id=paper_id,
            title=title,
            venue=RoboticsVenue.CORL,
            year=year,
            authors=authors,
            pdf_url=pdf_url,
            paper_page_url=page_url,
            alternate_pdf_urls=alternate,
            source_name="pmlr",
            download_status="pending",
            raw={"volume": volume, "list_url": list_url, "openreview": openreview_url},
        )
    return list(papers.values())


def _discover_corl_pmlr_volume_from_dblp(year: int, client: RoboticsHttpClient) -> Optional[str]:
    """When year is not in the static map, discover PMLR volume via DBLP TOC."""
    toc_key = f"conf/corl/corl{year}"
    url = (
        f"https://dblp.org/search/publ/api?"
        f"q=toc:db/{toc_key}.bht:&h=20&f=0&format=json"
    )
    try:
        data = client.get_json(url)
    except Exception as exc:
        logger.warning("CoRL %s DBLP volume discovery failed: %s", year, exc)
        return None
    hits = data.get("result", {}).get("hits", {}).get("hit", []) or []
    if isinstance(hits, dict):
        hits = [hits]
    for hit in hits:
        info = hit.get("info", {}) if isinstance(hit, dict) else {}
        for ee in _normalize_ee(info.get("ee")):
            m = re.search(r"proceedings\.mlr\.press/(v\d+)", ee)
            if m:
                return m.group(1)
    return None


def collect_corl_year(year: int, client: RoboticsHttpClient) -> list[RoboticsPaperRecord]:
    """Collect CoRL main track from PMLR open proceedings (full year)."""
    volume = CORL_PMLR_VOLUME_BY_YEAR.get(year)
    if not volume:
        volume = _discover_corl_pmlr_volume_from_dblp(year, client)
    if not volume:
        raise RuntimeError(
            f"No PMLR volume for CoRL {year} yet "
            f"(static map years={sorted(CORL_PMLR_VOLUME_BY_YEAR)}; DBLP had no mlr link)"
        )
    papers = _collect_corl_from_pmlr_volume(year, volume, client)
    if not papers:
        raise RuntimeError(f"CoRL {year}: PMLR volume {volume} parsed 0 papers")
    return papers


def _parse_dblp_authors(authors_field: Any) -> list[str]:
    if authors_field is None:
        return []
    raw = authors_field.get("author", []) if isinstance(authors_field, dict) else []
    if isinstance(raw, dict):
        raw = [raw]
    out: list[str] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        text = a.get("text") or ""
        if text:
            out.append(text.split(" 0")[0])
    return out


def _normalize_ee(ee: Any) -> list[str]:
    if not ee:
        return []
    if isinstance(ee, list):
        return [str(x) for x in ee if x]
    return [str(ee)]


def _icra_record_from_fields(
    *,
    year: int,
    title: str,
    authors: list[str],
    ee_list: list[str],
    dblp_key: Optional[str],
    toc_key: str,
) -> Optional[RoboticsPaperRecord]:
    title = title.rstrip(".")
    if not title:
        return None
    if (
        title.upper().startswith("IEEE INTERNATIONAL CONFERENCE ON ROBOTICS")
        and "ICRA" in title.upper()
        and len(title) < 120
        and ":" not in title
    ):
        return None

    doi = None
    ieee_url = None
    arxiv_pdf = None
    for ee in ee_list:
        if "doi.org/10.1109/" in ee or ee.startswith("10.1109/"):
            doi = ee.replace("https://doi.org/", "").replace("http://doi.org/", "")
            ieee_url = ee if ee.startswith("http") else f"https://doi.org/{doi}"
        elif "ieeexplore.ieee.org" in ee:
            ieee_url = ee
        elif "arxiv.org" in ee:
            arxiv_pdf = build_arxiv_pdf_url(ee)

    if not doi and not ieee_url and not arxiv_pdf and not ee_list:
        return None

    arnumber = extract_ieee_arnumber(ieee_url or "") or extract_ieee_arnumber(doi or "")
    pdf_url = None
    alternate: list[str] = []
    if arxiv_pdf:
        pdf_url = arxiv_pdf
    if arnumber:
        stamp = ieee_stamp_pdf_url(arnumber)
        if pdf_url is None:
            pdf_url = stamp
        else:
            alternate.append(stamp)
        alternate.append(ieee_document_url(arnumber))
    if doi and not pdf_url:
        pdf_url = f"https://doi.org/{doi}"

    key_suffix = sanitize_id_component(doi or arnumber or (dblp_key or "") or title[:40])
    paper_id = f"icra-{year}-{key_suffix}"
    return RoboticsPaperRecord(
        paper_id=paper_id,
        title=title,
        venue=RoboticsVenue.ICRA,
        year=year,
        authors=authors,
        pdf_url=pdf_url,
        paper_page_url=ieee_url or (ee_list[0] if ee_list else None),
        alternate_pdf_urls=alternate,
        doi=doi,
        source_name="dblp_ieee",
        download_status="pending" if pdf_url else "no_pdf",
        notes=[] if pdf_url else ["no ieee/arxiv pdf candidate on DBLP"],
        raw={
            "dblp_key": dblp_key,
            "ee": ee_list,
            "arnumber": arnumber,
            "toc": toc_key,
        },
    )


def collect_icra_year_from_html(
    year: int, client: RoboticsHttpClient, *, max_papers: int = 0,
) -> list[RoboticsPaperRecord]:
    """Single-request HTML TOC collect (avoids multi-page DBLP API rate limits)."""
    toc_key = f"conf/icra/icra{year}"
    url = f"https://dblp.org/db/{toc_key}.html"
    soup = client.get_soup(url)
    papers: dict[str, RoboticsPaperRecord] = {}
    for li in soup.select("li.entry"):
        title_el = li.select_one("span.title")
        title = title_el.get_text(" ", strip=True) if title_el else ""
        authors = [
            a.get_text(" ", strip=True)
            for a in li.select("span[itemprop='author'] span[itemprop='name']")
        ]
        if not authors:
            authors = [
                a.get_text(" ", strip=True)
                for a in li.select("span.author")
            ]
        ee_list: list[str] = []
        for a in li.find_all("a", href=True):
            href = str(a["href"])
            if any(k in href for k in ("doi.org", "ieeexplore", "arxiv.org")):
                ee_list.append(href)
        # DBLP key from data-pid or publ URL
        dblp_key = None
        for a in li.find_all("a", href=True):
            href = str(a["href"])
            if "/rec/conf/icra/" in href:
                dblp_key = href.rstrip(".html").split("/rec/")[-1]
                break
        rec = _icra_record_from_fields(
            year=year,
            title=title,
            authors=authors,
            ee_list=ee_list,
            dblp_key=dblp_key,
            toc_key=toc_key,
        )
        if rec is None:
            continue
        papers[rec.paper_id] = rec
        if max_papers > 0 and len(papers) >= max_papers:
            break
    if not papers:
        raise RuntimeError(f"ICRA {year}: HTML TOC {url} parsed 0 papers")
    return list(papers.values())


def collect_icra_year_from_json(
    year: int,
    client: RoboticsHttpClient,
    *,
    max_papers: int = 0,
) -> list[RoboticsPaperRecord]:
    """Paginated DBLP JSON API fallback.

    DBLP hard-caps ``@sent`` at **100** per request regardless of ``h``, so a
    1761-paper year needs ~18 round-trips (~4s each through the proxy). Prefer
    :func:`collect_icra_year_from_html` (one 15MB page + lxml ≈ 8s) for bulk.
    """
    toc_key = f"conf/icra/icra{year}"
    batch = 100  # DBLP API max page size in practice
    offset = 0
    papers: dict[str, RoboticsPaperRecord] = {}
    total = None

    while True:
        q = f"toc:db/{toc_key}.bht:"
        url = (
            f"https://dblp.org/search/publ/api?"
            f"q={q}&h={batch}&f={offset}&format=json"
        )
        data = client.get_json(url)
        hits_obj = data.get("result", {}).get("hits", {})
        total = int(hits_obj.get("@total", 0) or 0)
        hit_list = hits_obj.get("hit", []) or []
        if isinstance(hit_list, dict):
            hit_list = [hit_list]
        if total == 0 and offset == 0:
            raise RuntimeError(
                f"ICRA {year}: DBLP TOC empty for {toc_key} "
                f"(edition may not be indexed yet)"
            )

        for hit in hit_list:
            info = hit.get("info", {}) if isinstance(hit, dict) else {}
            rec = _icra_record_from_fields(
                year=year,
                title=str(info.get("title") or ""),
                authors=_parse_dblp_authors(info.get("authors")),
                ee_list=_normalize_ee(info.get("ee")),
                dblp_key=str(info.get("key") or "") or None,
                toc_key=toc_key,
            )
            if rec is None:
                continue
            papers[rec.paper_id] = rec
            if max_papers > 0 and len(papers) >= max_papers:
                return list(papers.values())

        offset += len(hit_list)
        if not hit_list or offset >= total:
            break

    if not papers:
        raise RuntimeError(f"ICRA {year}: parsed 0 papers from DBLP JSON (total={total})")
    return list(papers.values())


def collect_icra_year(
    year: int,
    client: RoboticsHttpClient,
    *,
    max_papers: int = 0,
) -> list[RoboticsPaperRecord]:
    """Collect full ICRA year catalog from DBLP.

    Prefers **HTML TOC + lxml** (one request, ~8s for 1760 papers). Falls back
    to paginated JSON (100/page) only if HTML fails. ``max_papers`` is smoke-only.
    """
    try:
        return collect_icra_year_from_html(year, client, max_papers=max_papers)
    except Exception as html_exc:
        logger.warning("ICRA %s HTML TOC failed (%s); falling back to JSON API", year, html_exc)
    return collect_icra_year_from_json(year, client, max_papers=max_papers)


def collect_venue_year(
    venue: RoboticsVenue,
    year: int,
    client: RoboticsHttpClient,
    *,
    max_papers: int = 0,
) -> list[RoboticsPaperRecord]:
    """Collect one venue×year cell. ``max_papers=0`` means the full year."""
    if venue is RoboticsVenue.RSS:
        papers = collect_rss_year(year, client)
    elif venue is RoboticsVenue.CORL:
        papers = collect_corl_year(year, client)
    elif venue is RoboticsVenue.ICRA:
        papers = collect_icra_year(year, client, max_papers=max_papers)
    else:
        raise ValueError(f"unsupported venue {venue}")
    if max_papers > 0 and venue is not RoboticsVenue.ICRA:
        papers = papers[:max_papers]
    return papers


def supported_years(venue: RoboticsVenue) -> Iterable[int]:
    if venue is RoboticsVenue.CORL:
        return sorted(CORL_PMLR_VOLUME_BY_YEAR)
    if venue is RoboticsVenue.RSS:
        # rss01 (2005) .. rss21 (2025) confirmed; keep wide but practical default later.
        return list(range(2005, 2026))
    if venue is RoboticsVenue.ICRA:
        return list(range(2015, 2026))
    return []
