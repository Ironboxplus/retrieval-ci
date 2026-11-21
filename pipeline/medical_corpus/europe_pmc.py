"""EuropePMC REST client — primary medical OA catalog (proxy-mandatory)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional
from urllib.parse import urlencode

import httpx
from tenacity import RetryCallState, RetryError, Retrying, stop_after_attempt

from pipeline.http_retry import DOWNLOAD_BACKOFF
from pipeline.medical_corpus.models import (
    MedicalPaperRecord,
    MedicalSourceKind,
    canonical_medical_id,
    europe_pmc_html_url,
    europe_pmc_pdf_url,
)
from pipeline.medical_corpus.proxy import MedicalProxyConfig, load_medical_proxy_from_settings
from pipeline.medical_corpus.shield import classify_http_shield, retry_delay_seconds

logger = logging.getLogger(__name__)

EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Default bulk query: Medline OA with PDF flag (measured ~7.3M / ~4.7M for 2020-2026).
DEFAULT_BULK_QUERY = "OPEN_ACCESS:Y AND HAS_PDF:Y AND SRC:MED"
DEFAULT_RECENT_QUERY = (
    "OPEN_ACCESS:Y AND HAS_PDF:Y AND SRC:MED AND (PUB_YEAR:[2020 TO 2026])"
)

DEFAULT_UA = (
    "retrieval-medical/0.1 (mailto:arc-retrieval@example.com; research corpus prep)"
)


class _RetryableShield(Exception):
    """A response the shield classifier says to retry; carries its own delay hint."""

    def __init__(self, decision) -> None:
        super().__init__(decision.message)
        self.decision = decision


def _shield_aware_wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, _RetryableShield):
        return retry_delay_seconds(exc.decision, state.attempt_number)
    return DOWNLOAD_BACKOFF(state)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, _RetryableShield):
        logger.warning(
            "europe_pmc retryable shield=%s attempt=%s delay=%.1fs",
            exc.decision.kind.value, state.attempt_number, state.next_action.sleep,
        )
    else:
        logger.warning("europe_pmc search failed attempt=%s err=%s", state.attempt_number, exc)


@dataclass(slots=True)
class EuropePMCSearchPage:
    hit_count: int
    next_cursor_mark: Optional[str]
    results: list[dict[str, Any]]
    raw: dict[str, Any]


def _pick_pdf_url(result: dict[str, Any]) -> Optional[str]:
    urls = (result.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    if isinstance(urls, dict):
        urls = [urls]
    for item in urls:
        if not isinstance(item, dict):
            continue
        style = str(item.get("documentStyle") or "").lower()
        avail = str(item.get("availabilityCode") or item.get("availability") or "").upper()
        url = item.get("url")
        if style == "pdf" and url and ("OA" in avail or "open" in avail.lower()):
            return str(url)
    pmcid = result.get("pmcid")
    if pmcid:
        return europe_pmc_pdf_url(str(pmcid))
    return None


def _pick_html_url(result: dict[str, Any]) -> Optional[str]:
    urls = (result.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    if isinstance(urls, dict):
        urls = [urls]
    for item in urls:
        if not isinstance(item, dict):
            continue
        style = str(item.get("documentStyle") or "").lower()
        url = item.get("url")
        if style in {"html", "doi"} and url:
            if style == "html":
                return str(url)
    pmcid = result.get("pmcid")
    if pmcid:
        return europe_pmc_html_url(str(pmcid))
    return None


def _authors(result: dict[str, Any]) -> list[str]:
    raw = (result.get("authorList") or {}).get("author") or []
    if isinstance(raw, dict):
        raw = [raw]
    out: list[str] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        name = a.get("fullName") or " ".join(
            x for x in [a.get("firstName"), a.get("lastName")] if x
        )
        if name:
            out.append(str(name))
    if not out and result.get("authorString"):
        out = [p.strip() for p in str(result["authorString"]).split(",") if p.strip()]
    return out


def record_from_europe_pmc_result(result: dict[str, Any]) -> MedicalPaperRecord:
    pmcid = result.get("pmcid")
    pmid = result.get("pmid")
    doi = result.get("doi")
    year_raw = result.get("pubYear")
    year = int(year_raw) if year_raw and str(year_raw).isdigit() else None
    journal = None
    info = result.get("journalInfo") or {}
    if isinstance(info, dict):
        journal = (info.get("journal") or {}).get("title") if isinstance(info.get("journal"), dict) else info.get("journal")
        journal = journal or result.get("journalTitle")
    paper_id = canonical_medical_id(
        pmcid=str(pmcid) if pmcid else None,
        pmid=str(pmid) if pmid else None,
        doi=str(doi) if doi else None,
        source_kind=MedicalSourceKind.EUROPE_PMC,
    )
    pdf_url = _pick_pdf_url(result)
    return MedicalPaperRecord(
        paper_id=paper_id,
        title=str(result.get("title") or "").strip(),
        source_kind=MedicalSourceKind.EUROPE_PMC,
        pmid=str(pmid) if pmid else None,
        pmcid=str(pmcid).upper() if pmcid else None,
        doi=str(doi) if doi else None,
        journal=str(journal) if journal else None,
        year=year,
        license=str(result.get("license") or "") or None,
        is_open_access=str(result.get("isOpenAccess") or "").upper() in {"Y", "YES", "TRUE", "1"},
        has_pdf=str(result.get("hasPDF") or "").upper() in {"Y", "YES", "TRUE", "1"} or bool(pdf_url),
        pdf_url=pdf_url,
        html_url=_pick_html_url(result),
        abstract=str(result.get("abstractText") or ""),
        authors=_authors(result),
        download_status="pending",
        raw={"europe_pmc_id": result.get("id"), "source": result.get("source")},
    )


class EuropePMCClient:
    """Cursor-based EuropePMC search client (always via local proxy)."""

    def __init__(
        self,
        *,
        proxy: MedicalProxyConfig | None = None,
        timeout_s: float = 60.0,
        min_interval_s: float = 0.35,
        user_agent: str = DEFAULT_UA,
        max_retries: int = 4,
    ):
        self.proxy = proxy or load_medical_proxy_from_settings()
        self.timeout_s = timeout_s
        self.min_interval_s = max(0.0, min_interval_s)
        self.user_agent = user_agent
        self.max_retries = max(1, max_retries)
        self._next_at = 0.0
        self._client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                proxy=self.proxy.as_httpx_proxy(),
                timeout=self.timeout_s,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "EuropePMCClient":
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

    def search_page(
        self,
        query: str,
        *,
        page_size: int = 100,
        cursor_mark: str = "*",
        result_type: str = "core",
    ) -> EuropePMCSearchPage:
        params = {
            "query": query,
            "format": "json",
            "resultType": result_type,
            "pageSize": max(1, min(int(page_size), 1000)),
            "cursorMark": cursor_mark,
        }
        url = f"{EUROPE_PMC_SEARCH}?{urlencode(params)}"

        def fetch_page() -> EuropePMCSearchPage:
            self._throttle()
            resp = self._http().get(url)
            decision = classify_http_shield(
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type"),
                body=resp.content[:4000],
                headers=resp.headers,
            )
            if decision.kind.value != "none" and decision.retryable:
                raise _RetryableShield(decision)
            if resp.status_code >= 400:
                resp.raise_for_status()
            payload = resp.json()
            results = (payload.get("resultList") or {}).get("result") or []
            if isinstance(results, dict):
                results = [results]
            return EuropePMCSearchPage(
                hit_count=int(payload.get("hitCount") or 0),
                next_cursor_mark=payload.get("nextCursorMark"),
                results=list(results),
                raw=payload,
            )

        try:
            return Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=_shield_aware_wait,
                before_sleep=_log_retry,
            )(fetch_page)
        except RetryError as exc:
            raise RuntimeError(
                f"EuropePMC search failed after retries: {exc.last_attempt.exception()}"
            ) from exc

    def iter_results(
        self,
        query: str,
        *,
        page_size: int = 100,
        max_records: int | None = None,
        result_type: str = "core",
    ) -> Iterator[dict[str, Any]]:
        cursor = "*"
        yielded = 0
        seen_cursors: set[str] = set()
        while True:
            page = self.search_page(
                query, page_size=page_size, cursor_mark=cursor, result_type=result_type,
            )
            if not page.results:
                break
            for row in page.results:
                yield row
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return
            nxt = page.next_cursor_mark
            if not nxt or nxt == cursor or nxt in seen_cursors:
                break
            seen_cursors.add(cursor)
            cursor = nxt

    def iter_records(
        self,
        query: str = DEFAULT_RECENT_QUERY,
        *,
        page_size: int = 100,
        max_records: int | None = None,
    ) -> Iterator[MedicalPaperRecord]:
        for row in self.iter_results(query, page_size=page_size, max_records=max_records):
            try:
                yield record_from_europe_pmc_result(row)
            except ValueError as exc:
                logger.warning("skip europe_pmc row without id: %s", exc)
