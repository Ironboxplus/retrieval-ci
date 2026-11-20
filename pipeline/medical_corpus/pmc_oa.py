"""PMC Open Access articles in NCBI's public S3 bucket (AWS Open Data).

Every article of the PMC OA and author-manuscript datasets sits under
``<PMCID>.<version>/`` with ``<PMCID>.<version>.pdf`` and the JATS full text
``<PMCID>.<version>.xml`` beside it. The bucket is the bulk-download channel NCBI
publishes for exactly this use, has no Cloudflare in front of it (Europe PMC's
``?pdf=render`` answers datacenter IPs with a Cloudflare 403), and gives the JATS
XML whose MathML keeps the formulas a PDF converter would have to guess at.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from pipeline.medical_corpus.proxy import MedicalProxyConfig
from pipeline.paper_crawler.runtime_files import build_write_in_progress_path

PMC_OA_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
_KEY_RE = re.compile(r"<Key>([^<]+)</Key>")


@dataclass(frozen=True, slots=True)
class PmcOaArticle:
    pmcid: str
    version: int
    pdf_url: str
    xml_url: Optional[str]


def normalize_pmcid(pmcid: str) -> str:
    token = pmcid.strip().upper()
    return token if token.startswith("PMC") else f"PMC{token}"


def resolve_pmc_oa(
    pmcid: str,
    *,
    proxy: MedicalProxyConfig,
    base_url: str = PMC_OA_BUCKET_URL,
    timeout_s: float = 30.0,
) -> Optional[PmcOaArticle]:
    """The newest version of ``pmcid`` that has a PDF in the bucket, or None.

    One ListObjectsV2 request on the prefix ``<PMCID>.``; an article has a
    handful of versions at most, far below a page.
    """
    token = normalize_pmcid(pmcid)
    with httpx.Client(proxy=proxy.as_httpx_proxy(), timeout=timeout_s) as client:
        resp = client.get(f"{base_url}/", params={"list-type": "2", "prefix": f"{token}."})
    resp.raise_for_status()
    keys = set(_KEY_RE.findall(resp.text))
    versions: dict[int, set[str]] = {}
    for key in keys:
        folder, _, name = key.partition("/")
        if not folder.startswith(f"{token}.") or not folder[len(token) + 1:].isdigit():
            continue
        versions.setdefault(int(folder[len(token) + 1:]), set()).add(name)
    for version in sorted(versions, reverse=True):
        stem = f"{token}.{version}"
        if f"{stem}.pdf" in versions[version]:
            xml = f"{base_url}/{stem}/{stem}.xml" if f"{stem}.xml" in versions[version] else None
            return PmcOaArticle(token, version, f"{base_url}/{stem}/{stem}.pdf", xml)
    return None


def jats_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(".xml")


def fetch_jats(
    article: PmcOaArticle,
    dest: Path,
    *,
    proxy: MedicalProxyConfig,
    timeout_s: float = 120.0,
) -> tuple[str, int]:
    """Write the article's JATS XML to ``dest``; returns (sha256, bytes).

    Written through a temp file and renamed, so a file at ``dest`` is always a
    complete document that starts as XML and holds an ``<article`` element.
    """
    if not article.xml_url:
        raise FileNotFoundError(f"{article.pmcid}.{article.version}: no JATS XML in the bucket")
    with httpx.Client(proxy=proxy.as_httpx_proxy(), timeout=timeout_s) as client:
        resp = client.get(article.xml_url)
    resp.raise_for_status()
    body = resp.content
    head = body[:4096].lstrip()
    if not head.startswith(b"<") or b"<article" not in body[:65536]:
        raise ValueError(f"{article.xml_url}: not a JATS article document")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = build_write_in_progress_path(dest)
    try:
        tmp.write_bytes(body)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return hashlib.sha256(body).hexdigest(), len(body)
