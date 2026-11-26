"""Canonical medical paper identity and manifest records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class MedicalSourceKind(str, Enum):
    """Provenance of a medical record in the preparation pipeline."""

    EUROPE_PMC = "europe_pmc"
    PUBMED = "pubmed"
    PREPRINT = "preprint"
    JOURNAL_OA = "journal_oa"
    GUIDELINE = "guideline"
    OTHER = "other"


def sanitize_id_component(value: str) -> str:
    """Turn a paper id into a portable path/filename stem."""
    cleaned = []
    for ch in (value or "").strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    stem = "".join(cleaned).strip("._") or "paper"
    return stem[:180]


def canonical_medical_id(
    *,
    pmcid: Optional[str] = None,
    pmid: Optional[str] = None,
    doi: Optional[str] = None,
    source_kind: MedicalSourceKind = MedicalSourceKind.EUROPE_PMC,
) -> str:
    """Prefer PMCID, then PMID, then DOI as the stable medical paper_id."""
    if pmcid:
        token = pmcid.strip().upper()
        if not token.startswith("PMC"):
            token = f"PMC{token}"
        return f"med-{token.lower()}"
    if pmid:
        return f"med-pmid-{str(pmid).strip()}"
    if doi:
        safe = sanitize_id_component(doi.strip().lower().replace("/", "_"))
        return f"med-doi-{safe}"
    raise ValueError(f"cannot build medical paper_id without pmcid/pmid/doi ({source_kind})")


@dataclass(slots=True)
class MedicalPaperRecord:
    """One medical paper row for manifest / dry-run / later download."""

    paper_id: str
    title: str
    source_kind: MedicalSourceKind = MedicalSourceKind.EUROPE_PMC
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    license: Optional[str] = None
    is_open_access: bool = False
    has_pdf: bool = False
    pdf_url: Optional[str] = None
    html_url: Optional[str] = None
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    pdf_path: Optional[str] = None
    pdf_sha256: Optional[str] = None
    pdf_bytes: Optional[int] = None
    download_status: str = "pending"  # pending|planned|downloaded|skipped|failed|blocked
    download_error: Optional[str] = None
    shield_status: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_kind"] = self.source_kind.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MedicalPaperRecord":
        kind = data.get("source_kind") or MedicalSourceKind.EUROPE_PMC.value
        return cls(
            paper_id=str(data["paper_id"]),
            title=str(data.get("title") or ""),
            source_kind=MedicalSourceKind(kind),
            pmid=data.get("pmid"),
            pmcid=data.get("pmcid"),
            doi=data.get("doi"),
            journal=data.get("journal"),
            year=data.get("year"),
            license=data.get("license"),
            is_open_access=bool(data.get("is_open_access")),
            has_pdf=bool(data.get("has_pdf")),
            pdf_url=data.get("pdf_url"),
            html_url=data.get("html_url"),
            abstract=str(data.get("abstract") or ""),
            authors=list(data.get("authors") or []),
            pdf_path=data.get("pdf_path"),
            pdf_sha256=data.get("pdf_sha256"),
            pdf_bytes=data.get("pdf_bytes"),
            download_status=str(data.get("download_status") or "pending"),
            download_error=data.get("download_error"),
            shield_status=data.get("shield_status"),
            notes=list(data.get("notes") or []),
            raw=dict(data.get("raw") or {}),
        )


def europe_pmc_pdf_url(pmcid: str) -> str:
    token = pmcid.strip().upper()
    if not token.startswith("PMC"):
        token = f"PMC{token}"
    return f"https://europepmc.org/articles/{token}?pdf=render"


def europe_pmc_html_url(pmcid: str) -> str:
    token = pmcid.strip().upper()
    if not token.startswith("PMC"):
        token = f"PMC{token}"
    return f"https://europepmc.org/articles/{token}"
