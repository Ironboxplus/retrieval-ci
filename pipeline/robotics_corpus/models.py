"""Canonical robotics paper identity and manifest records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class RoboticsVenue(str, Enum):
    ICRA = "ICRA"
    CORL = "CoRL"
    RSS = "RSS"


def sanitize_id_component(value: str) -> str:
    cleaned = []
    for ch in (value or "").strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    stem = "".join(cleaned).strip("._") or "paper"
    return stem[:180]


@dataclass(slots=True)
class RoboticsPaperRecord:
    """One robotics conference paper row for manifest / dry-run / download."""

    paper_id: str
    title: str
    venue: RoboticsVenue
    year: int
    authors: list[str] = field(default_factory=list)
    pdf_url: Optional[str] = None
    paper_page_url: Optional[str] = None
    alternate_pdf_urls: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    abstract: str = ""
    source_name: str = ""
    pdf_path: Optional[str] = None
    pdf_sha256: Optional[str] = None
    pdf_bytes: Optional[int] = None
    download_status: str = "pending"  # pending|planned|downloaded|skipped|failed|blocked|no_pdf
    download_error: Optional[str] = None
    shield_status: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["venue"] = self.venue.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoboticsPaperRecord":
        venue_raw = data.get("venue") or RoboticsVenue.ICRA.value
        return cls(
            paper_id=str(data["paper_id"]),
            title=str(data.get("title") or ""),
            venue=RoboticsVenue(venue_raw),
            year=int(data["year"]),
            authors=list(data.get("authors") or []),
            pdf_url=data.get("pdf_url"),
            paper_page_url=data.get("paper_page_url"),
            alternate_pdf_urls=list(data.get("alternate_pdf_urls") or []),
            doi=data.get("doi"),
            abstract=str(data.get("abstract") or ""),
            source_name=str(data.get("source_name") or ""),
            pdf_path=data.get("pdf_path"),
            pdf_sha256=data.get("pdf_sha256"),
            pdf_bytes=data.get("pdf_bytes"),
            download_status=str(data.get("download_status") or "pending"),
            download_error=data.get("download_error"),
            shield_status=data.get("shield_status"),
            notes=list(data.get("notes") or []),
            raw=dict(data.get("raw") or {}),
        )
