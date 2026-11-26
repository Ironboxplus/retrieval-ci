"""Top medical journal whitelist for corpus prep.

Only records whose journal title matches this set are kept. The list is the
clinical / biomedical "top tier" commonly treated as flagship venues (NEJM,
Lancet family, JAMA family, BMJ, Nature Medicine / Nature, Science, Cell,
Annals of Internal Medicine, plus a small set of high-IF specialty flagships).

Matching is case-insensitive and tolerant of common EuropePMC title variants
(e.g. "The Lancet", "N Engl J Med", "BMJ (Clinical research ed.)").
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

# Canonical display names (also used in docs / summary).
TOP_MEDICAL_JOURNALS: tuple[str, ...] = (
    # General medicine flagships
    "The New England Journal of Medicine",
    "The Lancet",
    "JAMA",
    "BMJ",
    "Annals of Internal Medicine",
    "Nature Medicine",
    "Nature",
    "Science",
    "Cell",
    "The Lancet Oncology",
    "The Lancet Neurology",
    "The Lancet Infectious Diseases",
    "The Lancet Digital Health",
    "JAMA Internal Medicine",
    "JAMA Oncology",
    "JAMA Neurology",
    "JAMA Network Open",
    # High-IF clinical specialty / methods flagships
    "Circulation",
    "European Heart Journal",
    "Journal of the American College of Cardiology",
    "Blood",
    "Journal of Clinical Oncology",
    "Gastroenterology",
    "Hepatology",
    "Gut",
    "The Lancet Respiratory Medicine",
    "The Lancet Psychiatry",
    "Nature Reviews Drug Discovery",
    "Nature Reviews Immunology",
    "Nature Reviews Cancer",
    "Nature Biotechnology",
    "Nature Genetics",
    "Cancer Discovery",
    "Immunity",
    "Journal of Experimental Medicine",
    "PLOS Medicine",
    "eLife",
    "The BMJ",
)

# Extra aliases / abbreviations that EuropePMC and PubMed often use.
_JOURNAL_ALIASES: dict[str, str] = {
    # NEJM
    "n engl j med": "The New England Journal of Medicine",
    "new england journal of medicine": "The New England Journal of Medicine",
    "the new england journal of medicine": "The New England Journal of Medicine",
    "nejm": "The New England Journal of Medicine",
    # Lancet family
    "lancet": "The Lancet",
    "the lancet": "The Lancet",
    "lancet oncol": "The Lancet Oncology",
    "the lancet oncology": "The Lancet Oncology",
    "lancet neurol": "The Lancet Neurology",
    "the lancet neurology": "The Lancet Neurology",
    "lancet infect dis": "The Lancet Infectious Diseases",
    "the lancet infectious diseases": "The Lancet Infectious Diseases",
    "lancet digit health": "The Lancet Digital Health",
    "the lancet digital health": "The Lancet Digital Health",
    "lancet respir med": "The Lancet Respiratory Medicine",
    "the lancet respiratory medicine": "The Lancet Respiratory Medicine",
    "lancet psychiatry": "The Lancet Psychiatry",
    "the lancet psychiatry": "The Lancet Psychiatry",
    # JAMA family
    "jama": "JAMA",
    "jama-journal of the american medical association": "JAMA",
    "journal of the american medical association": "JAMA",
    "jama intern med": "JAMA Internal Medicine",
    "jama internal medicine": "JAMA Internal Medicine",
    "jama oncol": "JAMA Oncology",
    "jama oncology": "JAMA Oncology",
    "jama neurol": "JAMA Neurology",
    "jama neurology": "JAMA Neurology",
    "jama netw open": "JAMA Network Open",
    "jama network open": "JAMA Network Open",
    # BMJ
    "bmj": "BMJ",
    "the bmj": "BMJ",
    "br med j": "BMJ",
    "british medical journal": "BMJ",
    "bmj (clinical research ed.)": "BMJ",
    "bmj (clinical research ed)": "BMJ",
    # Annals
    "ann intern med": "Annals of Internal Medicine",
    "annals of internal medicine": "Annals of Internal Medicine",
    # Nature / Science / Cell
    "nat med": "Nature Medicine",
    "nature medicine": "Nature Medicine",
    "nature": "Nature",
    "science": "Science",
    "science (new york, n.y.)": "Science",
    "cell": "Cell",
    "nat rev drug discov": "Nature Reviews Drug Discovery",
    "nature reviews drug discovery": "Nature Reviews Drug Discovery",
    "nat rev immunol": "Nature Reviews Immunology",
    "nature reviews immunology": "Nature Reviews Immunology",
    "nat rev cancer": "Nature Reviews Cancer",
    "nature reviews cancer": "Nature Reviews Cancer",
    "nat biotechnol": "Nature Biotechnology",
    "nature biotechnology": "Nature Biotechnology",
    "nat genet": "Nature Genetics",
    "nature genetics": "Nature Genetics",
    # Specialty
    "circulation": "Circulation",
    "eur heart j": "European Heart Journal",
    "european heart journal": "European Heart Journal",
    "j am coll cardiol": "Journal of the American College of Cardiology",
    "journal of the american college of cardiology": "Journal of the American College of Cardiology",
    "jacc": "Journal of the American College of Cardiology",
    "blood": "Blood",
    "j clin oncol": "Journal of Clinical Oncology",
    "journal of clinical oncology": "Journal of Clinical Oncology",
    "gastroenterology": "Gastroenterology",
    "hepatology": "Hepatology",
    "hepatology (baltimore, md.)": "Hepatology",
    "gut": "Gut",
    "cancer discov": "Cancer Discovery",
    "cancer discovery": "Cancer Discovery",
    "immunity": "Immunity",
    "j exp med": "Journal of Experimental Medicine",
    "journal of experimental medicine": "Journal of Experimental Medicine",
    "plos med": "PLOS Medicine",
    "plos medicine": "PLOS Medicine",
    "elife": "eLife",
}


def normalize_journal_key(name: str) -> str:
    """Lowercase, strip diacritics / punctuation noise for matching."""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    # Drop trailing publisher noise in parentheses for key lookup after alias map.
    return text


def _build_whitelist_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    for canonical in TOP_MEDICAL_JOURNALS:
        keys[normalize_journal_key(canonical)] = canonical
    for alias, canonical in _JOURNAL_ALIASES.items():
        keys[normalize_journal_key(alias)] = canonical
    return keys


_WHITELIST_KEYS = _build_whitelist_keys()


def match_top_journal(journal: Optional[str]) -> Optional[str]:
    """Return canonical top-journal name if ``journal`` is on the whitelist."""
    if not journal or not str(journal).strip():
        return None
    key = normalize_journal_key(str(journal))
    if key in _WHITELIST_KEYS:
        return _WHITELIST_KEYS[key]
    # Soft match: strip parenthetical edition notes then retry.
    stripped = re.sub(r"\s*\([^)]*\)\s*", " ", key).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    if stripped in _WHITELIST_KEYS:
        return _WHITELIST_KEYS[stripped]
    # Prefix match for long EuropePMC titles that append subtitle.
    for known, canonical in _WHITELIST_KEYS.items():
        if len(known) >= 8 and (key.startswith(known + " ") or key.startswith(known + ":")):
            return canonical
        if len(known) >= 8 and (known.startswith(key + " ") or known.startswith(key + ":")):
            return canonical
    return None


def is_top_medical_journal(journal: Optional[str]) -> bool:
    return match_top_journal(journal) is not None


def filter_top_journal_records(records: Iterable) -> tuple[list, dict[str, int]]:
    """Keep only top-journal records; return (kept, stats).

    ``records`` items must have a ``.journal`` attribute (MedicalPaperRecord).
    """
    kept = []
    dropped_no_journal = 0
    dropped_not_top = 0
    by_journal: dict[str, int] = {}
    for rec in records:
        journal = getattr(rec, "journal", None)
        canonical = match_top_journal(journal)
        if canonical is None:
            if not journal:
                dropped_no_journal += 1
            else:
                dropped_not_top += 1
            continue
        # Normalize stored journal name to canonical for cleaner manifests.
        try:
            rec.journal = canonical
        except Exception:
            pass
        kept.append(rec)
        by_journal[canonical] = by_journal.get(canonical, 0) + 1
    stats = {
        "kept": len(kept),
        "dropped_no_journal": dropped_no_journal,
        "dropped_not_top": dropped_not_top,
        "by_journal": by_journal,
        "whitelist_size": len(TOP_MEDICAL_JOURNALS),
    }
    return kept, stats


def europepmc_journal_clause(journals: Iterable[str] | None = None) -> str:
    """Build an EuropePMC OR-clause over journal titles for server-side prefilter.

    EuropePMC ``JOURNAL:\"...\"`` is approximate; client-side
    :func:`filter_top_journal_records` remains the authority.
    """
    names = list(journals) if journals is not None else list(TOP_MEDICAL_JOURNALS)
    # Keep query size bounded — use the highest-yield general flagships first.
    # Full whitelist is still applied client-side.
    priority = [
        "The New England Journal of Medicine",
        "Lancet",
        "JAMA",
        "BMJ",
        "Nature Medicine",
        "Annals of Internal Medicine",
        "Nature",
        "Science",
        "Cell",
        "Circulation",
        "Blood",
        "PLOS Medicine",
    ]
    ordered: list[str] = []
    for name in priority + names:
        if name not in ordered:
            ordered.append(name)
    # EuropePMC query length is limited; take first 20 for the API filter.
    parts = [f'JOURNAL:"{name}"' for name in ordered[:20]]
    return "(" + " OR ".join(parts) + ")"


def default_top_journal_query(
    *,
    year_from: int = 2020,
    year_to: int = 2026,
    open_access: bool = True,
    has_pdf: bool = True,
) -> str:
    """Default collect query: Medline OA+PDF + top-journal OR-clause + year window."""
    bits = ["SRC:MED"]
    if open_access:
        bits.append("OPEN_ACCESS:Y")
    if has_pdf:
        bits.append("HAS_PDF:Y")
    bits.append(f"(PUB_YEAR:[{year_from} TO {year_to}])")
    bits.append(europepmc_journal_clause())
    return " AND ".join(bits)
