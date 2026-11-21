"""Classify a proceedings entry into a publication *track* from its DBLP section
heading (and, for flat TOCs, its title prefix).

Why this exists
---------------
``DblpAcmProceedingsSource`` historically pulled an entire DBLP TOC and kept every
``10.1145/`` entry, so workshop / tutorial / demo / poster / doctoral-consortium /
keynote / late-breaking entries that share the main proceedings' DOI prefix were
ingested as if they were main-conference research papers (project memory: the
"section-blind" contamination, ~37% non-main for SIGIR). We do **not** delete those
rows; instead every paper is *tagged* with its real track so retrieval can flag
"not a main-conference paper (low value)".

Design
------
We recognise only the **closed vocabulary of non-research tracks**; anything we do
not recognise defaults to ``research`` (keep-by-default -- never silently drop, the
W89 lesson). Section heading is the primary signal; for venues whose DBLP TOC is
flat (DAC/EuroSys) we fall back to a colon-terminated title prefix
("POSTER:", "Late Breaking Results:", "Invited:") which is safe against false hits
like "Posterior ...".

``track_type`` values
---------------------
main (``is_main``):       research, short, industry
non-main (low value):     workshop, tutorial, doctoral, demo, poster, keynote,
                          invited, panel, late-breaking, wip, award, front-matter,
                          abstract, other
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAIN_TRACKS = frozenset({"research", "short", "industry"})


@dataclass(frozen=True)
class TrackVerdict:
    track_type: str

    @property
    def is_main(self) -> bool:
        return self.track_type in MAIN_TRACKS


# Section-heading rules, evaluated in order; first match wins. Non-research tracks
# are matched FIRST so a junk section never falls through to a main label. The two
# main "special" tracks (short / industry) are matched last, before the research
# default. Patterns run against a lower-cased section name.
_SECTION_RULES: list[tuple[str, str]] = [
    (r"work\s*shop", "workshop"),
    (r"tutorial", "tutorial"),
    (r"doctoral|consortium|\bph\.?\s?d\b", "doctoral"),
    (r"keynote|plenary", "keynote"),
    (r"\binvited\b", "invited"),
    (r"\bpanel", "panel"),
    (r"demo(?:nstration)?s?\b", "demo"),
    (r"poster", "poster"),
    (r"late[\s\-]*breaking|\blbr\b", "late-breaking"),
    (r"work[\s\-]*in[\s\-]*progress|\bwip\b", "wip"),
    (r"\baward\b|test of time", "award"),
    (r"reception|lightning|town\s*hall|birds.*feather|\bbof\b|exhibit|sponsor", "other"),
    (r"opening|closing|welcome|\bremarks\b|ceremony", "other"),
    (r"front\s*matter|table of contents|preface|foreword", "front-matter"),
    (r"extended abstract|^\s*abstracts?\s*$", "abstract"),
    # --- main "special" tracks ---
    (r"\bshort\s+(?:research\s+)?(?:paper|track|session|contribution)", "short"),
    (r"\b(?:industry|industrial)\s*(?:track|session|paper)|applied\s*data|\bsirip\b", "industry"),
]
_COMPILED_SECTION = [(re.compile(p, re.I), tt) for p, tt in _SECTION_RULES]

# Flat-TOC fallback: colon-terminated title prefixes. The trailing ':' is mandatory
# so "Posterior ...", "Demonstrating ...", "Invited Reviewers ..." stay research.
_TITLE_RULES: list[tuple[str, str]] = [
    (r"^\s*poster\b\s*:", "poster"),
    (r"^\s*demo(?:nstration)?s?\b\s*:", "demo"),
    (r"^\s*late[\s\-]*breaking(?:\s+results?)?\b\s*:", "late-breaking"),
    (r"^\s*(?:lbr|wip)\b\s*:", "wip"),
    (r"^\s*work[\s\-]*in[\s\-]*progress\b\s*:", "wip"),
    (r"^\s*invited\b\s*:", "invited"),
    (r"^\s*tutorial\b\s*:", "tutorial"),
    (r"^\s*keynote\b\s*:", "keynote"),
    (r"^\s*panel\b\s*:", "panel"),
    (r"^\s*extended abstract\b\s*:", "abstract"),
]
_COMPILED_TITLE = [(re.compile(p, re.I), tt) for p, tt in _TITLE_RULES]


def classify_track(section_name: str | None, title: str | None = "") -> TrackVerdict:
    """Return the :class:`TrackVerdict` for one proceedings entry.

    ``section_name`` is the DBLP TOC heading the entry sits under (may be empty for
    flat venues). ``title`` is used only as a fallback when the section is absent or
    looks like a generic research session.
    """
    section = (section_name or "").strip().lower()
    for rx, tt in _COMPILED_SECTION:
        if rx.search(section):
            return TrackVerdict(tt)

    # No non-research / special section matched -> consult the title prefix (flat TOCs).
    text = title or ""
    for rx, tt in _COMPILED_TITLE:
        if rx.search(text):
            return TrackVerdict(tt)

    return TrackVerdict("research")
