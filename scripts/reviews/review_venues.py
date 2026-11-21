"""Registry of OpenReview venues that expose PUBLIC reviews (probed 2026-06-17).

Single source of truth for: canonical short name, CCF tier, the OpenReview venue
id template, and the api2-era years to harvest. Org-prefixed ids (colmweb.org/..,
auai.org/.., robot-learning.org/..) MUST carry an explicit name — the id's first
component is an org domain, not the acronym.

Confirmed public (n_reviews>0 unauth): ICLR, NeurIPS (+ Datasets&Benchmarks),
COLM, CoRL, UAI, MIDL, and — newly confirmed 2026-06-17 — **ICML 2025** (ICML
adopted public reviews in 2025; accepted + opt-in rejected. ICML <=2024 stays
decisions-only). Per-venue review field names differ; harvest_openreview.py's
REVIEW_TEXT_MAP/REVIEW_SCORE_MAP cover them (ICML 2025 puts the critique in
``other_strengths_and_weaknesses`` + ``questions_for_authors`` +
``overall_recommendation``, NOT ``weaknesses``).

OTHER PLATFORMS explored 2026-06-17 (why they're NOT here):
  * TMLR — CONFIRMED public on OpenReview (4,086 papers; review invitation
    ``Review``; critique in ``requested_changes`` + ``summary_of_contributions``).
    NOT added: it's a continuous journal (no year-scoped invitation) and >1000
    notes, so it shares the unauth rate-limit/auth blocker of ICLR/NeurIPS — pick
    it up once the bulk-harvest auth question is settled.
  * alphaXiv — informal arXiv comments (millions, ML-heavy) but the public API is
    an OAuth MCP server (paper search/retrieval), NO comments feed by arXiv id
    (probed: all 404). Not formal reviews, not bulk-harvestable.
  * Hugging Face Papers — /api/papers/{arxiv} returns metadata only; no public
    discussion/comments JSON endpoint (probed: 404). Informal comments anyway.
  * PubPeer — DOI/arXiv/PubMed keyed but integrity/fraud-flag focused, life-sci
    heavy, no open API. SciPost — physics/astro (no CS overlap). eLife / F1000 /
    PLOS / Copernicus transparent peer review — life/geo sci, outside our corpus.
  => For FORMAL structured ML reviews at scale, OpenReview is the only source;
     the lever is its venue list (below), not other platforms.

Confirmed PRIVATE / not-harvestable: ICML <=2024 (decisions only), AISTATS / RLC
/ LoG (no public reviews on sampled years), AAAI / KDD / CVPR (CMT). ACL/EMNLP
use ARR (OpenReview) but with a monthly-cycle / portable-review structure not
covered by the simple venueid query — deferred.

Tiers: ICLR/NeurIPS/ICML = CCF-A, UAI = CCF-B; COLM/CoRL/MIDL not CCF-ranked.
"""
from __future__ import annotations

REVIEW_VENUES = [
    {"name": "ICLR",       "tier": "CCF-A", "id": "ICLR.cc/{year}/Conference",
     "years": [2024, 2025, 2026]},
    {"name": "NeurIPS",    "tier": "CCF-A", "id": "NeurIPS.cc/{year}/Conference",
     "years": [2023, 2024, 2025]},
    {"name": "NeurIPS-DB", "tier": "CCF-A",
     "id": "NeurIPS.cc/{year}/Datasets_and_Benchmarks_Track",
     "years": [2023, 2024, 2025]},
    {"name": "ICML",       "tier": "CCF-A", "id": "ICML.cc/{year}/Conference",
     "years": [2025]},  # public reviews only from 2025 on
    {"name": "COLM",       "tier": None,    "id": "colmweb.org/COLM/{year}/Conference",
     "years": [2024, 2025]},
    {"name": "CoRL",       "tier": None,    "id": "robot-learning.org/CoRL/{year}/Conference",
     "years": [2023, 2024]},
    {"name": "UAI",        "tier": "CCF-B", "id": "auai.org/UAI/{year}/Conference",
     "years": [2022, 2023, 2024, 2025]},
    {"name": "MIDL",       "tier": None,    "id": "MIDL.io/{year}/Conference",
     "years": [2023, 2024, 2025]},
]

# venue name (UPPER) -> CCF tier, for preindex payload. Only tiered venues listed.
VENUE_TIER = {v["name"].upper(): v["tier"] for v in REVIEW_VENUES if v["tier"]}
