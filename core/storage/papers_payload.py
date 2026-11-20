"""Single executable payload schema for the `papers` Qdrant collection.

`papers` is a heterogeneous container of FOUR document profiles (measured in
production 2026-09-01, see docs/papers_payload_contract_plan.md 1.1):

    source_type | points        | paper_id | written by
    paper       | 13,550,265    | yes      | local_ingest.py / api_routes.py
    review      | 2,201         | yes      | same 8-field whitelist as paper
    upload      | 3,786         | no       | QdrantStore.add_chunks
    webpage     | 126           | no       | QdrantStore.add_chunks

`upload`/`webpage` are not papers — forcing the 8-field paper whitelist onto
them would silently drop business fields (`original_filename`, `status_code`,
...) that have no SQL home (D1). This module dispatches on `source_type` and
is the ONLY place allowed to decide what a `papers` payload contains; every
write path (api_routes.py, QdrantStore.add_chunks, local_ingest.py,
recreate_qdrant_collection.py) must route through it instead of hand-rolling
its own key list — that drift is exactly what let `api_routes.py` carry a
stale "9-field" comment while local_ingest.py had already moved to 8
(plan.md 1.4).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, Optional

_logger = logging.getLogger(__name__)

# paper/review share one 8-field whitelist. `data_source` is deliberately
# excluded (D3): it is meaningless for a paper (JOIN documents.data_source
# via doc_id when needed) and its presence on ~1,327 production paper/review
# points is contamination, not a legitimate field.
PAPER_PROFILE_KEYS: FrozenSet[str] = frozenset({
    "doc_id", "paper_id", "source_type", "level",
    "venue", "year", "venue_tier", "content",
})

# Keys a paper/review point MAY carry beyond the 8 above. Unlike those, an
# optional key is absent unless it says something: `withdrawn` is written only as
# True (an arXiv paper whose newest version is a withdrawal). A False/None value
# is dropped, never stored, so the ~13.5M existing points need no migration and
# "not withdrawn" stays the absence of the key. Qdrant matches an absent key as
# not-equal, so `must_not withdrawn=true` keeps points without it; see
# lib/segment/src/index/query_estimator.rs::invert_estimation. A BOOL payload
# index covers the key (scripts/recreate_qdrant_collection.py).
# `preprint_full_text` follows the same rule: True only when the indexed text is
# an arXiv/open-repository copy rather than the publisher's version (SQL
# papers.full_text_source says which); absent means publisher or unknown.
OPTIONAL_PAPER_KEYS: FrozenSet[str] = frozenset({"withdrawn", "preprint_full_text"})
_OPTIONAL_KEYS_BY_SOURCE_TYPE: Dict[str, FrozenSet[str]] = {
    "paper": OPTIONAL_PAPER_KEYS,
    "review": OPTIONAL_PAPER_KEYS,
}

# Keys EVERY profile may carry, kept only as a non-empty string. `embed_fp` is the
# fingerprint of the embedding profile that produced the point's vectors
# (core.indexing.embedding_profile); a point without it predates the profile and
# is the one a re-embed selects. A KEYWORD payload index covers it
# (scripts/recreate_qdrant_collection.py); it is not in OPTIONAL_PAPER_KEYS because
# those are BOOL-indexed paper flags.
COMMON_OPTIONAL_STRING_KEYS: FrozenSet[str] = frozenset({"embed_fp"})

# Exact production key set for `upload` (plan.md 1.1, sampled 2026-09-01).
# No paper_id — an upload is not a paper. `data_source` IS legitimate here
# (real business provenance: system vs. user-uploaded).
UPLOAD_PROFILE_KEYS: FrozenSet[str] = frozenset({
    "chunk_id", "chunk_size", "content", "data_source", "doc_id", "end_idx",
    "file_size", "level", "original_filename", "source", "source_type",
    "start_idx", "title",
})

# Exact production key set for `webpage` (plan.md 1.1, sampled 2026-09-01).
WEBPAGE_PROFILE_KEYS: FrozenSet[str] = frozenset({
    "chunk_id", "content", "crawl_time", "crawled_at", "data_source", "doc_id",
    "end_idx", "level", "proxy_used", "retry_count", "section_name", "source",
    "source_type", "start_idx", "status_code", "title",
})

# source_type -> profile. `review` is text-only OpenReview commentary tied to
# a paper_id (core.db.models.SourceType.review) and shares the paper profile
# object exactly — not a copy, so the two can never drift apart independently.
PROFILE_BY_SOURCE_TYPE: Dict[str, FrozenSet[str]] = {
    "paper": PAPER_PROFILE_KEYS,
    "review": PAPER_PROFILE_KEYS,
    "upload": UPLOAD_PROFILE_KEYS,
    "webpage": WEBPAGE_PROFILE_KEYS,
}


class UnknownSourceTypeError(ValueError):
    """Raised when a payload's `source_type` has no registered profile.

    Fail-closed by design: a 5th document profile must be an explicit,
    reviewed addition to PROFILE_BY_SOURCE_TYPE, never an accidental
    pass-through of arbitrary payload into the shared `papers` collection
    (D1 — this period we do not add a 5th profile).
    """


def project_papers_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Project a raw payload dict onto the `papers` collection's one schema.

    Dispatches on ``raw["source_type"]`` to pick the profile's key whitelist,
    normalizes ``doc_id`` to ``str`` (D2 — Qdrant's KEYWORD index takes only
    string values, so the ~6.9M int-typed `doc_id` values already in
    production are absent from it, which is exactly the observed 49.3% index
    coverage. An int-VALUED filter still reaches them through a payload scan,
    but every caller queries ``str(doc_id)``, and that returns 0 matches over
    those points without raising. Both facts are pinned against Qdrant 1.18 in
    tests/test_papers_payload_real_qdrant.py), and drops both out-of-profile
    keys and None-valued keys
    (a missing SQL value must never write a literal null over real data).

    Pure / no I/O other than a warning log when keys are dropped, so a new
    out-of-bound write shows up in logs instead of silently matching legacy
    behavior.
    """
    source_type = raw.get("source_type")
    profile = PROFILE_BY_SOURCE_TYPE.get(source_type)
    if profile is None:
        raise UnknownSourceTypeError(
            f"papers collection has no payload profile for "
            f"source_type={source_type!r}; known profiles: "
            f"{sorted(PROFILE_BY_SOURCE_TYPE)}"
        )

    optional = _OPTIONAL_KEYS_BY_SOURCE_TYPE.get(source_type, frozenset())
    out: Dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in raw.items():
        if key == "doc_id":
            continue  # handled below — needs str normalization, not a raw copy
        if key in COMMON_OPTIONAL_STRING_KEYS:
            if isinstance(value, str) and value:
                out[key] = value
            continue
        if key in optional:
            if value is True:
                out[key] = True
            continue
        if key not in profile:
            dropped.append(key)
            continue
        if value is not None:
            out[key] = value

    if "doc_id" in profile:
        doc_id: Optional[Any] = raw.get("doc_id")
        if doc_id is not None:
            out["doc_id"] = str(doc_id)
        # else: stays absent. A stringified "None" would satisfy a keyword
        # filter that should return zero matches — worse than missing.

    if dropped:
        _warn_dropped_once(source_type, dropped)

    return out


# Distinct (source_type, dropped key set) combinations already reported.
# QdrantStore.add_chunks structurally adds chunk_id/start_idx/end_idx/title/
# source to EVERY point, and none of those are in the paper profile, so an
# unconditional warning fires on every chunk — 13.5M lines for one full
# reindex, which buries the signal it exists to provide. Report each distinct
# combination once instead: a genuinely new out-of-bound key still produces a
# fresh line, which is the observability the warning was added for.
_warned_drops: set = set()
_WARNED_DROPS_CAP = 256  # write paths are few; a runaway would be the bug


def _warn_dropped_once(source_type: Optional[str], dropped: list[str]) -> None:
    signature = (source_type, frozenset(dropped))
    if signature in _warned_drops:
        return
    if len(_warned_drops) < _WARNED_DROPS_CAP:
        _warned_drops.add(signature)
    _logger.warning(
        "[papers_payload] dropped out-of-profile keys for source_type=%s: %s "
        "(reported once per distinct key set, not per point)",
        source_type, sorted(dropped),
    )
