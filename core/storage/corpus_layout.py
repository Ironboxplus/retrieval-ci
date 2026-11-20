"""Which Qdrant collection a paper-corpus point belongs to, switchable at runtime.

The corpus is moving from one `papers` collection to three: formal papers, arXiv
preprints (`arxiv`) and reviews (`reviews`). The layout is read from a small JSON
file on every call (cached by mtime), so the backend and the ingest scripts switch
together without a restart:

    {"layout": "mixed" | "split", "papers_legacy_mixed": true | false}

`mixed` is the old single-collection layout and the default. `split` routes each
point by its payload. `papers_legacy_mixed` stays true until the legacy preprint and
review copies have been deleted from `papers`; while it is true, searches of
`papers` keep excluding them so a merged search never returns a chunk twice.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.models.schemas import PREPRINT_TIER
from config import settings

_logger = logging.getLogger(__name__)

QDRANT_PAPERS_COLLECTION = os.environ.get("QDRANT_PAPERS_COLLECTION", "papers").strip() or "papers"
QDRANT_ARXIV_COLLECTION = os.environ.get("QDRANT_ARXIV_COLLECTION", "arxiv").strip() or "arxiv"
QDRANT_REVIEWS_COLLECTION = (
    os.environ.get("QDRANT_REVIEWS_COLLECTION", "reviews").strip() or "reviews")
REVIEW = "review"

# Default-exclusion keys whose payload field exists, and is payload-indexed, in one
# collection role only. `withdrawn` is written (as `true`, never `false`) onto points of
# withdrawn arXiv papers and has a BOOL index in `arxiv`; `papers` has neither the key
# nor the index. A request is sent the exclusion only for the role named here: anywhere
# else it could never match and would only add an unindexed condition to every visited
# point of a 13M-point collection.
ROLE_ONLY_EXCLUSIONS = {"withdrawn": "arxiv"}


def paper_collections() -> dict[str, str]:
    """Role -> collection name, in search-merge order."""
    return {"papers": QDRANT_PAPERS_COLLECTION, "arxiv": QDRANT_ARXIV_COLLECTION,
            "reviews": QDRANT_REVIEWS_COLLECTION}


def is_paper_collection(name: str) -> bool:
    return name in paper_collections().values()


@dataclass(frozen=True)
class CorpusLayout:
    split: bool = False
    papers_legacy_mixed: bool = True


def layout_path() -> Path:
    configured = os.environ.get("QDRANT_CORPUS_LAYOUT_FILE")
    if configured:
        return Path(configured)
    return Path(settings.index_dir) / "corpus_layout.json"


_cache_lock = threading.Lock()
_cache: tuple[Path, float, CorpusLayout] | None = None


def _parse(raw: Mapping[str, Any]) -> CorpusLayout:
    layout = str(raw.get("layout", "mixed")).strip().lower()
    if layout not in {"mixed", "split"}:
        raise ValueError(f"unknown corpus layout {layout!r}")
    legacy = raw.get("papers_legacy_mixed", True)
    if not isinstance(legacy, bool):
        raise ValueError("papers_legacy_mixed must be a JSON boolean")
    return CorpusLayout(split=layout == "split", papers_legacy_mixed=legacy)


def current_layout() -> CorpusLayout:
    """The layout file's content; `mixed` if it is absent. An unreadable or invalid
    file keeps the last good layout (or `mixed`) and logs, rather than guessing."""
    global _cache
    path = layout_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return CorpusLayout()
    with _cache_lock:
        if _cache is not None and _cache[0] == path and _cache[1] == mtime:
            return _cache[2]
        try:
            layout = _parse(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            fallback = _cache[2] if _cache is not None and _cache[0] == path else CorpusLayout()
            _logger.error("[corpus] ignoring invalid layout file %s (%s); using %s",
                          path, exc, fallback)
            return fallback
        _cache = (path, mtime, layout)
        return layout


def write_layout(layout: CorpusLayout, path: Path | None = None) -> Path:
    """Atomically replace the layout file (ops scripts and tests)."""
    target = path or layout_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "layout": "split" if layout.split else "mixed",
        "papers_legacy_mixed": layout.papers_legacy_mixed,
    }), encoding="utf-8")
    os.replace(tmp, target)
    return target


def role_for_payload(payload: Mapping[str, Any]) -> str:
    """The collection role a paper-corpus point belongs to under the split layout."""
    if payload.get("source_type") == REVIEW:
        return "reviews"
    if payload.get("venue_tier") == PREPRINT_TIER:
        return "arxiv"
    return "papers"


def collection_for_payload(payload: Mapping[str, Any],
                           layout: CorpusLayout | None = None) -> str:
    layout = layout or current_layout()
    if not layout.split:
        return QDRANT_PAPERS_COLLECTION
    return paper_collections()[role_for_payload(payload)]


class MisroutedPaperPoint(ValueError):
    """A paper-corpus point was about to be written to a collection it does not
    belong to; raised before the write, so nothing lands in the wrong place."""


def check_paper_point(collection: str, payload: Mapping[str, Any],
                      layout: CorpusLayout | None = None) -> None:
    """Pre-write guard for every write into a paper collection. `arxiv` only takes
    preprints and `reviews` only reviews in any layout; `papers` refuses both once
    the split is live. Non-paper collections are not this guard's business."""
    roles = {name: role for role, name in paper_collections().items()}
    role = roles.get(collection)
    if role is None:
        return
    belongs = role_for_payload(payload)
    if role == "papers":
        layout = layout or current_layout()
        if not layout.split or belongs == "papers":
            return
    elif belongs == role:
        return
    raise MisroutedPaperPoint(
        f"refusing to write a {belongs} point (paper_id={payload.get('paper_id')!r}, "
        f"venue_tier={payload.get('venue_tier')!r}, source_type={payload.get('source_type')!r}) "
        f"into `{collection}`")
