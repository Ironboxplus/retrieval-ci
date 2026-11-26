"""Paper search over three collections: formal papers, arXiv preprints and reviews.

`papers` used to hold all three, and default search kept preprints and reviews out
with a must_not on `venue_tier`/`source_type`. That filter matched ~14M points, so
Qdrant ran filtered HNSW and read the cold `point_to_values` index file at every
visited node — the main cost of a 9-16 s cold query (journal 2026-10-22/27). With
preprints in `arxiv` and reviews in `reviews`, each request goes to the collections
its filters can match, and conditions that are always true there are dropped, so the
default request reaches `papers` with no filter at all.

`route` reads the filter dict that `retrieve_filters` already builds — the
`__exclude__` defaults plus any `venue_tier`/`source_type`/`venue` scope — so the
search pipeline needs no new request fields. The layout (mixed/split, and whether
`papers` still holds its legacy preprint/review copies) is read per request from
`core.storage.corpus_layout`, so the split goes live without a restart; while the
legacy copies remain they are excluded from `papers`, so a merged request never
returns the same chunk twice.

Results from several collections are merged by score. The dense and sparse scores
come from one model and are directly comparable; hybrid scores are Qdrant's
per-collection RRF of ranks, so interleaving them by score interleaves the ranks.
The reranker re-scores the merged pool either way.

Writes go to the collection each point's payload belongs to (all to `papers` under
the mixed layout). Deletes reach every paper collection that exists, so a copy made
by the migration cannot outlive its document. Counts and by-document lookups read
where the points are served from, minus the legacy copies still in `papers`.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.models.schemas import PREPRINT_TIER
from core.storage.corpus_layout import (
    REVIEW,
    ROLE_ONLY_EXCLUSIONS,
    CorpusLayout,
    collection_for_payload,
    current_layout,
    paper_collections,
    role_for_payload,
)
from core.storage.qdrant_store import QdrantCollectionMissing, QdrantStore

_logger = logging.getLogger(__name__)

PAPER = "paper"
ARXIV_VENUE = "arxiv"


def _values(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _match(key: str, value: Any) -> FieldCondition:
    return FieldCondition(key=key, match=MatchValue(value=value))


# The preprint and review copies `papers` keeps until the split is cleaned up.
LEGACY_COPIES = [_match("source_type", REVIEW), _match("venue_tier", PREPRINT_TIER)]


def _is_arxiv(venue: Any) -> bool:
    return isinstance(venue, str) and venue.lower() == ARXIV_VENUE


def _narrow(filters: Dict[str, Any], key: str, keep: Callable[[Any], bool]) -> None:
    """Keep only the values of `key` that `keep` accepts (the key is known to keep
    at least one); a single survivor is stored as a scalar, as callers pass it."""
    if key not in filters:
        return
    kept = [v for v in _values(filters[key]) if keep(v)]
    filters[key] = kept[0] if len(kept) == 1 else kept


def _without_exclusions(exclude: Dict[str, Any], drop: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in exclude.items():
        kept = [v for v in _values(value) if v != drop.get(key)]
        if kept:
            out[key] = kept
    return out


def _split_role_only(exclude: Dict[str, Any]) -> tuple:
    """(exclusions valid in every collection, {role: exclusions valid in that role only})."""
    common = {k: v for k, v in exclude.items() if k not in ROLE_ONLY_EXCLUSIONS}
    only: Dict[str, Dict[str, Any]] = {}
    for key, role in ROLE_ONLY_EXCLUSIONS.items():
        if key in exclude:
            only.setdefault(role, {})[key] = exclude[key]
    return common, only


def for_unsplit_papers(filter_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The filter for a request served entirely from `papers` (unsplit layout, or a
    split collection that cannot be opened): `papers` has none of the role-only
    payload keys, so their exclusions are dropped."""
    filters = dict(filter_dict or {})
    if filters.get("__exclude__"):
        common, _ = _split_role_only(filters["__exclude__"])
        if common:
            filters["__exclude__"] = common
        else:
            del filters["__exclude__"]
    return filters


def route(filter_dict: Optional[Dict[str, Any]], *,
          papers_legacy_mixed: bool) -> Dict[str, Dict[str, Any]]:
    """Map one retrieve-stage filter dict to {collection role: residual filter dict}.

    Roles are "papers", "arxiv" and "reviews", in that order. A role is included only
    if its points can satisfy the filter; its residual filter drops what is always
    true in that collection (venue_tier=preprint and venue=arXiv in `arxiv`,
    source_type=review in `reviews`) and the default exclusions that cannot match
    there, including the role-only ones (`ROLE_ONLY_EXCLUSIONS`: `withdrawn` goes to
    `arxiv` alone). Every other condition, including a caller's own `__exclude__`,
    is kept.
    """
    filters = deepcopy(filter_dict or {})
    exclude, role_only = _split_role_only(filters.pop("__exclude__", None) or {})
    excluded_tiers = set(_values(exclude.get("venue_tier", [])))
    excluded_sources = set(_values(exclude.get("source_type", [])))
    tiers = set(_values(filters["venue_tier"])) if "venue_tier" in filters else None
    sources = set(_values(filters["source_type"])) if "source_type" in filters else None
    venues = _values(filters["venue"]) if "venue" in filters else None

    has_formal_venue = venues is None or any(not _is_arxiv(v) for v in venues)
    formal_tier = tiers is None or bool(tiers - {PREPRINT_TIER})
    arxiv_ok = (PREPRINT_TIER not in excluded_tiers
                and (tiers is None or PREPRINT_TIER in tiers)
                and (sources is None or PAPER in sources)
                and (venues is None or any(_is_arxiv(v) for v in venues)))
    reviews_ok = (REVIEW not in excluded_sources
                  and (sources is None or REVIEW in sources)
                  and formal_tier and has_formal_venue)
    papers_ok = formal_tier and has_formal_venue and (sources is None or bool(sources - {REVIEW}))

    defaults = {"venue_tier": PREPRINT_TIER, "source_type": REVIEW}
    plan: Dict[str, Dict[str, Any]] = {}
    if papers_ok:
        residual = deepcopy(filters)
        _narrow(residual, "venue_tier", lambda v: v != PREPRINT_TIER)
        _narrow(residual, "source_type", lambda v: v != REVIEW)
        _narrow(residual, "venue", lambda v: not _is_arxiv(v))
        kept = _without_exclusions(exclude, defaults)
        if papers_legacy_mixed:
            kept["source_type"] = sorted(set(kept.get("source_type", [])) | {REVIEW})
            kept["venue_tier"] = sorted(set(kept.get("venue_tier", [])) | {PREPRINT_TIER})
        if kept:
            residual["__exclude__"] = kept
        plan["papers"] = residual
    if arxiv_ok:
        residual = deepcopy(filters)
        for key in ("venue_tier", "venue", "source_type"):
            residual.pop(key, None)
        kept = {**_without_exclusions(exclude, defaults), **role_only.get("arxiv", {})}
        if kept:
            residual["__exclude__"] = kept
        plan["arxiv"] = residual
    if reviews_ok:
        residual = deepcopy(filters)
        residual.pop("source_type", None)
        _narrow(residual, "venue_tier", lambda v: v != PREPRINT_TIER)
        _narrow(residual, "venue", lambda v: not _is_arxiv(v))
        kept = _without_exclusions(exclude, defaults)
        if kept:
            residual["__exclude__"] = kept
        plan["reviews"] = residual
    return plan


class CorpusStore:
    """QdrantStore-compatible search over papers / arxiv / reviews (see module doc).

    `layout` is a fixed CorpusLayout (tests) or a callable returning the current one
    (default: the runtime layout file). The arxiv/reviews stores are built on first
    use under the split layout; if one cannot be opened (collection not created yet)
    the request is served the mixed way from `papers`, which still holds everything
    until the legacy copies are deleted, and the failure is logged.
    """

    def __init__(self, *, papers: QdrantStore, arxiv: QdrantStore | None = None,
                 reviews: QdrantStore | None = None,
                 layout: Union[CorpusLayout, Callable[[], CorpusLayout], None] = None):
        self._stores: Dict[str, Optional[QdrantStore]] = {
            "papers": papers, "arxiv": arxiv, "reviews": reviews}
        self._layout = layout if layout is not None else current_layout
        self._open_lock = threading.Lock()
        self.collection_name = papers.collection_name

    def layout(self) -> CorpusLayout:
        return self._layout() if callable(self._layout) else self._layout

    def _store(self, role: str) -> QdrantStore:
        store = self._stores[role]
        if store is not None:
            return store
        with self._open_lock:
            if self._stores[role] is None:
                self._stores[role] = QdrantStore(collection_name=paper_collections()[role],
                                                  create_if_missing=False)
            return self._stores[role]

    @property
    def supports_sparse(self) -> bool:
        return self._stores["papers"].supports_sparse

    def _fan_out(self, filter_dict: Optional[Dict[str, Any]], top_k: int,
                 run: Callable[[QdrantStore, Dict[str, Any]], List[Dict]]) -> List[Dict]:
        if top_k <= 0:
            return []
        layout = self.layout()
        if not layout.split:
            return run(self._stores["papers"], for_unsplit_papers(filter_dict))
        plan = route(filter_dict, papers_legacy_mixed=layout.papers_legacy_mixed)
        if not plan:
            return []
        try:
            stores = {role: self._store(role) for role in plan}
        except Exception:
            _logger.exception("[corpus] split collection unavailable; serving from papers")
            return run(self._stores["papers"], for_unsplit_papers(filter_dict))
        if len(plan) == 1:
            (role, residual), = plan.items()
            return run(stores[role], residual)
        with ThreadPoolExecutor(max_workers=len(plan)) as executor:
            groups = list(executor.map(lambda item: run(stores[item[0]], item[1]),
                                       plan.items()))
        merged = [row for group in groups for row in group]
        merged.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        return merged[:top_k]

    @staticmethod
    def _fold(filter_dict, level, source_type, data_source) -> Dict[str, Any]:
        folded = dict(filter_dict or {})
        for key, value in (("level", level), ("source_type", source_type),
                           ("data_source", data_source)):
            if value is not None:
                folded[key] = value
        return folded

    def search(self, query_embedding: List[float], top_k: int = 10,
               filter_dict: Dict[str, Any] = None, query_filter=None,
               level: int = None, source_type: str = None,
               data_source: str = None) -> List[Dict]:
        if query_filter is not None:
            return self._stores["papers"].search(query_embedding, top_k=top_k,
                                                 query_filter=query_filter)
        return self._fan_out(
            self._fold(filter_dict, level, source_type, data_source), top_k,
            lambda store, residual: store.search(query_embedding, top_k=top_k,
                                                 filter_dict=residual))

    def sparse_search(self, sparse_vector, top_k: int = 10,
                      filter_dict: Dict[str, Any] = None, query_filter=None,
                      level: int = None, source_type: str = None,
                      data_source: str = None) -> List[Dict]:
        if query_filter is not None:
            return self._stores["papers"].sparse_search(sparse_vector, top_k=top_k,
                                                        query_filter=query_filter)
        return self._fan_out(
            self._fold(filter_dict, level, source_type, data_source), top_k,
            lambda store, residual: store.sparse_search(sparse_vector, top_k=top_k,
                                                        filter_dict=residual))

    def hybrid_search(self, dense_vector: List[float], sparse_vector, top_k: int = 10,
                      prefetch_limit: int = 100, filter_dict: Dict[str, Any] = None,
                      query_filter=None, level: int = None, source_type: str = None,
                      data_source: str = None) -> List[Dict]:
        if query_filter is not None:
            return self._stores["papers"].hybrid_search(
                dense_vector, sparse_vector, top_k=top_k, prefetch_limit=prefetch_limit,
                query_filter=query_filter)
        return self._fan_out(
            self._fold(filter_dict, level, source_type, data_source), top_k,
            lambda store, residual: store.hybrid_search(
                dense_vector=dense_vector, sparse_vector=sparse_vector, top_k=top_k,
                prefetch_limit=prefetch_limit, filter_dict=residual))

    # The bounded-parallel batch only calls self.hybrid_search, so it routes as is.
    hybrid_search_many_exact = QdrantStore.hybrid_search_many_exact

    # ---- writes ----

    def build_points(self, chunks, **kwargs) -> List[PointStruct]:
        # The three collections share one payload contract, so projecting through
        # `papers` is the same projection each target would apply.
        return self._stores["papers"].build_points(chunks, **kwargs)

    def upsert_points(self, points: List[PointStruct], wait: bool = True) -> None:
        """Write each point to the collection it belongs to under the current layout.
        A missing target collection raises before anything is written: falling back
        to `papers` would put the point where the split no longer serves it."""
        layout = self.layout()
        groups: Dict[str, List[PointStruct]] = {}
        for point in points or []:
            groups.setdefault(collection_for_payload(point.payload or {}, layout),
                              []).append(point)
        roles = {name: role for role, name in paper_collections().items()}
        stores = {name: self._store(roles[name]) for name in groups}
        for name, group in groups.items():
            stores[name].upsert_points(group, wait=wait)

    def add_chunks(self, chunks, source_type: str = None, data_source: str = None,
                   doc_title: str = None, doc_source: str = None, use_upsert: bool = True,
                   wait: bool = True, sparse_vectors=None) -> None:
        self.upsert_points(self.build_points(
            chunks, source_type=source_type, data_source=data_source,
            doc_title=doc_title, doc_source=doc_source, sparse_vectors=sparse_vectors,
        ), wait=wait)

    # ---- deletes ----

    def _existing_stores(self) -> List[QdrantStore]:
        stores = [self._stores["papers"]]
        for role in ("arxiv", "reviews"):
            try:
                stores.append(self._store(role))
            except QdrantCollectionMissing:
                continue
        return stores

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete the document's points from every paper collection that exists, in
        either layout: during the migration copy a document lives in two of them."""
        return sum(store.delete_by_doc_id(doc_id) for store in self._existing_stores())

    # ---- counts and lookups ----

    def _served(self, roles=None) -> List[tuple]:
        """(role, store, must_not) for the collections serving `roles` (default all)
        under the current layout, `papers` excluding its legacy copies. Falls back
        to all of `papers` if a split collection is missing, as search does."""
        papers = self._stores["papers"]
        layout = self.layout()
        if not layout.split:
            return [("papers", papers, None)]
        roles = list(roles or paper_collections())
        try:
            stores = {role: self._store(role) for role in roles}
        except Exception:
            _logger.exception("[corpus] split collection unavailable; reading papers")
            return [("papers", papers, None)]
        legacy = LEGACY_COPIES if layout.papers_legacy_mixed else None
        return [(role, store, legacy if role == "papers" else None)
                for role, store in stores.items()]

    def count_points(self, key: str, value: Any, *, roles=None) -> int:
        """Exact count of points with payload `key == value` in the collections
        serving `roles`."""
        return sum(
            store.client.count(
                store.collection_name,
                count_filter=Filter(must=[_match(key, value)], must_not=must_not),
                exact=True,
            ).count
            for _, store, must_not in self._served(roles)
        )

    def point_counts(self) -> Dict[str, int]:
        """Points per serving collection, plus `legacy_copies`: the preprint/review
        copies `papers` still holds. Those are estimated per keyword value from the
        payload index rather than counted exactly over ~24M points, and each total is the
        collection's own ``points_count`` (metadata, no scan), so this stays cheap
        enough for a dashboard to poll."""
        counts: Dict[str, int] = {"legacy_copies": 0}
        for role, store, must_not in self._served():
            counts[role] = _collection_points(store)
            for condition in must_not or []:
                counts["legacy_copies"] += store.client.count(
                    store.collection_name, count_filter=Filter(must=[condition]),
                    exact=False,
                ).count
        return counts

    def count(self) -> int:
        """Distinct corpus points: the serving collections minus legacy copies."""
        try:
            counts = self.point_counts()
        except Exception:
            return 0
        legacy = counts.pop("legacy_copies")
        return sum(counts.values()) - legacy

    def count_by_doc_id(self, doc_id: str) -> int:
        try:
            return self.count_points("doc_id", doc_id)
        except Exception:
            return 0

    def exists_by_doc_id(self, doc_id: str) -> bool:
        return self.count_by_doc_id(doc_id) > 0


def _collection_points(store: QdrantStore) -> int:
    """Points in the store's collection from collection metadata (no scan). Qdrant
    documents ``points_count`` as approximate while an optimizer is rewriting segments;
    that is the same figure the dashboard already shows from ``GET /collections/{c}``.
    Falls back to an exact count only if the server does not report it."""
    points = store.client.get_collection(store.collection_name).points_count
    if points is None:
        points = store.client.count(store.collection_name, exact=True).count
    return int(points)


def role_for_document(*, source_type: Any = None, venue_tier: Any = None) -> str:
    """The collection role of a SQL document's points: reviews by source_type,
    preprints by the paper's venue_tier, everything else in `papers`."""
    source = getattr(source_type, "value", source_type)
    return role_for_payload({"source_type": source, "venue_tier": venue_tier})


def paper_store(collection_name: Optional[str] = None):
    """The store for a write/lookup service: the routed corpus for the papers
    collection, a plain QdrantStore for any other."""
    papers = paper_collections()["papers"]
    name = collection_name or papers
    if name == papers:
        return CorpusStore(papers=QdrantStore(collection_name=name))
    return QdrantStore(collection_name=name)


def build_search_store() -> CorpusStore:
    """The store behind SearchPipeline; serves the mixed layout until the split
    layout file says otherwise."""
    return CorpusStore(papers=QdrantStore())
