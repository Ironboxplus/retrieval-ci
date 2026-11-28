"""End-to-end duplicate scan against a real Qdrant instance.

tests/test_duplicate_rollup.py covers the pure roll-up. What it cannot cover is
the half that actually talks to a vector database: batched nearest-neighbour
queries keyed by point id, the score_threshold cut, and the owner/project filter
that keeps one tenant's files out of another's report. Those are Qdrant
behaviours, so they are tested against Qdrant.

Vectors here are hand-built unit vectors rather than BGE-M3 output. That is not
a stand-in for the search engine — every query, filter and threshold below is
executed by a real Qdrant — it just makes "these two chunks are duplicates" an
exact property of the fixture instead of a property of an embedding model, so a
failure means the scan is wrong rather than that the model drifted.

Skipped unless a Qdrant is reachable on loopback.
"""
from __future__ import annotations

import time
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.code_kb.duplicates import scan_duplicates  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from core.code_kb.duplicate_rollup import DEFAULT_EXCLUDE_GLOBS  # noqa: E402
from core.code_kb.vector_records import CodeChunkVectorRecord  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

# Loopback only and hardcoded, never read from settings/QDRANT_URL, so a deploy
# host's config cannot point this suite at the 2.2M-point production collection.
LOCAL_URL = "http://127.0.0.1:6333"

# A scratch name of our own, NOT `code_chunks`: nothing here can collide with a
# real code index even if one happens to live on this instance.
COLLECTION = "code_chunks_dupscan_test"
DIM = 8
OWNER = "dup-owner"
PROJECT = "/repo"

# Unit basis vectors: identical vector => cosine 1.0, different basis => 0.0.
# With threshold 0.92 the duplicate structure below is exact, not approximate.
E = [[1.0 if i == j else 0.0 for i in range(DIM)] for j in range(DIM)]

# alpha/beta share TWO chunks -> a finding. gamma shares exactly ONE with each of
# them -> the measured noise shape, dropped by the default min. delta shares
# nothing. storage/dump.py would be a two-chunk finding but is excluded by path.
LAYOUT: list[tuple[str, list[int]]] = [
    ("app/alpha.py", [0, 1]),
    ("app/beta.py", [0, 1]),
    ("app/gamma.py", [0, 2]),
    ("app/delta.py", [3, 4]),
    ("storage/dump.py", [0, 1]),
]


def _records(path: str, basis: list[int]) -> list[tuple[CodeChunkVectorRecord, list[float]]]:
    out = []
    for ordinal, b in enumerate(basis):
        start = ordinal * 10 + 1
        out.append((
            CodeChunkVectorRecord(
                rel_path=path, language="python", start_line=start, end_line=start + 9,
                label=f"chunk{ordinal}", content=f"basis {b}", code=f"basis {b}",
                chunk_ordinal=ordinal,
            ),
            E[b],
        ))
    return out


def _drop_collection(client: QdrantClient, *, attempts: int) -> Exception | None:
    """Drop COLLECTION, tolerating the transient Windows rename failure."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return None
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    return last


@pytest.fixture()
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()  # fail fast when Qdrant is not reachable
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")

    # Unconditionally drop whatever is under this name before creating it.
    #
    # tests/test_papers_payload_real_qdrant.py must SKIP instead, because it is
    # forced to use the literal name `papers`, which may well be someone else's.
    # This name is private to this fixture, so nothing here can belong to anyone
    # else and a skip would be strictly worse: two failure modes make it stick
    # permanently, and a suite that silently stops running is the thing these
    # tests exist to prevent.
    #   1. a crashed run leaves the collection registered;
    #   2. on Windows a teardown can unregister the collection but leave
    #      ./storage/collections/<name> on disk, after which create returns
    #      400 "Collection data already exists" forever.
    # delete_collection fixes both: on an unregistered name it returns False and
    # still clears the orphaned directory (verified against Qdrant on 2026-09-01).
    #
    # Retried because the drop itself is the transient part on Windows: it 500s with
    # "failed to rename file ... .deleted/" while segment files are still mapped, and
    # the mapping is released a moment later. Setup is now the load-bearing cleanup,
    # so a single unlucky attempt must not error the whole file.
    _drop_collection(client, attempts=4)

    from config import settings
    dense_is_named = settings.embedding.supports_sparse
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": params} if dense_is_named else params,
    )

    # Only the two attributes a real QdrantStore carries as instance state. An
    # earlier version also set `dense_is_named`, an attribute no real store has
    # — and duplicates.py then read it, so this fixture was feeding the code
    # under test the very thing production would not provide. It passed here and
    # returned HTTP 500 on `.88`. Named-vs-bare vectors come from the class's
    # `supports_sparse` property, which __new__ instances inherit for free.
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = COLLECTION
    inner.client = client

    cvs = CodeVectorStore(collection_name=COLLECTION, store=inner)
    cvs.ensure_search_payload_indexes()
    try:
        yield cvs
    finally:
        # Best-effort. On Windows this intermittently 500s with "failed to rename
        # file ... to ./storage/.deleted/..." because the segment files are still
        # mapped, and failing teardown would report a PASSING test as an error.
        # Setup is the authoritative cleanup: it drops this name unconditionally,
        # so a leaked collection cannot survive into the next run either way.
        failure = _drop_collection(client, attempts=3)
        if failure is not None:
            warnings.warn(
                f"could not drop `{COLLECTION}` in teardown ({failure}); the next "
                "run's setup will clear it", stacklevel=1)


def _seed(cvs: CodeVectorStore, *, owner: str = OWNER, project: str = PROJECT,
          layout: list[tuple[str, list[int]]] | None = None) -> None:
    items, vectors = [], []
    for path, basis in (layout if layout is not None else LAYOUT):
        for record, vec in _records(path, basis):
            items.append(record)
            vectors.append(vec)
    cvs.upsert(owner_user_id=owner, project=project, host="testhost",
               items=items, dense_vectors=vectors)


def _seed_vectors(cvs: CodeVectorStore, *, project: str,
                  files: list[tuple[str, list[list[float]]]],
                  owner: str = OWNER) -> None:
    """Seed explicit vectors, for similarities that are not 0.0 or 1.0."""
    items, vectors = [], []
    for path, vecs in files:
        for ordinal, vec in enumerate(vecs):
            start = ordinal * 10 + 1
            items.append(CodeChunkVectorRecord(
                rel_path=path, language="python", start_line=start, end_line=start + 9,
                label=f"chunk{ordinal}", content="x", code="x", chunk_ordinal=ordinal))
            vectors.append(vec)
    cvs.upsert(owner_user_id=owner, project=project, host="testhost",
               items=items, dense_vectors=vectors)


def test_two_shared_chunks_are_reported_as_one_file_pair(store):
    _seed(store)
    result = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT)

    assert [(p.path_a, p.path_b) for p in result.pairs] == [("app/alpha.py", "app/beta.py")]
    assert result.pairs[0].shared_chunks == 2
    assert result.pairs[0].max_score == pytest.approx(1.0, abs=1e-4)
    assert not result.truncated


def test_single_shared_chunk_appears_only_when_the_min_is_lowered(store):
    _seed(store)
    loose = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                            min_shared_chunks=1)
    reported = {(p.path_a, p.path_b) for p in loose.pairs}
    assert ("app/alpha.py", "app/gamma.py") in reported
    assert ("app/beta.py", "app/gamma.py") in reported
    # The default suppressed both of these; that suppression is the whole feature.
    assert reported > {(p.path_a, p.path_b) for p in
                       scan_duplicates(store, owner_user_id=OWNER, project=PROJECT).pairs}


def test_files_sharing_nothing_never_appear(store):
    _seed(store)
    result = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                             min_shared_chunks=1)
    assert not any("app/delta.py" in (p.path_a, p.path_b) for p in result.pairs)


def test_excluded_paths_are_absent_even_though_they_are_perfect_duplicates(store):
    """storage/dump.py duplicates alpha exactly; only the exclusion keeps it out."""
    _seed(store)
    result = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                             min_shared_chunks=1)
    assert not any("storage/dump.py" in (p.path_a, p.path_b) for p in result.pairs)

    without = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                              min_shared_chunks=1, exclude_globs=())
    assert any("storage/dump.py" in (p.path_a, p.path_b) for p in without.pairs)


def test_another_owners_identical_files_do_not_leak_into_the_report(store):
    """Scope isolation is a filter pushed into Qdrant, so it is tested in Qdrant."""
    _seed(store)
    _seed(store, owner="other-owner", layout=[("app/theirs.py", [0, 1])])

    result = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                             min_shared_chunks=1)
    assert not any("app/theirs.py" in (p.path_a, p.path_b) for p in result.pairs)

    theirs = scan_duplicates(store, owner_user_id="other-owner", project=PROJECT,
                             min_shared_chunks=1)
    assert theirs.pairs == []  # one file alone is never a duplicate of itself


def test_another_project_of_the_same_owner_is_out_of_scope(store):
    _seed(store)
    _seed(store, project="/other-repo", layout=[("app/elsewhere.py", [0, 1])])
    result = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                             min_shared_chunks=1)
    assert not any("app/elsewhere.py" in (p.path_a, p.path_b) for p in result.pairs)


def test_threshold_cuts_a_similar_but_not_duplicate_pair(store):
    """Qdrant's score_threshold, exercised on chunks that are genuinely partway
    similar (cosine 0.7071) rather than identical or orthogonal."""
    h = 2 ** -0.5
    m01 = [h, h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    m12 = [0.0, h, h, 0.0, 0.0, 0.0, 0.0, 0.0]
    _seed_vectors(store, project="/angled", files=[
        ("app/p.py", [E[0], E[1]]),
        ("app/q.py", [m01, m12]),
    ])

    loose = scan_duplicates(store, owner_user_id=OWNER, project="/angled",
                            threshold=0.5, min_shared_chunks=1)
    assert [(p.path_a, p.path_b) for p in loose.pairs] == [("app/p.py", "app/q.py")]
    assert loose.pairs[0].max_score == pytest.approx(h, abs=1e-4)

    strict = scan_duplicates(store, owner_user_id=OWNER, project="/angled",
                             threshold=0.92, min_shared_chunks=1)
    assert strict.pairs == []
    assert strict.scanned_chunks == 4  # the chunks were scanned; only the score cut


def test_scan_counts_the_chunks_and_files_it_actually_examined(store):
    _seed(store)
    result = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT)
    # 5 files x 2 chunks, minus the excluded storage/dump.py.
    assert result.scanned_files == 4
    assert result.scanned_chunks == 8


def test_hitting_the_chunk_cap_is_reported_not_hidden(store):
    """A silently truncated scan reads as 'no duplicates found', which is a lie."""
    _seed(store)
    result = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT, max_chunks=3)
    assert result.truncated
    assert result.notice and "3" in result.notice


def test_an_empty_scope_is_an_empty_report_not_an_error(store):
    result = scan_duplicates(store, owner_user_id="nobody", project=PROJECT)
    assert result.pairs == [] and result.scanned_chunks == 0 and not result.truncated


def test_default_exclusions_are_the_shared_constant(store):
    """The API default must not drift from the documented constant."""
    _seed(store)
    explicit = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                               min_shared_chunks=1, exclude_globs=DEFAULT_EXCLUDE_GLOBS)
    default = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT,
                              min_shared_chunks=1)
    assert [(p.path_a, p.path_b) for p in explicit.pairs] == [
        (p.path_a, p.path_b) for p in default.pairs]


def test_examples_carry_line_numbers_so_a_finding_is_actionable(store):
    _seed(store)
    pair = scan_duplicates(store, owner_user_id=OWNER, project=PROJECT).pairs[0]
    example = pair.examples[0]
    assert example.a.path == "app/alpha.py" and example.b.path == "app/beta.py"
    assert example.a.start_line > 0 and example.a.end_line >= example.a.start_line


# --------------------------------------------------------------------------- #
# Through the HTTP route, with a real service over the same real Qdrant.
# --------------------------------------------------------------------------- #

def _real_service(cvs: CodeVectorStore):
    from app.services.code_kb.service import CodeKBService

    def _no_embedding_should_be_needed(_texts):
        raise AssertionError(
            "a duplicate scan queries by point id and must never embed anything")

    return CodeKBService(embed_fn=_no_embedding_should_be_needed, vector_store=cvs)


def test_route_returns_the_scan_over_a_real_index(store):
    from fastapi.testclient import TestClient
    from app.api.routes import code_kb as route_module
    import main

    _seed(store)
    service = _real_service(store)
    original_owner = route_module.resolve_owner
    original_service = route_module.get_code_kb_service
    route_module.resolve_owner = lambda *_args: OWNER
    route_module.get_code_kb_service = lambda: service
    try:
        with TestClient(main.app) as client:
            response = client.post("/api/code/duplicates",
                                   json={"project": PROJECT, "min_shared_chunks": 2})
    finally:
        route_module.resolve_owner = original_owner
        route_module.get_code_kb_service = original_service

    assert response.status_code == 200
    body = response.json()
    assert body["project"] == PROJECT
    assert body["total"] == 1
    assert (body["pairs"][0]["path_a"], body["pairs"][0]["path_b"]) == (
        "app/alpha.py", "app/beta.py")
    assert body["pairs"][0]["shared_chunks"] == 2
    assert body["scanned_files"] == 4
    assert body["truncated"] is False
    assert body["pairs"][0]["examples"][0]["a"]["path"] == "app/alpha.py"


def test_route_rejects_an_out_of_range_threshold_before_touching_qdrant(store):
    from fastapi.testclient import TestClient
    from app.api.routes import code_kb as route_module
    import main

    original_owner = route_module.resolve_owner
    route_module.resolve_owner = lambda *_args: OWNER
    try:
        with TestClient(main.app) as client:
            response = client.post("/api/code/duplicates",
                                   json={"project": PROJECT, "threshold": 1.5})
    finally:
        route_module.resolve_owner = original_owner
    assert response.status_code == 422


def test_scanner_only_reads_store_attributes_that_really_exist(store):
    """Regression for the HTTP 500 on `.88`, 2026-09-02.

    `_neighbour_matches` read `store.dense_is_named` to choose between a named
    and a bare vector. No QdrantStore has that attribute — it existed solely
    because this fixture used to set it, so the scanner was being handed a
    capability production would never provide. Fourteen real-Qdrant tests
    passed and the first production call returned 500.

    The real decision comes from the `supports_sparse` property, which is also
    what QdrantStore itself uses when creating a collection
    (core/storage/qdrant_store.py:203).
    """
    assert not hasattr(QdrantStore, "dense_is_named")
    assert "dense_is_named" not in vars(store.store), (
        "the fixture must not invent attributes the real store lacks")
    assert isinstance(store.store.supports_sparse, bool)

    # And the instance state this fixture fabricates must stay a subset of what
    # a genuinely constructed store carries, so the next invented attribute
    # fails here rather than in production.
    # (`client` is a property; an assigned client lives in `_pinned_client`.)
    assert set(vars(store.store)) <= {"collection_name", "_pinned_client", "qdrant_dir"}


class TestAliasScopesDoNotMultiplyTheEvidence:
    """Production regression, measured on `.88` 2026-09-02.

    `find_duplicates` unions a Windows scope with its historical case-variant
    spellings, exactly like search does. That is right for search — you get your
    results whichever way you spell the path — and wrong here, because the same
    physical file is then present once per spelling. Every cross-file chunk match
    multiplies by (number of spellings)^2.

    Measured on the real index: this repository was indexed under THREE spellings
    (`C:/users/arc/prd/retrival`, `C:/Users/Arc/prd/retrival`,
    `C:/Users/Arc/Prd/retrival`) sharing 566 files, and the scan reported 484
    shared chunks for a pair the original probe measured at 65, with all three
    `examples` printing byte-identical `path:line` citations.

    The damage is not cosmetic. The module's whole design rests on the measured
    fact that 176 of 190 noise pairs share exactly ONE chunk, so
    `min_shared_chunks=2` filters them out. Under 3 spellings that single
    boilerplate chunk counts as NINE and clears the gate — the primary filter
    stops filtering.
    """

    ALIASES = ["/repo", "/Repo", "/REPO"]

    def _seed_every_spelling(self, store):
        for spelling in self.ALIASES:
            _seed(store, project=spelling)

    def test_shared_chunk_count_is_not_multiplied_by_the_spelling_count(self, store):
        self._seed_every_spelling(store)
        one = scan_duplicates(store, owner_user_id=OWNER, project=self.ALIASES[0])
        allof = scan_duplicates(store, owner_user_id=OWNER, project=self.ALIASES)

        assert [(p.path_a, p.path_b) for p in one.pairs] == [
            ("app/alpha.py", "app/beta.py")]
        assert one.pairs[0].shared_chunks == 2
        assert [(p.path_a, p.path_b) for p in allof.pairs] == [
            ("app/alpha.py", "app/beta.py")], (
            "the same file pair, however the scope is spelled")
        assert allof.pairs[0].shared_chunks == 2, (
            "alpha and beta share two chunks in reality; scanning three spellings "
            f"of one repo must not report {allof.pairs[0].shared_chunks}")

    def test_the_min_shared_chunks_gate_still_suppresses_one_chunk_noise(self, store):
        """gamma shares exactly ONE chunk with alpha — the measured noise shape."""
        self._seed_every_spelling(store)
        result = scan_duplicates(store, owner_user_id=OWNER, project=self.ALIASES)
        reported = {(p.path_a, p.path_b) for p in result.pairs}
        assert ("app/alpha.py", "app/gamma.py") not in reported, (
            "a single shared chunk seen under three spellings must not be "
            "promoted past min_shared_chunks=2")

    def test_examples_are_distinct_locations_not_the_same_one_repeated(self, store):
        self._seed_every_spelling(store)
        result = scan_duplicates(store, owner_user_id=OWNER, project=self.ALIASES,
                                 max_examples=3)
        examples = result.pairs[0].examples
        rendered = [
            (e.a.path, e.a.start_line, e.a.end_line, e.b.path, e.b.start_line, e.b.end_line)
            for e in examples
        ]
        assert len(rendered) == len(set(rendered)), (
            f"examples repeat the same citation: {rendered}")

    def test_scanned_totals_describe_the_repo_not_the_spellings(self, store):
        self._seed_every_spelling(store)
        one = scan_duplicates(store, owner_user_id=OWNER, project=self.ALIASES[0])
        allof = scan_duplicates(store, owner_user_id=OWNER, project=self.ALIASES)
        assert allof.scanned_files == one.scanned_files
        assert allof.scanned_chunks == one.scanned_chunks, (
            "scanning three spellings of one repo must not treble the chunk count, "
            "which is what pushed a 606-file scope into the 20,000-chunk cap")
