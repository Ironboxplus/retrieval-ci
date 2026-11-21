"""Indexing must not hold every chunk's vectors at once.

2026-09-04 incident. An ``index_code`` of E:/Go/Openlist with its nested
repositories reached ~295,000 chunks and systemd-oomd killed the whole backend
mid-request (``status=9/KILL``), taking 17 minutes of unrelated work with it.
No access line was ever logged, because uvicorn writes that after the response.

The dominant term was not the request body. ``_embed`` returned
``[v.tolist() for v in res.dense]`` — numpy float32 rows converted to Python
lists of Python floats — and ``_embed_batches`` accumulated **all** of them
before the first upsert. Measured on real arrays at dim=1024:

    list[list[float]]   41,316 B/chunk      295,000 chunks -> 11.35 GiB
    numpy float32        4,096 B/chunk      295,000 chunks ->  1.13 GiB
    one EMBED_BATCH                                        -> 20.2 MiB

against MemoryMax=5GiB with the two models already resident at ~2.5GiB.

So the property under test is not "is it fast" or "does it use numpy" — it is
that the vectors in flight are bounded by the batch rather than by the
repository. Embedding a batch and upserting it immediately makes the array
library irrelevant, because nothing survives the loop iteration.

Real Qdrant on loopback, private scratch collection, real (injected) embed
function. Skipped unless Qdrant is reachable.
"""
from __future__ import annotations

import tracemalloc
import warnings

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.services.code_kb.schemas import CodeChunkIn, CodeFileIn, CodeIndexRequest  # noqa: E402
from app.services.code_kb.service import EMBED_BATCH, CodeKBService  # noqa: E402
from app.services.code_kb.vector_store import CodeVectorStore  # noqa: E402
from core.storage.qdrant_store import QdrantStore  # noqa: E402

# Loopback only and hardcoded, never from settings/QDRANT_URL, so a deploy
# host's config cannot aim this suite at the 2.2M-point production collection.
LOCAL_URL = "http://127.0.0.1:6333"
COLLECTION = "code_chunks_streaming_test"
OWNER = "streaming-owner"
PROJECT = "/repo"

# The real embedding width. The whole effect being measured is per-float object
# overhead, so a toy dimension would erase the very signal under test.
DIM = 1024


def _drop_collection(client: QdrantClient, attempts: int = 4):
    """Windows intermittently 500s here while segment files are still mapped."""
    import time
    failure = None
    for attempt in range(attempts):
        try:
            client.delete_collection(COLLECTION)
            return None
        except Exception as exc:  # noqa: BLE001
            failure = exc
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
    return failure


@pytest.fixture()
def store():
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=30)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")

    # Setup is the authoritative cleanup: a crashed run leaves the collection
    # registered, and on Windows a teardown can unregister it but leave
    # ./storage/collections/<name> behind, after which create returns 400
    # "Collection data already exists" forever.
    _drop_collection(client)

    from config import settings
    params = qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": params} if settings.embedding.supports_sparse else params,
    )

    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name = COLLECTION
    inner.client = client
    cvs = CodeVectorStore(collection_name=COLLECTION, store=inner)
    cvs.ensure_search_payload_indexes()
    try:
        yield cvs
    finally:
        failure = _drop_collection(client, attempts=3)
        if failure is not None:
            warnings.warn(f"could not drop `{COLLECTION}` in teardown ({failure}); "
                          "the next run's setup will clear it", stacklevel=1)


class _RecordingStore:
    """Delegates every call to the REAL store while recording upsert order.

    Not a stand-in for Qdrant: every point genuinely lands in the collection.
    It exists only so a test can see WHEN the writes happen relative to the
    embed calls, which is the difference between streaming and accumulating.
    """

    def __init__(self, inner: CodeVectorStore, events: list[str]):
        self._inner = inner
        self._events = events
        self.upsert_calls: list[int] = []
        self.owner_file_scans = 0

    def upsert(self, **kwargs):
        self._events.append("upsert")
        self.upsert_calls.append(len(kwargs["items"]))
        return self._inner.upsert(**kwargs)

    def list_owner_files(self, **kwargs):
        self.owner_file_scans += 1
        return self._inner.list_owner_files(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _service(store: CodeVectorStore, events: list[str]) -> tuple[CodeKBService, _RecordingStore]:
    def embed(texts):
        events.append("embed")
        # A real deterministic embedding: unit-ish vectors derived from the text,
        # returned as the same list[list[float]] shape production returns.
        out = []
        for text in texts:
            seed = float(len(text) % 7 + 1)
            row = [0.0] * DIM
            row[hash(text) % DIM] = seed
            out.append(row)
        return out

    recording = _RecordingStore(store, events)
    return CodeKBService(embed_fn=embed, vector_store=recording), recording


def _request(*, n_chunks: int, files: int = 40, project: str = PROJECT) -> CodeIndexRequest:
    # Ceiling, so the requested total is produced exactly; floor division silently
    # dropped the remainder and made the count assertions compare the wrong number.
    per_file = max(1, -(-n_chunks // files))
    payload_files = []
    made = 0
    for f in range(files):
        chunks = []
        for c in range(per_file):
            if made >= n_chunks:
                break
            # Deliberately short text: the term under test is the VECTOR
            # accumulation, so chunk text must not dominate the measurement.
            body = f"def f{f}_{c}(): return {c}"
            chunks.append(CodeChunkIn(start_line=c * 2, end_line=c * 2 + 1,
                                      label=f"f{f}_{c}", content=body, code=body))
            made += 1
        if chunks:
            payload_files.append(CodeFileIn(rel_path=f"src/m{f}.py", language="python",
                                            chunks=chunks))
    return CodeIndexRequest(project=project, host="testhost", root_dir=project,
                            files=payload_files)


class TestVectorsInFlightAreBoundedByTheBatch:

    def test_upserts_interleave_with_embedding_instead_of_waiting_for_all_of_it(self, store):
        """The structural signature of streaming, independent of any measurement.

        Accumulating gives embed*N then a single upsert. Streaming gives them
        interleaved, so a batch's vectors can be released before the next batch
        is produced.
        """
        events: list[str] = []
        svc, recording = _service(store, events)
        n = EMBED_BATCH * 5 + 7

        svc.index(_request(n_chunks=n), owner_user_id=OWNER)

        assert events.count("embed") >= 5, events[:12]
        first_upsert = events.index("upsert")
        last_embed = len(events) - 1 - events[::-1].index("embed")
        assert first_upsert < last_embed, (
            "every upsert happened after all embedding finished, so all vectors "
            f"were held at once: {events[:8]} ... {events[-4:]}")
        assert sum(recording.upsert_calls) == n
        # A batch closes at the first file that takes it to EMBED_BATCH and never
        # splits that file, so it can overshoot by up to one file's chunks. The
        # bound that matters is that it does not scale with the repository.
        biggest_file = max(len(f.chunks) for f in _request(n_chunks=n).files)
        assert max(recording.upsert_calls) <= EMBED_BATCH + biggest_file, (
            f"a single upsert carried {max(recording.upsert_calls)} chunks; the "
            f"bound is EMBED_BATCH({EMBED_BATCH}) + one file ({biggest_file})")

    def test_peak_python_memory_is_bounded_by_the_batch_not_the_repo(self, store):
        """The property the incident was actually about.

        8,000 chunks accumulated as list[list[float]] at dim=1024 is ~330 MiB.
        One EMBED_BATCH is ~5 MiB. The bound below sits far from both, so this
        cannot fail on allocator noise — only on a design that accumulates.
        """
        events: list[str] = []
        svc, _ = _service(store, events)
        request = _request(n_chunks=8000, files=80)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            svc.index(request, owner_user_id=OWNER)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        accumulated = 8000 * 41316
        assert peak < 80 * 1024 * 1024, (
            f"peak python allocation {peak/2**20:.0f} MiB; accumulating all "
            f"8,000 chunks would cost about {accumulated/2**20:.0f} MiB, so the "
            "vectors are still being held for the whole repository")

    def test_a_bigger_repo_does_not_raise_the_peak_proportionally(self, store):
        """Bounded means bounded: 3x the chunks must not cost ~3x the peak."""
        events: list[str] = []
        svc, _ = _service(store, events)

        def peak_for(n: int, project: str) -> int:
            request = _request(n_chunks=n, files=40, project=project)
            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                svc.index(request, owner_user_id=OWNER)
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        small = peak_for(2000, "/repo/small")
        large = peak_for(6000, "/repo/large")
        # Chunk text still scales with the payload, so allow generous slack; an
        # accumulating implementation would land near 3x.
        assert large < small * 2, (
            f"peak grew {large/small:.1f}x for 3x the chunks "
            f"({small/2**20:.0f} MiB -> {large/2**20:.0f} MiB)")


class TestCrossScopeDiscoveryDoesNotRepeatPerBatch:
    """Streaming must not turn a once-per-job scan into a once-per-batch scan.

    `_fan_out_sync` starts by calling `list_owner_files`, which enumerates EVERY
    file this owner has indexed across ALL scopes — on the production owner that
    is 150+ scopes. Under the old design fan-out ran once at the end of the job,
    so that cost was paid once. Moving fan-out inside the batch loop silently
    multiplied it by the batch count: 27 seconds of fan-out for 3 synced files on
    a 209-chunk job (2026-09-04, observed in production), and ~2,300 scans for the
    295,000-chunk repository the streaming work was written for — far worse than
    the memory problem it replaced.

    The snapshot is safe to reuse: `_fan_out_sync` already re-reads each target's
    point ids under that target's own lock and skips a file whose points are gone,
    precisely because discovery was always allowed to be stale.
    """

    def test_the_owner_wide_scan_happens_once_no_matter_how_many_batches(self, store):
        events: list[str] = []
        svc, recording = _service(store, events)
        # A second scope for the same owner, so fan-out has something to discover
        # and cannot exit early on an empty candidate set.
        other = _request(n_chunks=30, files=3, project="/repo/vendored")
        svc.index(other, owner_user_id=OWNER)
        recording.owner_file_scans = 0

        n = EMBED_BATCH * 6 + 5
        svc.index(_request(n_chunks=n, files=40), owner_user_id=OWNER)

        assert len(recording.upsert_calls) >= 6, (
            f"expected several batches, got {len(recording.upsert_calls)}")
        assert recording.owner_file_scans <= 1, (
            f"list_owner_files ran {recording.owner_file_scans} times for "
            f"{len(recording.upsert_calls)} batches; it enumerates every file in "
            "every scope this owner has and must not scale with batch count")


class TestBatchingDoesNotChangeWhatGetsIndexed:
    """Streaming changed WHEN vectors are written, not WHAT is written.

    The concern is fair on its face — the outer grouping changed from a flat
    128-chunk window across file boundaries to whole files grouped to >=128 — so
    it is worth proving rather than arguing. Two reasons it cannot matter:

    * a chunk's vector depends only on its own text. Both embedders call
      ``self._model.encode(texts, batch_size=settings.embedding.batch_size)``,
      so FlagEmbedding re-slices whatever list it is handed into its OWN batches;
      the outer grouping never reaches the model. ``normalize_embeddings=True``
      normalises each vector individually, not across a batch.
    * ``point_id`` is derived from (owner, project, rel_path, start_line, label,
      chunk_ordinal), none of which depend on batch membership.

    The test below is the empirical form: index identical content under two very
    different batch sizes and require the stored points to match exactly.
    """

    def test_two_different_batch_sizes_produce_an_identical_index(self, store, monkeypatch):
        import app.services.code_kb.service as service_module

        def index_with(batch: int, project: str) -> dict:
            monkeypatch.setattr(service_module, "EMBED_BATCH", batch)
            svc, _ = _service(store, [])
            svc.index(_request(n_chunks=900, files=15, project=project),
                      owner_user_id=OWNER)
            points, _ = store.store.client.scroll(
                COLLECTION, limit=4000, with_payload=True, with_vectors=True)
            out = {}
            for p in points:
                payload = p.payload or {}
                if payload.get("project") != project:
                    continue
                vector = p.vector
                if isinstance(vector, dict):
                    vector = vector.get("dense")
                key = (payload.get("path"), payload.get("start_line"),
                       payload.get("label"), payload.get("chunk_ordinal"))
                out[key] = (tuple(vector), payload.get("content"), payload.get("code"))
            return out

        tiny = index_with(7, "/repo/batch7")
        huge = index_with(4096, "/repo/batch4096")

        # Non-vacuity: 128 chunks per call vs 7 vs 4096 are three genuinely
        # different groupings, and all 900 chunks have to be in each result.
        assert len(tiny) == 900, f"batch=7 indexed {len(tiny)} of 900 chunks"
        assert len(huge) == 900, f"batch=4096 indexed {len(huge)} of 900 chunks"
        assert set(tiny) == set(huge), (
            "different chunks were indexed: "
            f"only-in-7={sorted(set(tiny) - set(huge))[:3]} "
            f"only-in-4096={sorted(set(huge) - set(tiny))[:3]}")
        differing = [k for k in tiny if tiny[k] != huge[k]]
        assert not differing, (
            f"{len(differing)} chunks got different vectors or content depending "
            f"on batch size, e.g. {differing[:3]}")


class TestStreamingKeepsTheDeleteInvariants:

    def test_nothing_is_deleted_when_embedding_fails_midway(self, store):
        """The half of invariant #3 that must survive streaming.

        Upserts now interleave with embedding, so a mid-embed crash can leave
        some files updated — but it must never leave the project DELETED and
        not rebuilt. Stale-point pruning stays after all embedding completes.
        """
        events: list[str] = []
        svc, _ = _service(store, events)
        svc.index(_request(n_chunks=EMBED_BATCH * 2, files=8), owner_user_id=OWNER)
        before = svc.vector_store.list_paths(owner_user_id=OWNER, project=PROJECT)
        assert before, "seed index produced nothing"

        calls = {"n": 0}

        def failing_embed(texts):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("embedder died mid-run")
            return [[0.0] * DIM for _ in texts]

        broken = CodeKBService(embed_fn=failing_embed, vector_store=svc.vector_store)
        with pytest.raises(RuntimeError):
            broken.index(_request(n_chunks=EMBED_BATCH * 6, files=30),
                         owner_user_id=OWNER)

        after = svc.vector_store.list_paths(owner_user_id=OWNER, project=PROJECT)
        assert before <= after, (
            f"a failed run removed indexed paths: missing {sorted(before - after)[:5]}")

    def test_a_completed_stream_still_prunes_stale_points(self, store):
        """Shrinking a file must leave no orphan chunks behind."""
        events: list[str] = []
        svc, _ = _service(store, events)
        svc.index(_request(n_chunks=600, files=6), owner_user_id=OWNER)
        wide = len(svc.vector_store.list_point_ids_for_paths(
            owner_user_id=OWNER, project=PROJECT))

        svc.index(_request(n_chunks=60, files=6), owner_user_id=OWNER)
        narrow = len(svc.vector_store.list_point_ids_for_paths(
            owner_user_id=OWNER, project=PROJECT))

        assert narrow < wide, f"stale points survived the shrink: {wide} -> {narrow}"
        assert narrow == 60, f"expected exactly the 60 current chunks, got {narrow}"
