"""scripts/split_papers_corpus.py end to end on a real Qdrant.

A scratch `papers` holding formal papers, preprints and reviews (with TurboQuant,
on-disk vectors and keyword/integer payload indexes like production) is split:
the copies get the source config and indexes, an interrupted copy resumes from its
checkpoint, verify catches a missing or altered copy, and the irreversible delete
refuses without the split layout / confirmation, then removes only the legacy
copies. Scratch names stand in for the three. Skipped unless a Qdrant is reachable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from core.storage import corpus_layout  # noqa: E402
from core.storage.corpus_layout import CorpusLayout, write_layout  # noqa: E402
from scripts import split_papers_corpus as split  # noqa: E402
from tests.qdrant_scratch import drop, scratch_names  # noqa: E402

LOCAL_URL = "http://127.0.0.1:6333"
NAMES: dict[str, str] = {}  # fresh per test (see tests/qdrant_scratch.py)
DIM = 8
FORMAL = {"paper_id": "iclr-2025-1", "venue": "ICLR", "venue_tier": "CCF-A",
          "source_type": "paper", "year": 2025, "level": 2}
PREPRINT = {"paper_id": "arxiv-2609.1", "venue": "arXiv", "venue_tier": "preprint",
            "source_type": "paper", "year": 2026, "level": 2}
REVIEW = {"paper_id": "iclr-2025-1", "venue": "ICLR", "venue_tier": "CCF-A",
          "source_type": "review", "year": 2025, "level": 2}
# 5 formal, 7 preprints, 3 reviews; ids spread so random verify windows hit them.
SOURCE = ([(1000 + i, FORMAL) for i in range(5)]
          + [(i * 2 ** 59 + 7, PREPRINT) for i in range(7)]
          + [(5000 + i, REVIEW) for i in range(3)])


def _vector(pid: int) -> dict:
    return {"dense": [((pid % 97) + k) / 100 for k in range(DIM)],
            "sparse": qmodels.SparseVector(indices=[pid % 50, 60], values=[0.5, 0.25])}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    try:
        client = QdrantClient(url=LOCAL_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"no local Qdrant on {LOCAL_URL}: {exc}")
    NAMES.update(scratch_names("split"))
    monkeypatch.setattr(corpus_layout, "QDRANT_PAPERS_COLLECTION", NAMES["papers"])
    monkeypatch.setattr(corpus_layout, "QDRANT_ARXIV_COLLECTION", NAMES["arxiv"])
    monkeypatch.setattr(corpus_layout, "QDRANT_REVIEWS_COLLECTION", NAMES["reviews"])
    monkeypatch.setenv("QDRANT_CORPUS_LAYOUT_FILE", str(tmp_path / "corpus_layout.json"))
    rest = split.Rest(LOCAL_URL)
    rest.call("PUT", f"/collections/{NAMES['papers']}", {
        "vectors": {"dense": {"size": DIM, "distance": "Cosine", "on_disk": True,
                              "hnsw_config": {"m": 16, "ef_construct": 100, "on_disk": True}}},
        "sparse_vectors": {"sparse": {"index": {"on_disk": True}}},
        "on_disk_payload": True,
        "quantization_config": {"turbo": {"bits": "bits2", "always_ram": True}},
    })
    for field in ("venue_tier", "source_type", "paper_id", "venue"):
        client.create_payload_index(NAMES["papers"], field, qmodels.KeywordIndexParams(
            type=qmodels.KeywordIndexType.KEYWORD, on_disk=True), wait=True)
    client.create_payload_index(NAMES["papers"], "year", qmodels.PayloadSchemaType.INTEGER,
                                wait=True)
    client.upsert(NAMES["papers"], wait=True, points=[
        qmodels.PointStruct(id=pid, vector=_vector(pid), payload={**payload, "doc_id": str(pid)})
        for pid, payload in SOURCE])
    try:
        yield client
    finally:
        drop(client, NAMES.values())


def _ids(client, name, flt=None) -> list[int]:
    points, _ = client.scroll(name, scroll_filter=flt, limit=100)
    return sorted(p.id for p in points)


def test_create_reproduces_the_source_config_and_indexes(client):
    rest = split.Rest(LOCAL_URL)
    split.create(rest, NAMES, max_segment_size_kb=500000, defer_indexing=True,
                 quantized_on_disk=True)
    split.create(rest, NAMES, max_segment_size_kb=500000, defer_indexing=True)  # idempotent
    source = rest.call("GET", f"/collections/{NAMES['papers']}")
    for role in split.ROLES:
        info = rest.call("GET", f"/collections/{NAMES[role]}")
        params = info["config"]["params"]
        assert params["vectors"] == source["config"]["params"]["vectors"]
        assert params["sparse_vectors"] == source["config"]["params"]["sparse_vectors"]
        assert params["on_disk_payload"] is True
        assert info["config"]["quantization_config"] == {
            "turbo": {"bits": "bits2", "always_ram": False}}
        assert info["config"]["optimizer_config"]["indexing_threshold"] == 0
        assert info["config"]["optimizer_config"]["max_segment_size"] == 500000
        def schema(c):
            return {k: (v["data_type"], v.get("params")) for k, v in c["payload_schema"].items()}
        assert schema(info) == schema(source)
    split.enable_indexing(rest, NAMES, 20000)
    assert rest.call("GET", f"/collections/{NAMES['arxiv']}")["config"][
        "optimizer_config"]["indexing_threshold"] == 20000


def test_copy_resumes_verifies_and_delete_removes_only_the_copies(client, tmp_path):
    split.create(split.Rest(LOCAL_URL), NAMES, max_segment_size_kb=None,
                 defer_indexing=False)
    state = tmp_path / "state.json"
    # Interrupted after two batches of 3, then resumed from the checkpoint.
    partial = split.copy(client, NAMES, state, batch=3, pause_s=0, min_mem_gb=0,
                         max_batches=2)
    assert partial["arxiv"]["copied"] == 6 and not partial["arxiv"]["done"]
    assert split.verify(client, NAMES, sample=0)  # incomplete copy is reported
    done = split.copy(client, NAMES, state, batch=3, pause_s=0, min_mem_gb=0)
    assert done["arxiv"] == {"offset": None, "copied": 7, "done": True}
    assert done["reviews"]["copied"] == 3 and done["reviews"]["done"]

    preprints = [pid for pid, p in SOURCE if p is PREPRINT]
    reviews = [pid for pid, p in SOURCE if p is REVIEW]
    assert _ids(client, NAMES["arxiv"]) == sorted(preprints)
    assert _ids(client, NAMES["reviews"]) == sorted(reviews)
    assert split.verify(client, NAMES, sample=20) == []

    # An altered copy is caught point by point.
    client.set_payload(NAMES["arxiv"], payload={"venue": "tampered"},
                       points=[preprints[0]], wait=True)
    assert any("payload differs" in p for p in split.verify(client, NAMES, sample=20))
    client.set_payload(NAMES["arxiv"], payload={"venue": "arXiv"},
                       points=[preprints[0]], wait=True)

    # The irreversible step refuses under the mixed layout and without confirmation.
    assert split.main(["--url", LOCAL_URL, "delete-legacy", "--confirm-irreversible"]) == 2
    write_layout(CorpusLayout(split=True, papers_legacy_mixed=True),
                 tmp_path / "corpus_layout.json")
    assert split.main(["--url", LOCAL_URL, "delete-legacy"]) == 2
    assert len(_ids(client, NAMES["papers"])) == len(SOURCE)

    assert split.main(["--url", LOCAL_URL, "delete-legacy", "--confirm-irreversible",
                       "--batch", "2", "--pause-ms", "0",
                       "--min-mem-available-gb", "0"]) == 0
    assert _ids(client, NAMES["papers"]) == sorted(pid for pid, p in SOURCE if p is FORMAL)
    assert _ids(client, NAMES["arxiv"]) == sorted(preprints)  # copies untouched


def test_delete_refuses_a_batch_whose_copies_are_missing(client):
    split.create(split.Rest(LOCAL_URL), NAMES, max_segment_size_kb=None,
                 defer_indexing=False)
    with pytest.raises(RuntimeError, match="lacks copies"):
        split.delete_legacy(client, NAMES, batch=50, pause_s=0, min_mem_gb=0)
    assert len(_ids(client, NAMES["papers"])) == len(SOURCE)


def test_layout_subcommand_writes_the_runtime_file(client, tmp_path):
    assert split.main(["--url", LOCAL_URL, "layout", "split"]) == 0
    assert corpus_layout.current_layout() == CorpusLayout(split=True, papers_legacy_mixed=True)
    assert split.main(["--url", LOCAL_URL, "layout", "split", "--clean"]) == 0
    assert corpus_layout.current_layout() == CorpusLayout(split=True, papers_legacy_mixed=False)


def test_memory_guard_waits_until_memory_returns():
    readings = iter([1.0, 1.5, 3.0])
    split.wait_for_memory(2.0, sleep_s=0, probe=lambda: next(readings))
    assert next(readings, None) is None  # stopped at the first reading above 2G
