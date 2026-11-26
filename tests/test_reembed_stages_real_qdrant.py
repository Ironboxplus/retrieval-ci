"""The re-embed pipeline's stages against real Qdrant, real SQLite and the real tokenizer.

Scratch collections with unique names on loopback Qdrant; chunk sizes are real token counts
(``text_of_tokens``); the stored "old" vectors are arbitrary unit vectors. The model enters only
through the ``encode`` boundary (a deterministic hash function here, so identical text gets an
identical vector; the real model runs in test_reembed_pipeline_real_model.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qdrant_client.http import models as qm

from app.services.pipeline import stages as st
from app.services.pipeline.reembed_pipeline import ReembedContext, build_reembed_pipeline
from core.indexing.embedding_profile import count_tokens, current_embed_fp, get_profile, load_tokenizer
from tests.embedding_model_support import bge_m3_snapshot, text_of_tokens
from tests.reembed_support import (
    DIM, create_scratch_collection, drop_collection, new_store, qdrant_client, read_chunk_shard,
    stored_point, write_host_registry,
)

CODE_INDEXES = ("owner_user_id", "project", "path", "language", "embed_fp")
JOURNAL_INDEXES = ("project_slug", "owner_user_id", "entry_id", "source_kind", "client", "embed_fp")
PAPER_INDEXES = ("source_type", "embed_fp", "doc_id")
FP = current_embed_fp()


@pytest.fixture(autouse=True)
def tokenizer_env(monkeypatch):
    monkeypatch.setenv("BGE_M3_PATH", bge_m3_snapshot())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def det_encode(texts):
    """Deterministic stand-in for the model: a unit vector and a sparse vector from the text."""
    dense, sparse = [], []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        vec = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        dense.append(vec / np.linalg.norm(vec))
        sparse.append({seed % 4000 + j: 0.25 + j / 10 for j in range(3)})
    return np.stack(dense) if dense else np.zeros((0, DIM), np.float32), sparse


class Env:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.client = qdrant_client()
        self.store, self.session_factory = new_store()
        self.collections: list[str] = []
        registry = tmp_path / "embed_hosts.json"
        write_host_registry(registry, workdir=tmp_path / "hostwork", python="python", worker=tmp_path / "w.py",
                            model_dir="m", shard_size=3)
        monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(registry))

    def collection(self, prefix, indexes):
        name = create_scratch_collection(self.client, prefix, keyword_indexes=indexes)
        self.collections.append(name)
        return name

    def ctx(self, scope: str, collection: str, **kw) -> ReembedContext:
        task_id = self.store.create_task(task_type="reembed_pipeline")
        kw.setdefault("execution_mode", "host")
        kw.setdefault("ssh_target", "local" if kw["execution_mode"] == "local" else "shim-host")
        return ReembedContext(
            venue=scope, year=0, store=self.store, task_id=task_id, project_root=self.tmp_path,
            scope_name=scope, collection=collection,
            client_factory=lambda: self.client, embed_fn=det_encode, pause_s=0.0, **kw)

    def close(self):
        for name in self.collections:
            drop_collection(self.client, name)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch)
    yield e
    e.close()


def stages(ctx):
    return {spec.key: spec for spec in build_reembed_pipeline(ctx)}


def run(spec, ctx):
    lines: list[str] = []
    spec.run(ctx, lambda m, level="info": lines.append(m))
    return lines


def embed_locally(ctx, encode=det_encode):
    """What the embed stage does on the host, done here with the real shard loop."""
    worker = st._embed_worker()
    assert worker.embed_shards(st._embed_chunks_dir(ctx), st._embed_vectors_dir(ctx), encode,
                               log=lambda m: None) is True


def code_payload(path: str, text: str, **extra) -> dict:
    return {"owner_user_id": "u", "project": "/repo", "host": "h", "path": path, "language": "python",
            "start_line": 1, "end_line": 9, "label": "f", "content": text, "code": text, **extra}


def shard_records(ctx) -> dict[int, dict]:
    out = {}
    for path in sorted(st._embed_chunks_dir(ctx).glob("chunks_*.jsonl.gz")):
        for rec in read_chunk_shard(path):
            out[rec["point_id"]] = rec
    return out


def select_dir(ctx) -> Path:
    return st._embed_task_dir(ctx) / "select"


def stamp_ids(ctx) -> set[int]:
    ids: set[int] = set()
    for path in sorted(select_dir(ctx).glob("stamp_*.txt")):
        ids.update(int(x) for x in path.read_text().split())
    return ids


def oversized(ctx) -> dict[int, dict]:
    path = select_dir(ctx) / "oversized.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["point_id"]: row for row in rows}


# ------------------------------------------------------------------ 1. select
SIZES = {1: 100, 2: 600, 3: 3000, 4: 9000}          # point id -> real token count of its text


def seed_sized_points(env, collection, make_payload):
    tokenizer = load_tokenizer(get_profile())
    texts = {pid: text_of_tokens(tokenizer, n) for pid, n in SIZES.items()}
    points = [stored_point(pid, make_payload(pid, texts[pid])) for pid in SIZES]
    # not stale: already under the current fingerprint (any size)
    points.append(stored_point(5, make_payload(5, texts[3], embed_fp=FP)))
    # stale by another fingerprint
    points.append(stored_point(6, make_payload(6, texts[1], embed_fp="0123456789abcdef")))
    env.client.upsert(collection, points=points, wait=True)
    return texts


@pytest.mark.parametrize("scope, chunks, stamps, big", [
    ("code_chunks:truncated", {2, 3}, set(), {4}),
    ("code_chunks:align", {1, 6}, set(), set()),
    ("public_code_chunks:truncated", {2, 3}, set(), {4}),
])
def test_select_writes_exactly_the_stale_points_by_real_token_count_code(env, scope, chunks, stamps, big):
    name = env.collection("sel_code", CODE_INDEXES)
    texts = seed_sized_points(env, name, lambda pid, text, **extra: code_payload(f"f{pid}.py", text, **extra))
    ctx = env.ctx(scope, name)

    run(stages(ctx)["select"], ctx)

    records = shard_records(ctx)
    assert set(records) == chunks
    for pid, rec in records.items():
        assert rec == {"point_id": pid, "content": texts[1 if pid == 6 else pid]}   # point 6 holds point 1's text
    assert stamp_ids(ctx) == stamps
    assert set(oversized(ctx)) == big
    for pid in big:                       # a code point keeps its full payload for the re-split
        assert oversized(ctx)[pid]["payload"]["content"] == texts[pid]
        assert oversized(ctx)[pid]["tokens"] == count_tokens([texts[pid]], get_profile())[0]
    assert (ctx.state_get("select_shard_start"), ctx.state_get("select_complete")) == (0, True)
    assert ctx.state_get("select_shard_end") == len(list(st._embed_chunks_dir(ctx).glob("chunks_*.jsonl.gz")))


def test_select_papers_scopes_use_the_source_type_filter_and_stamp_what_fits_2048(env):
    name = env.collection("sel_papers", PAPER_INDEXES)
    tokenizer = load_tokenizer(get_profile())
    texts = {pid: text_of_tokens(tokenizer, n) for pid, n in SIZES.items()}
    points = []
    for source_type, base in (("upload", 10), ("webpage", 20), ("paper", 30)):
        for pid, text in texts.items():
            points.append(stored_point(base + pid, {"source_type": source_type, "doc_id": "d", "content": text}))
    points.append(stored_point(40, {"source_type": "upload", "doc_id": "d", "content": texts[1], "embed_fp": FP}))
    env.client.upsert(name, points=points, wait=True)

    uploads = env.ctx("papers:uploads", name)
    run(stages(uploads)["select"], uploads)
    assert set(shard_records(uploads)) == {11, 12, 13, 21, 22, 23}      # every size up to 8192, both source types
    assert set(oversized(uploads)) == {14, 24} and stamp_ids(uploads) == set()

    pipeline = env.ctx("papers:pipeline", name)
    run(stages(pipeline)["select"], pipeline)
    assert stamp_ids(pipeline) == {31, 32}                              # <= 2048 tokens: only the fingerprint
    assert set(shard_records(pipeline)) == {33}                          # 2049..8192: embedded in full
    assert set(oversized(pipeline)) == {34}
    assert oversized(pipeline)[34]["payload"] is None                    # papers' chunks live in SQL: listed only
    counts = pipeline.state_get("select_counts")
    assert counts == {"stamp": 2, "chunk": 1, "oversized": 1, "skip": 0}


def test_select_resumes_from_its_checkpoint_and_a_completed_select_is_not_rerun(env):
    name = env.collection("sel_resume", CODE_INDEXES)
    tokenizer = load_tokenizer(get_profile())
    text = text_of_tokens(tokenizer, 600)
    env.client.upsert(name, points=[stored_point(i, code_payload(f"f{i}.py", text)) for i in range(1, 8)], wait=True)
    ctx = env.ctx("code_chunks:truncated", name)
    first = run(stages(ctx)["select"], ctx)
    assert ctx.state_get("select_complete") is True and len(shard_records(ctx)) == 7
    shards = sorted(st._embed_chunks_dir(ctx).glob("chunks_*.jsonl.gz"))
    assert len(shards) == 3                                            # registry shard size 3: 3 + 3 + 1
    before = [p.read_bytes() for p in shards]
    env.client.delete(name, points_selector=[1, 2, 3], wait=True)       # a rerun that scanned again would lose them

    again = run(stages(ctx)["select"], ctx)

    assert [p.read_bytes() for p in shards] == before and len(shard_records(ctx)) == 7
    assert any("already complete" in line for line in again) and first != again


# ------------------------------------------------------------------- 2. write
def seed_stale_code(env, name, count=5, tokens=600):
    tokenizer = load_tokenizer(get_profile())
    texts = {i: text_of_tokens(tokenizer, tokens) + f" unique{i}" for i in range(1, count + 1)}
    env.client.upsert(name, points=[stored_point(i, code_payload(f"f{i}.py", t)) for i, t in texts.items()], wait=True)
    return texts


def test_write_replaces_dense_and_sparse_and_stamps_the_fingerprint(env):
    name = env.collection("wr_code", CODE_INDEXES)
    texts = seed_stale_code(env, name)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)

    run(spec["write"], ctx)

    expected_dense, expected_sparse = det_encode([texts[i] for i in sorted(texts)])
    for i, point in zip(sorted(texts), env.client.retrieve(name, ids=sorted(texts), with_vectors=True,
                                                          with_payload=True)):
        assert point.id == i
        np.testing.assert_allclose(point.vector["dense"], expected_dense[i - 1], atol=2e-3)   # stored as float16
        assert dict(zip(point.vector["sparse"].indices, point.vector["sparse"].values)).keys() == expected_sparse[i - 1].keys()
        assert point.payload["embed_fp"] == FP
        assert point.payload["content"] == texts[i] and point.payload["path"] == f"f{i}.py"   # payload kept


def test_a_point_reindexed_by_a_client_between_select_and_write_is_not_overwritten(env):
    name = env.collection("wr_race", CODE_INDEXES)
    texts = seed_stale_code(env, name, count=4)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)

    # A client re-indexes point 2 with the current profile after select ran; point 3 is deleted.
    client_vector = stored_point(2, code_payload("f2.py", texts[2], embed_fp=FP), seed=999)
    env.client.upsert(name, points=[client_vector], wait=True)
    env.client.delete(name, points_selector=[3], wait=True)

    lines = run(spec["write"], ctx)

    kept = env.client.retrieve(name, ids=[2], with_vectors=True, with_payload=True)[0]
    np.testing.assert_allclose(kept.vector["dense"], client_vector.vector["dense"], atol=1e-6)   # untouched
    assert kept.payload["embed_fp"] == FP
    assert env.client.retrieve(name, ids=[3]) == []                                              # not resurrected
    for i in (1, 4):
        point = env.client.retrieve(name, ids=[i], with_vectors=True, with_payload=True)[0]
        assert point.payload["embed_fp"] == FP
        np.testing.assert_allclose(point.vector["dense"], det_encode([texts[i]])[0][0], atol=2e-3)
    assert any("skipped" in line and "missing=1" in line for line in lines), lines


def test_write_is_resumable_per_parquet(env):
    name = env.collection("wr_resume", CODE_INDEXES)
    seed_stale_code(env, name, count=5)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)
    run(spec["write"], ctx)
    again = run(spec["write"], ctx)
    assert any("already written" in line for line in again)
    assert env.client.count(name, exact=True).count == 5


# ----------------------------------------------------------------- 3. journal
def seed_journal(env, name):
    """Two live entries as the OLD chunker left them: one with a single 9,000-token chunk."""
    from app.services.journal.repository import JournalRepository, new_chunk_id
    from app.services.journal.vector_store import _point_id

    repo = JournalRepository(env.session_factory)
    tokenizer = load_tokenizer(get_profile())
    long_body = text_of_tokens(tokenizer, 9000)
    entries = {}
    for label, body in (("long", long_body), ("short", "A short note about the embedding fingerprint.")):
        entry, project, _ = repo.create_entry(
            project_slug="/proj", client="mcp", source_kind="doc", title=f"notes/{label}.md", summary=None,
            body=body, tags=[], metadata={}, owner_user_id="u1")
        chunk_id = new_chunk_id(entry.entry_id, 1, 2, body)
        repo.replace_chunks(entry.entry_id, [
            {"chunk_id": chunk_id, "chunk_index": 1, "level": 2, "heading": None, "content": body,
             "token_count": 9000, "metadata": {}}])
        repo.mark_index_result(entry.entry_id, success=True)
        env.client.upsert(name, points=[stored_point(_point_id(chunk_id), {
            "entry_id": entry.entry_id, "chunk_id": chunk_id, "content": body, "project_slug": "/proj"})], wait=True)
        entries[label] = (entry.entry_id, _point_id(chunk_id))
    return repo, entries


def test_journal_entry_points_equal_what_the_live_write_path_stores(env):
    """``journal_entry_points`` is the live write path's own chunk/payload/id code."""
    from app.services.journal.chunker import JournalChunker
    from app.services.journal.repository import JournalRepository
    from app.services.journal.schemas import JournalRecordRequest
    from app.services.journal.service import JournalService, journal_entry_points
    from app.services.journal.vector_store import JournalVectorStore
    from core.db.models import JournalEntry
    from core.storage.qdrant_store import QdrantStore

    name = env.collection("jr_live", JOURNAL_INDEXES)
    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name, inner.client = name, env.client

    def embed(texts, include_sparse, *, priority="interactive"):
        dense, sparse = det_encode(list(texts))
        return [row.tolist() for row in dense], sparse

    repo = JournalRepository(env.session_factory)
    service = JournalService(repository=repo, chunker=JournalChunker(),
                             vector_store=JournalVectorStore(collection_name=name, store=inner), embed_fn=embed)
    tokenizer = load_tokenizer(get_profile())
    body = "\n".join(["# Design", text_of_tokens(tokenizer, 3000), "", "## Notes", "short section"])
    result = service.record(JournalRecordRequest(
        project_slug="/proj", client="mcp", source_kind="doc", title="notes/design.md", summary="s",
        body=body, tags=["a", "b"], owner_user_id="u1", dedupe_policy="append"))
    assert result.indexed

    stored = {p.id: p.payload for p in env.client.scroll(name, limit=1000, with_payload=True)[0]}
    with repo.session() as session:
        entry = session.query(JournalEntry).filter_by(entry_id=result.entry.entry_id).one()
        _, points = journal_entry_points(entry)
    assert len(points) == len(stored) >= 3
    assert {pid: payload for pid, payload, _ in points} == stored
    assert sorted(text for _, _, text in points) == sorted(p["content"] for p in stored.values())


def test_journal_rebuild_replaces_oversized_chunks_and_follows_in_sql(env):
    from app.services.journal.service import journal_entry_points
    from core.db.models import JournalChunk, JournalEntry

    name = env.collection("jr", JOURNAL_INDEXES)
    repo, entries = seed_journal(env, name)
    ctx = env.ctx("journal_chunks:rebuild", name, journal_repository=repo)
    spec = stages(ctx)

    run(spec["select"], ctx)
    embed_locally(ctx)
    run(spec["write"], ctx)

    profile = get_profile()
    points = env.client.scroll(name, limit=1000, with_payload=True)[0]
    assert points and all(p.payload["embed_fp"] == FP for p in points)
    assert max(count_tokens([p.payload["content"] for p in points], profile)) <= profile.chunk_max_tokens
    long_id, long_old_point = entries["long"]
    short_id, short_old_point = entries["short"]
    assert env.client.retrieve(name, ids=[long_old_point]) == []              # old: one 9,000-token chunk
    # The short entry's body chunk is unchanged, so it keeps its id: replaced in place, not deleted.
    kept = env.client.retrieve(name, ids=[short_old_point], with_payload=True, with_vectors=True)
    assert len(kept) == 1 and kept[0].payload["embed_fp"] == FP
    np.testing.assert_allclose(kept[0].vector["dense"], det_encode([kept[0].payload["content"]])[0][0], atol=2e-3)
    for entry_id in (long_id, short_id):
        with repo.session() as session:
            entry = session.query(JournalEntry).filter_by(entry_id=entry_id).one()
            _, expected = journal_entry_points(entry)
            sql_rows = session.query(JournalChunk).filter_by(entry_id=entry_id).all()
            assert {pid for pid, _, _ in expected} == {p.id for p in points if p.payload["entry_id"] == entry_id}
            assert {r.chunk_id for r in sql_rows} == {p["chunk_id"] for _, p, _ in expected}
            assert max(r.token_count for r in sql_rows) <= profile.chunk_max_tokens
    assert spec["verify"].gate(ctx).ok


def test_journal_select_lists_the_old_point_ids_and_skips_pending_entries(env):
    name = env.collection("jr_sel", JOURNAL_INDEXES)
    repo, entries = seed_journal(env, name)
    repo.mark_index_result(entries["short"][0], success=False, error="down")      # pending: not indexed
    ctx = env.ctx("journal_chunks:rebuild", name, journal_repository=repo)

    run(stages(ctx)["select"], ctx)

    old_ids = {int(x) for x in (select_dir(ctx) / "journal_old_ids.txt").read_text().split()}
    assert old_ids == {entries["long"][1]}                                         # the pending entry is left alone
    assert ctx.state_get("select_counts")["pending_skipped"] == 1
    assert all(rec["content"] and json.loads(rec["payload_json"])["entry_id"] == entries["long"][0]
               for rec in shard_records(ctx).values())


def test_journal_keyword_search_and_reindex_pending_see_only_the_rebuilt_chunks(env):
    from app.services.journal.chunker import JournalChunker
    from app.services.journal.schemas import JournalSearchRequest
    from app.services.journal.service import JournalService
    from app.services.journal.vector_store import JournalVectorStore
    from core.db.models import JournalChunk
    from core.storage.qdrant_store import QdrantStore

    name = env.collection("jr_sql", JOURNAL_INDEXES)
    repo, entries = seed_journal(env, name)
    long_id, long_old_point = entries["long"]
    with repo.session() as session:
        old_chunk_ids = {r.chunk_id for r in session.query(JournalChunk).filter_by(entry_id=long_id)}
    assert len(old_chunk_ids) == 1

    ctx = env.ctx("journal_chunks:rebuild", name, journal_repository=repo)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)
    run(spec["write"], ctx)

    inner = QdrantStore.__new__(QdrantStore)
    inner.collection_name, inner.client = name, env.client

    def embed(texts, include_sparse, *, priority="interactive"):
        dense, sparse = det_encode(list(texts))
        return [row.tolist() for row in dense], sparse

    service = JournalService(repository=repo, chunker=JournalChunker(),
                             vector_store=JournalVectorStore(collection_name=name, store=inner), embed_fn=embed)
    new_chunk_ids = {c.chunk_id for c in repo.get_entry(long_id).chunks}
    assert len(new_chunk_ids) >= 3 and not (new_chunk_ids & old_chunk_ids)
    limit = get_profile().chunk_max_tokens

    # Keyword search: the entry is found through its body, and both chunk paths (all chunks of the
    # entry, and the per-hit chunk_limit that matches chunk text) return only rebuilt chunks.
    for chunk_limit in (None, 20):
        response = service.search(JournalSearchRequest(
            query="retrieval", project_slug="/proj", owner_user_id="u1", mode="keyword", top_k=10,
            include_chunks=True, chunk_limit=chunk_limit))
        hit = next(h for h in response.hits if h.entry.entry_id == long_id)
        assert hit.chunks and {c.chunk_id for c in hit.chunks} <= new_chunk_ids, chunk_limit
        assert all((c.token_count or 0) <= limit for c in hit.chunks)

    # reindex_pending re-embeds the SQL chunks of a pending entry: those are the rebuilt ones, so the
    # collection ends with exactly the rebuilt point set and the 9,000-token chunk does not come back.
    repo.mark_index_result(long_id, success=False, error="embed endpoint down")
    assert service.reindex_pending() == {"reindexed": 1, "failed": 0, "remaining": 0}
    points = [p for p in env.client.scroll(name, limit=1000, with_payload=True)[0] if p.payload["entry_id"] == long_id]
    assert {p.payload["chunk_id"] for p in points} == new_chunk_ids
    assert env.client.retrieve(name, ids=[long_old_point]) == []
    assert max(count_tokens([p.payload["content"] for p in points], get_profile())) <= limit


# ------------------------------------------------------------------ 4. resplit
def test_resplit_cuts_a_10000_token_code_point_into_line_pieces_and_drops_the_old_point(env):
    from core.code_kb.vector_records import CodeChunkVectorRecord, code_chunk_point_id

    name = env.collection("rs", CODE_INDEXES)
    lines = [f"value_{i} = compute(a{i}, b{i}) + offset_{i}  # step {i}" for i in range(1000)]
    while count_tokens(["\n".join(lines)], get_profile())[0] < 10_000:
        lines += [f"extra_{i} = {i}" for i in range(200)]
    code = "\n".join(lines) + "\n"
    header = "# path: big.py\n"
    old_id = 777
    payload = code_payload("big.py", header + code, start_line=10, end_line=10 + code.count("\n") - 1, label="big",
                           code=code)
    env.client.upsert(name, points=[stored_point(old_id, payload)], wait=True)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    assert set(oversized(ctx)) == {old_id}
    assert not shard_records(ctx)

    run(spec["resplit"], ctx)

    assert env.client.retrieve(name, ids=[old_id]) == []
    pieces = sorted(env.client.scroll(name, limit=1000, with_payload=True)[0], key=lambda p: p.payload["start_line"])
    assert len(pieces) >= 6
    profile = get_profile()
    assert max(count_tokens([p.payload["content"] for p in pieces], profile)) <= profile.chunk_max_tokens
    rebuilt, expected_start = "", 10
    for piece in pieces:
        pl = piece.payload
        assert pl["start_line"] == expected_start and pl["content"] == f"{header}{pl['code']}"
        assert pl["end_line"] == pl["start_line"] + pl["code"].rstrip("\n").count("\n")
        assert pl["embed_fp"] == FP and (pl["owner_user_id"], pl["project"], pl["host"], pl["label"]) == ("u", "/repo", "h", "big")
        expected_start = pl["end_line"] + 1
        rebuilt += pl["code"]
    assert rebuilt == code                                                       # the pieces partition the code
    assert len({p.id for p in pieces}) == len(pieces)
    first_id = code_chunk_point_id("u:/repo", CodeChunkVectorRecord(
        rel_path="big.py", language="python", start_line=10, end_line=pieces[0].payload["end_line"], label="big",
        content=pieces[0].payload["content"], code=pieces[0].payload["code"], chunk_ordinal=0))
    assert first_id in {p.id for p in pieces}                                    # the collection's own identity function
    # searchable: the piece's own text finds it first
    dense, _ = det_encode([pieces[2].payload["content"]])
    hit = env.client.query_points(name, query=dense[0].tolist(), using="dense", limit=1).points[0]
    assert hit.id == pieces[2].id


def _big_code_point(env, name):
    """A 10,000-token point stored under the id its own writer gives a first chunk: piece 0 reuses it."""
    from core.code_kb.vector_records import CodeChunkVectorRecord, code_chunk_point_id

    code = "".join(f"value_{i} = compute(a{i}, b{i}) + offset_{i}  # step {i}\n" for i in range(1800))
    assert count_tokens([code], get_profile())[0] > 10_000
    content = "# path: big.py\n" + code
    old_id = code_chunk_point_id("u:/repo", CodeChunkVectorRecord(
        rel_path="big.py", language="python", start_line=1, end_line=1800, label="big", content=content, code=code))
    env.client.upsert(name, points=[stored_point(old_id, code_payload(
        "big.py", content, start_line=1, end_line=1800, label="big", code=code))], wait=True)
    return old_id


def test_a_resplit_run_again_after_it_completed_leaves_the_pieces_alone(env):
    # Production 2026-11-14 (#1610): the retried resplit met its own pieces (piece 0 had taken the
    # old point's id, so the old point looked unsplit) and stopped with "pieces would overwrite".
    name = env.collection("rs_again", CODE_INDEXES)
    old_id = _big_code_point(env, name)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    assert set(oversized(ctx)) == {old_id}
    run(spec["resplit"], ctx)
    pieces = {p.id: p.payload for p in env.client.scroll(name, limit=1000, with_payload=True)[0]}
    assert old_id in pieces and len(pieces) >= 6                     # piece 0 took the old id

    lines = run(spec["resplit"], ctx)                                 # a retry re-drives the stage

    assert {p.id: p.payload for p in env.client.scroll(name, limit=1000, with_payload=True)[0]} == pieces
    assert lines[-1] == "resplit done: 0 of 1 oversized points"


def test_resplit_leaves_a_point_its_client_reindexed_after_select(env):
    name = env.collection("rs_client", CODE_INDEXES)
    old_id = _big_code_point(env, name)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    fresh = code_payload("big.py", "# path: big.py\nvalue = 1\n", start_line=1, end_line=1, label="big", code="value = 1\n")
    env.client.upsert(name, points=[stored_point(old_id, fresh, seed=5)], wait=True)

    run(spec["resplit"], ctx)

    assert [(p.id, p.payload) for p in env.client.scroll(name, limit=1000, with_payload=True)[0]] == [(old_id, fresh)]


# ------------------------------------------------------------------- 5. verify
def test_verify_fails_while_one_stale_point_remains_and_passes_after_it_is_written(env):
    name = env.collection("vf", CODE_INDEXES)
    texts = seed_stale_code(env, name, count=4)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)
    run(spec["write"], ctx)
    assert spec["verify"].gate(ctx).ok, spec["verify"].gate(ctx).reason

    tokenizer = load_tokenizer(get_profile())
    env.client.upsert(name, points=[stored_point(99, code_payload("late.py", text_of_tokens(tokenizer, 700)))], wait=True)
    gate = spec["verify"].gate(ctx)
    assert not gate.ok and "1 point" in gate.reason and "chunk=1" in gate.reason

    late = env.ctx("code_chunks:truncated", name)
    late_spec = stages(late)
    run(late_spec["select"], late)
    embed_locally(late)
    run(late_spec["write"], late)
    assert late_spec["verify"].gate(late).ok


def test_verify_sample_check_fails_when_a_stored_vector_does_not_match_a_re_encode(env):
    name = env.collection("vf_sample", CODE_INDEXES)
    texts = seed_stale_code(env, name, count=3)
    ctx = env.ctx("code_chunks:truncated", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)
    run(spec["write"], ctx)
    ok = spec["verify"].gate(ctx)
    assert ok.ok and "sample" in ok.reason

    wrong = env.ctx("code_chunks:truncated", name)
    wrong.embed_fn = lambda batch: det_encode([t + " altered" for t in batch])      # the "model" now disagrees
    gate = stages(wrong)["verify"].gate(wrong)
    assert not gate.ok and "cosine" in gate.reason


def test_stamping_waits_for_the_sample_check_and_a_failed_check_stamps_nothing(env):
    name = env.collection("st", PAPER_INDEXES)
    tokenizer = load_tokenizer(get_profile())
    texts = {i: text_of_tokens(tokenizer, 300) + f" unique{i}" for i in range(1, 5)}

    def seed(match: bool):
        points = []
        for i, text in texts.items():
            point = stored_point(i, {"source_type": "paper", "doc_id": "d", "content": text})
            if match:                                  # the model's own output for this text
                dense, sparse = det_encode([text])
                point.vector["dense"] = dense[0].tolist()
                point.vector["sparse"] = qm.SparseVector(indices=list(sparse[0]), values=list(sparse[0].values()))
            points.append(point)
        env.client.upsert(name, points=points, wait=True)

    seed(match=False)                                           # stored vectors are NOT the model's output
    ctx = env.ctx("papers:pipeline", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    assert stamp_ids(ctx) == {1, 2, 3, 4}
    with pytest.raises(RuntimeError, match="cosine"):
        run(spec["sample"], ctx)
    run_failed = spec["stamp"]
    with pytest.raises(RuntimeError, match="sample"):
        run(run_failed, ctx)
    assert env.client.count(name, exact=True).count == 4
    assert all("embed_fp" not in p.payload for p in env.client.retrieve(name, ids=[1, 2, 3, 4]))

    seed(match=True)                                            # the unchanged-vector claim holds
    ok_ctx = env.ctx("papers:pipeline", name)
    ok_spec = stages(ok_ctx)
    run(ok_spec["select"], ok_ctx)
    run(ok_spec["sample"], ok_ctx)
    run(ok_spec["stamp"], ok_ctx)
    assert all(p.payload["embed_fp"] == FP for p in env.client.retrieve(name, ids=[1, 2, 3, 4]))
    assert ok_spec["verify"].gate(ok_ctx).ok


# --------------------------------------------------------- 6. local (in-process) execution mode
def test_the_embed_stage_of_each_mode_takes_its_own_resource_and_only_write_and_resplit_hold_qdrant(env):
    host = stages(env.ctx("code_chunks:truncated", "unused_a"))
    local = stages(env.ctx("public_code_chunks:truncated", "unused_b", execution_mode="local"))
    assert (host["embed"].resource_key, local["embed"].resource_key) == ("gpu-shim-host", "embed-service")
    for key in ("write", "resplit"):
        assert host[key].resource_key == local[key].resource_key == "qdrant"
    for key in ("select", "sample", "stamp", "verify"):             # read-only or payload-only: no lock
        assert host[key].resource_key == local[key].resource_key == ""
    assert [s.key for s in host.values()] == [s.key for s in local.values()] == [
        "select", "embed", "write", "resplit", "sample", "stamp", "verify"]
    with pytest.raises(ValueError, match="execution_mode"):
        build_reembed_pipeline(env.ctx("code_chunks:truncated", "unused_c", execution_mode="elsewhere"))


def test_local_mode_runs_the_whole_pipeline_in_process_through_the_orchestrator(env, monkeypatch):
    from app.services.pipeline.orchestrator import PipelineOrchestrator

    name = env.collection("lm_code", CODE_INDEXES)
    texts = seed_stale_code(env, name, count=5)
    from config import settings
    from core.indexing import embedders
    from tests.vllm_support import EMBED_ENV, service_url

    # the local embed stage's preflight asks the embedding service whether it is up (no GPU query any more)
    monkeypatch.setattr(settings.embedding, "service_url", service_url(EMBED_ENV))
    monkeypatch.setattr(embedders, "_SHARED_EMBEDDERS", {})            # a fresh cache, put back afterwards
    remote_calls: list = []
    monkeypatch.setattr(st, "run_remote", lambda *a, **k: remote_calls.append(a))
    monkeypatch.setenv("PIPELINE_EMBED_HOSTS", str(env.tmp_path / "no_registry.json"))     # ssh_target "local": none needed
    ctx = env.ctx("public_code_chunks:truncated", name, execution_mode="local")

    status = PipelineOrchestrator(ctx.task_id, build_reembed_pipeline(ctx), ctx, env.store).run()

    assert status == "completed", [env.store.get_subtask(ctx.task_id, k) for k in ("select", "embed", "write", "verify")]
    assert remote_calls == []                                              # nothing went to a host
    assert [p.name for p in ctx.vectors_dir.glob("vectors_*.parquet")] == ["vectors_0000.parquet"]     # one local shard
    expected, _ = det_encode([texts[i] for i in sorted(texts)])
    for i, point in zip(sorted(texts), env.client.retrieve(name, ids=sorted(texts), with_vectors=True, with_payload=True)):
        assert point.payload["embed_fp"] == FP
        np.testing.assert_allclose(point.vector["dense"], expected[i - 1], atol=2e-3)   # ctx.encode was the encoder


# ------------------------------------------------------ 7. select and stamp take no qdrant lock
def test_select_and_stamp_run_to_completion_while_another_thread_holds_the_qdrant_lock(env):
    import threading

    from app.services.pipeline.orchestrator import PipelineOrchestrator
    from app.services.pipeline.stage_lock import StageLockSet

    name = env.collection("nolock", PAPER_INDEXES)
    tokenizer = load_tokenizer(get_profile())
    texts = {i: text_of_tokens(tokenizer, 300) + f" unique{i}" for i in range(1, 4)}
    points = []
    for i, text in texts.items():          # vectors that already are the model's output: only the fingerprint is missing
        dense, sparse = det_encode([text])
        point = stored_point(i, {"source_type": "paper", "doc_id": "d", "content": text})
        point.vector["dense"] = dense[0].tolist()
        point.vector["sparse"] = qm.SparseVector(indices=list(sparse[0]), values=list(sparse[0].values()))
        points.append(point)
    env.client.upsert(name, points=points, wait=True)
    ctx = env.ctx("papers:pipeline", name)
    specs = [s for s in build_reembed_pipeline(ctx) if s.key in ("select", "sample", "stamp")]
    assert all(s.resource_key == "" for s in specs)

    locks = StageLockSet()
    held, release = threading.Event(), threading.Event()

    def other_task_holds_qdrant():                 # an ingest stage, or another re-embed's write
        assert locks.acquire("qdrant", threading.Event())
        held.set()
        release.wait(60)
        locks.release("qdrant")

    holder = threading.Thread(target=other_task_holds_qdrant)
    holder.start()
    try:
        assert held.wait(10)
        done: list = []
        driver = threading.Thread(target=lambda: done.append(
            PipelineOrchestrator(ctx.task_id, specs, ctx, env.store, locks=locks).run()))
        driver.start()
        driver.join(timeout=60)
        assert done == ["completed"], (done, "select/stamp waited for the qdrant lock")
        assert "qdrant" in locks.holders()                                  # still held by the other thread throughout
    finally:
        release.set()
        holder.join(10)
    assert all(p.payload["embed_fp"] == FP for p in env.client.retrieve(name, ids=[1, 2, 3]))
