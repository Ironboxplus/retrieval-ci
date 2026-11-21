"""papers:rebuild / arxiv:rebuild against a real SQLite file, real Qdrant, the real tokenizer and a real vLLM.

Documents are seeded the way the old pipeline left them (chunks over the profile limit, a body
mistaken for an abstract, two L1 pieces) in a SQLite database with the production DDL, with their
Markdown on disk and their old points in a scratch Qdrant collection. The select/embed/write/verify
stages then run for real; vectors come from the BGE-M3 vLLM service (RETRIEVAL_TEST_VLLM_EMBED_URL,
tests/vllm_support.py).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from qdrant_client.http import models as qm

from app.models.schemas import Chunk
from app.services.pipeline import reembed_papers
from app.services.pipeline import stages as st
from app.services.pipeline.orchestrator import PipelineOrchestrator
from app.services.pipeline.reembed_pipeline import ReembedContext, build_reembed_pipeline
from app.services.pipeline.reembed_scopes import collection_name, get_scope
from core.indexing.chunker import create_chunker, enforce_budget
from core.indexing.embedders import vllm_client
from core.indexing.embedding_profile import count_tokens, current_embed_fp, get_profile, load_tokenizer
from core.indexing.paper_chunk_reconcile import ExistingChunkRecord, chunk_point_id
from core.storage.corpus_layout import CorpusLayout, paper_collections, write_layout
from core.storage.papers_payload import project_papers_payload
from scripts.remote_embed.dump_chunks import build_payload as dump_payload
from scripts.remote_embed.dump_chunks import chunk_id_int, chunk_id_str, iter_chunks
from scripts.remote_embed.local_ingest import canonicalize_payload
from tests.embedding_model_support import bge_m3_snapshot, text_of_tokens
from tests.reembed_support import create_scratch_collection, drop_collection, new_store, qdrant_client, stored_point
from tests.sqlite_task_db import make_session_factory
from tests.vllm_support import EMBED_ENV, service_url

FP = current_embed_fp()
INDEXES = ("source_type", "embed_fp", "doc_id", "paper_id")
LIMIT = get_profile().chunk_max_tokens


@pytest.fixture(autouse=True)
def tokenizer_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BGE_M3_PATH", bge_m3_snapshot())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("QDRANT_CORPUS_LAYOUT_FILE", str(tmp_path / "corpus_layout.json"))   # absent: the mixed layout


@pytest.fixture()
def encode(monkeypatch):
    from config import settings

    url = service_url(EMBED_ENV)
    monkeypatch.setenv("EMBEDDING__SERVICE_URL", url)
    monkeypatch.setattr(settings.embedding, "service_url", url)
    profile = get_profile()
    # Interactive priority (SERVICE_PRIORITY): the shared service may hold a long background backlog
    # (143 s for one request at 5, 2026-11-14), and these tests send a handful of tiny requests.
    return lambda texts: vllm_client.encode(list(texts), profile, base_url=url, priority=0)


# --------------------------------------------------------------------------- the database
class PaperDb:
    """The paper tables with the production DDL (``Base.metadata.create_all``) in a SQLite file."""

    def __init__(self, path: Path):
        self.path = path
        make_session_factory(path)
        (path.parent / "markdown").mkdir(exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add_document(self, doc_id: int, *, title: str, markdown: str | None, venue_tier: str = "CCF-A",
                     source_type: str = "paper", venue: str = "ICML", year: int = 2026) -> str:
        paper_id = f"{venue.lower()}-{year}-{doc_id}"
        file_path = None
        if markdown is not None:
            md = self.path.parent / "markdown" / f"doc{doc_id}.md"
            md.write_text(markdown, encoding="utf-8")
            file_path = str(md)
        else:
            file_path = str(self.path.parent / "markdown" / f"missing{doc_id}.md")
        with self.connect() as conn:
            conn.execute("INSERT INTO documents (id, title, source, source_type, file_path, metadata_json) "
                         "VALUES (?, ?, ?, ?, ?, ?)",
                         (doc_id, title, f"https://example.org/{paper_id}", source_type, file_path,
                          json.dumps({"paper_id": paper_id})))
            conn.execute("INSERT INTO papers (paper_id, doc_id, title, year, venue, venue_tier, withdrawn, citation_count) "
                         "VALUES (?, ?, ?, ?, ?, ?, 0, 0)", (paper_id, doc_id, title, year, venue, venue_tier))
        return paper_id

    def add_chunks(self, doc_id: int, rows: list[dict]) -> None:
        with self.connect() as conn:
            for index, row in enumerate(rows):
                conn.execute(
                    "INSERT INTO chunks (doc_id, content, level, chunk_index, start_idx, end_idx, token_count, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, row["content"], row["level"], index, row["start_idx"], row["end_idx"],
                     len(row["content"]), json.dumps(row["metadata"])))

    def chunk_rows(self, doc_id: int) -> list[tuple]:
        with self.connect() as conn:
            return conn.execute("SELECT id, content, level, chunk_index, start_idx, end_idx, token_count, metadata_json "
                                "FROM chunks WHERE doc_id = ? ORDER BY chunk_index, id", (doc_id,)).fetchall()

    def point_ids(self, doc_id: int) -> set[int]:
        return {chunk_id_int(chunk_id_str(doc_id, content, level, start_idx or 0))
                for _, content, level, _, start_idx, *_ in self.chunk_rows(doc_id)}

    def digest(self) -> str:
        with self.connect() as conn:
            sha = hashlib.sha256()
            for row in conn.execute("SELECT * FROM chunks ORDER BY id"):
                sha.update(repr(row).encode("utf-8"))
        return sha.hexdigest()


def as_rows(chunks: list[Chunk]) -> list[dict]:
    return [{"content": c.content, "level": c.level, "start_idx": c.start_idx, "end_idx": c.end_idx,
             "metadata": c.metadata} for c in chunks]


def markdown_paper(tokenizer, title: str, *, sections: int = 4, abstract: str | None = None,
                   abstract_only: bool = False) -> str:
    """A paper in Marker's shape. ``abstract_only``: ``## Abstract`` is the last heading."""
    abstract = abstract or "We study how much of a paper one embedding can carry and what happens beyond it."
    text = f"# {title}\n\n## Abstract\n\n{abstract}\n"
    if abstract_only:
        return text
    for number in range(1, sections + 1):
        text += f"\n## {number} Section {number}\n\n{text_of_tokens(tokenizer, 180 + 25 * number)}\n"
    return text


def expected_chunks(doc_id: int, markdown: str, title: str) -> list[Chunk]:
    return create_chunker("section_l1").chunk_document(str(doc_id), markdown, title_hint=title)


# ------------------------------------------------------------------------------------ the world
class World:
    """Real Qdrant scratch collections, a real task store, the SQLite file and the encoder."""

    def __init__(self, tmp_path: Path, encode):
        self.tmp_path = tmp_path
        self.encode = encode
        self.client = qdrant_client()
        self.store, _ = new_store()
        self.db = PaperDb(tmp_path / "retrieval.db")
        self.collections: list[str] = []
        self.tokenizer = load_tokenizer(get_profile())
        self.old_points: dict[int, set[int]] = {}          # doc_id -> point ids as the old pipeline left them
        self.old_stale: dict[int, set[int]] = {}           # doc_id -> old point ids that must be gone afterwards

    def collection(self) -> str:
        name = create_scratch_collection(self.client, "paper_rb", keyword_indexes=INDEXES)
        self.collections.append(name)
        return name

    def ctx(self, scope: str, collection: str, **kw) -> ReembedContext:
        task_id = self.store.create_task(task_type="reembed_pipeline")
        return ReembedContext(
            venue=scope, year=0, store=self.store, task_id=task_id, project_root=self.tmp_path,
            execution_mode="local", ssh_target="local", scope_name=scope, collection=collection,
            client_factory=lambda: self.client, embed_fn=self.encode, pause_s=0.0, papers_db=self.db.path,
            mem_poll_s=0.05, **kw)

    def close(self) -> None:
        for name in self.collections:
            drop_collection(self.client, name)

    # ------------------------------------------------------------------ seeding
    def seed(self, collection: str, doc_id: int, title: str, markdown: str | None, old_rows: list[dict], *,
             venue_tier: str = "CCF-A", stale_rows: tuple[int, ...] = (), source_type: str = "paper") -> None:
        """A document and its old chunk rows, with one stale-vector point per row (the old worker's payload)."""
        paper_id = self.db.add_document(doc_id, title=title, markdown=markdown, venue_tier=venue_tier,
                                        source_type=source_type)
        self.db.add_chunks(doc_id, old_rows)
        points = []
        for row in old_rows:
            pid = chunk_id_int(chunk_id_str(doc_id, row["content"], row["level"], row["start_idx"]))
            payload = project_papers_payload({
                "doc_id": doc_id, "paper_id": paper_id, "source_type": source_type, "level": row["level"],
                "venue": "ICML", "year": 2026, "venue_tier": venue_tier, "content": row["content"]})
            points.append(stored_point(pid, payload))
        self.client.upsert(collection, points=points, wait=True)
        self.old_points[doc_id] = {p.id for p in points}
        self.old_stale[doc_id] = {points[i].id for i in stale_rows}

    def seed_documents(self, collection: str) -> dict[str, int]:
        """The five document shapes of the plan: (a) a >2048-token L2 chunk, (b) a giant abstract, (c) two
        L1 pieces, (d) only short chunks, (e) a chunk of >2048 characters that is far under 2048 tokens."""
        tk = self.tokenizer
        # (a)
        title_a = "A paper with one chunk that swallowed two sections"
        md_a = markdown_paper(tk, title_a, sections=5)
        exp_a = expected_chunks(1, md_a, title_a)
        giant_l2 = {"content": text_of_tokens(tk, 3000) + " tail", "level": 2, "start_idx": 0,
                    "end_idx": 0, "metadata": {"section_name": "5 Section 5"}}
        giant_l2["end_idx"] = len(giant_l2["content"])
        rows_a = as_rows(exp_a[:-1]) + [giant_l2]
        self.seed(collection, 1, title_a, md_a, rows_a, stale_rows=(len(rows_a) - 1,))
        # (b)
        title_b = "A paper whose Abstract is the last heading"
        body_b = text_of_tokens(tk, 5000)
        md_b = markdown_paper(tk, title_b, abstract=body_b, abstract_only=True)
        exp_b = expected_chunks(2, md_b, title_b)
        giant_l1 = {"content": f"{title_b}\n\n{body_b}", "level": 1, "start_idx": 0, "end_idx": len(body_b),
                    "metadata": {"section_name": "title_summary", "title": title_b, "l1_source": "abstract"}}
        rows_b = [giant_l1] + as_rows([c for c in exp_b if c.level == 2])
        self.seed(collection, 2, title_b, md_b, rows_b, stale_rows=(0,))
        # (c)
        title_c = "A paper whose giant abstract was cut into two L1 pieces"
        body_c = text_of_tokens(tk, 3500)
        md_c = markdown_paper(tk, title_c, abstract=body_c, abstract_only=True)
        exp_c = expected_chunks(3, md_c, title_c)
        pieces = enforce_budget([Chunk(id="g", doc_id="3", content=f"{title_c}\n\n{body_c}", level=1, start_idx=0,
                                       end_idx=len(body_c), metadata={"section_name": "title_summary",
                                                                      "title": title_c, "l1_source": "abstract"})])
        assert len(pieces) >= 2
        rows_c = as_rows(pieces) + as_rows([c for c in exp_c if c.level == 2])
        self.seed(collection, 3, title_c, md_c, rows_c, stale_rows=tuple(range(len(pieces))))
        # (d)
        title_d = "A paper that is fine"
        md_d = markdown_paper(tk, title_d, sections=3)
        self.seed(collection, 4, title_d, md_d, as_rows(expected_chunks(4, md_d, title_d)))
        # (e)
        title_e = "A paper with a chunk of 2600 characters"
        md_e = markdown_paper(tk, title_e, sections=3)
        exp_e = expected_chunks(5, md_e, title_e)
        wide = {"content": ("retrieval index chunk vector " * 100)[:2600], "level": 2, "start_idx": 0, "end_idx": 2600,
                "metadata": {"section_name": "1 Section 1"}}
        assert 2048 < len(wide["content"]) and count_tokens([wide["content"]], get_profile())[0] < LIMIT
        self.seed(collection, 5, title_e, md_e, as_rows(exp_e[:-1]) + [wide])
        return {"long_chunk": 1, "giant_abstract": 2, "two_l1": 3, "fine": 4, "wide_chars": 5}

    # ---------------------------------------------------------------- reading back
    def doc_points(self, collection: str, doc_id: int, with_vectors: bool = False) -> list:
        paper_id = f"icml-2026-{doc_id}"
        flt = qm.Filter(should=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=str(doc_id))),
                                qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))])
        points, offset = [], None
        while True:
            page, offset = self.client.scroll(collection, scroll_filter=flt, limit=500, offset=offset,
                                              with_payload=True, with_vectors=with_vectors)
            points.extend(page)
            if offset is None:
                return points

    def all_ids(self, collection: str) -> set[int]:
        out, offset = set(), None
        while True:
            page, offset = self.client.scroll(collection, limit=1000, offset=offset, with_payload=False)
            out.update(p.id for p in page)
            if offset is None:
                return out


@pytest.fixture()
def world(tmp_path, encode):
    w = World(tmp_path, encode)
    yield w
    w.close()


def stages(ctx):
    return {spec.key: spec for spec in build_reembed_pipeline(ctx)}


def run(spec, ctx, sink=None):
    lines: list[str] = []
    spec.run(ctx, sink or (lambda message, level="info": lines.append(message)))
    return lines


def embed_locally(ctx):
    worker = st._embed_worker()
    assert worker.embed_shards(st._embed_chunks_dir(ctx), st._embed_vectors_dir(ctx), ctx.embed_fn,
                               log=lambda m: None) is True


def shard_records(ctx) -> list[dict]:
    out = []
    for path in sorted(st._embed_chunks_dir(ctx).glob("chunks_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            out.extend(json.loads(line) for line in fh)
    return out


def run_whole_pipeline(ctx, monkeypatch) -> str:
    monkeypatch.setattr(st, "_local_idle_gpus", lambda **kw: [0])      # the hardware probe; the card is not the subject
    return PipelineOrchestrator(ctx.task_id, build_reembed_pipeline(ctx), ctx, ctx.store).run()


# ------------------------------------------------------------------------------ 0. wiring
def test_the_rebuild_scopes_belong_to_their_collections_and_the_reconcile_code_has_one_home():
    assert get_scope("papers:rebuild").kind == get_scope("arxiv:rebuild").kind == "paper_rebuild"
    assert collection_name(get_scope("papers:rebuild")) == paper_collections()["papers"]
    assert collection_name(get_scope("arxiv:rebuild")) == paper_collections()["arxiv"]

    import core.indexing.paper_chunk_reconcile as shared
    import scripts.reconcile_paper_chunk_schema as script
    assert script.build_document_reconciliation is shared.build_document_reconciliation
    assert script.load_existing_chunks is shared.load_existing_chunks
    assert script.apply_document_plan_atomically is shared.apply_document_plan_atomically
    assert not hasattr(script, "_apply_document_plan") and not hasattr(script, "ExistingChunkRecord")
    assert ExistingChunkRecord is shared.ExistingChunkRecord


# ------------------------------------------------------------------------------- 1. select
def test_select_takes_the_long_giant_abstract_and_double_l1_documents_and_nothing_else(world):
    name = world.collection()
    docs = world.seed_documents(name)
    sql_before, ids_before = world.db.digest(), world.all_ids(name)
    ctx = world.ctx("papers:rebuild", name, scan_batch=40)

    run(stages(ctx)["select"], ctx)

    plans = reembed_papers.read_plans(ctx)
    assert set(plans) == {docs["long_chunk"], docs["giant_abstract"], docs["two_l1"]}
    records = shard_records(ctx)
    ids = [r["point_id"] for r in records]
    assert len(ids) == len(set(ids))
    # only chunks without a point are embedded: the retained chunks keep theirs
    expected_new = set()
    for doc_id, plan in plans.items():
        chunks = [reembed_papers.chunk_from_json(doc_id, c) for c in plan["expected"]]
        assert sum(c.level == 1 for c in chunks) == 1 and all(c.level in (1, 2) for c in chunks)
        expected_new |= {chunk_point_id(doc_id, c) for c in chunks} - world.old_points[doc_id]
        assert set(plan["stale_point_ids"]) == world.old_stale[doc_id]
    assert set(ids) == expected_new and expected_new
    assert {r["doc_id"] for r in records} == set(plans)
    for record in records:
        assert record["chunk_id_str"] == chunk_id_str(record["doc_id"], record["content"], record["level"],
                                                      record["start_idx"])
        assert record["point_id"] == chunk_id_int(record["chunk_id_str"])
    counts = ctx.state_get("select_counts")
    assert counts["docs"] == 3 and counts["chunk"] == len(records) and counts.get("failed", 0) == 0
    assert ctx.state_get("select_complete") is True
    assert world.db.digest() == sql_before and world.all_ids(name) == ids_before        # select only reads


def test_select_resumes_from_its_checkpoint_after_an_interruption(world):
    name = world.collection()
    world.seed_documents(name)
    whole = world.ctx("papers:rebuild", name, scan_batch=30, checkpoint_rows=1)
    run(stages(whole)["select"], whole)
    reference = (set(reembed_papers.read_plans(whole)), sorted(r["point_id"] for r in shard_records(whole)))

    ctx = world.ctx("papers:rebuild", name, scan_batch=30, checkpoint_rows=1)
    max_id = world.db.connect().execute("SELECT MAX(id) FROM chunks").fetchone()[0]

    def interrupt_after_the_first_scanned_batch(message, level="info"):
        if (ctx.state_get("select_offset") or 0) > 0:
            ctx.abort.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        run(stages(ctx)["select"], ctx, interrupt_after_the_first_scanned_batch)
    stopped_at = ctx.state_get("select_offset")
    assert not ctx.state_get("select_complete") and 0 < stopped_at < max_id

    ctx.abort.clear()
    lines = run(stages(ctx)["select"], ctx)

    assert ctx.state_get("select_complete") is True
    assert any(f"resum" in line for line in lines)
    resumed = sorted(r["point_id"] for r in shard_records(ctx))
    assert resumed == reference[1] and len(resumed) == len(set(resumed))               # nothing twice, nothing lost
    assert set(reembed_papers.read_plans(ctx)) == reference[0]
    again = run(stages(ctx)["select"], ctx)
    assert any("already complete" in line for line in again)


def test_select_lists_a_missing_markdown_file_and_an_empty_one_instead_of_skipping_them(world):
    name = world.collection()
    world.seed_documents(name)
    giant = {"content": text_of_tokens(world.tokenizer, 3000), "level": 2, "start_idx": 0, "end_idx": 1,
             "metadata": {"section_name": "x"}}
    world.seed(name, 6, "Markdown is gone", None, [dict(giant, content=giant["content"] + " a")])
    world.seed(name, 7, "Markdown is empty", "   \n", [dict(giant, content=giant["content"] + " b")])
    ctx = world.ctx("papers:rebuild", name)

    run(stages(ctx)["select"], ctx)

    failures = {row["doc_id"]: row["error"] for row in reembed_papers.read_failures(ctx)}
    assert set(failures) == {6, 7}
    assert "markdown" in failures[6].lower() and "no chunks" in failures[7]
    assert set(reembed_papers.read_plans(ctx)) == {1, 2, 3}                              # the others went on
    assert ctx.state_get("select_counts")["failed"] == 2


def test_select_counts_a_chunk_with_no_token_count_by_its_text_instead_of_overlooking_it(world):
    name = world.collection()
    world.seed_documents(name)
    title = "A paper whose giant chunk has no token_count"
    markdown = markdown_paper(world.tokenizer, title, sections=3)
    expected = expected_chunks(9, markdown, title)
    giant = {"content": text_of_tokens(world.tokenizer, 2600) + " nine", "level": 2, "start_idx": 0, "end_idx": 1,
             "metadata": {"section_name": "3 Section 3"}}
    world.seed(name, 9, title, markdown, as_rows(expected[:-1]) + [giant], stale_rows=(len(expected) - 1,))
    with world.db.connect() as conn:
        conn.execute("UPDATE chunks SET token_count = NULL WHERE doc_id = 9")
    ctx = world.ctx("papers:rebuild", name)

    run(stages(ctx)["select"], ctx)

    assert 9 in reembed_papers.read_plans(ctx)


# --------------------------------------------------------------- 2. whole pipeline, then again
def live_payloads(db: PaperDb) -> dict[int, dict]:
    """Every point payload the live paper path would write for the SQL rows: the dump, the embed
    worker's payload and the ingest's projection, run over the final SQL."""
    worker = st._embed_worker()
    out = {}
    with db.connect() as conn:
        for row in iter_chunks(conn):
            _, doc_id, content, level, start_idx, end_idx, *_ = row
            cid = chunk_id_str(doc_id, content, level, start_idx)
            record = {"chunk_id_str": cid, "doc_id": doc_id, "content": content, "level": level,
                      "start_idx": start_idx, "end_idx": end_idx, "metadata": dump_payload(row)}
            out[chunk_id_int(cid)] = canonicalize_payload(worker.build_payload(record))
    return out


def test_the_whole_pipeline_replaces_the_documents_and_a_second_run_changes_nothing(world, monkeypatch):
    name = world.collection()
    docs = world.seed_documents(name)
    ctx = world.ctx("papers:rebuild", name)

    status = run_whole_pipeline(ctx, monkeypatch)

    assert status == "completed", [world.store.get_subtask(ctx.task_id, k) for k in ("select", "embed", "write", "verify")]
    live = live_payloads(world.db)
    rebuilt = (docs["long_chunk"], docs["giant_abstract"], docs["two_l1"])
    for doc_id in rebuilt:
        rows = world.db.chunk_rows(doc_id)
        assert [r[2] for r in rows].count(1) == 1 and rows[0][2] == 1                    # exactly one L1, first
        assert max(count_tokens([r[1] for r in rows], get_profile())) <= LIMIT
        assert all(r[6] == len(r[1]) for r in rows)                                       # token_count is characters
        points = world.doc_points(name, doc_id)
        assert {p.id for p in points} == world.db.point_ids(doc_id)                       # one point per SQL chunk
        assert not world.old_stale[doc_id] & {p.id for p in points}                       # the old ones are gone
        for point in points:
            payload = dict(point.payload)
            if point.id in live and payload.get("embed_fp"):
                assert payload == live[point.id]                                          # the live path's payload
            else:
                assert {k: v for k, v in live[point.id].items() if k != "embed_fp"} == payload
            assert set(payload) <= {"doc_id", "paper_id", "source_type", "level", "venue", "year", "venue_tier",
                                    "content", "embed_fp"}
            assert isinstance(payload["doc_id"], str)
    l1 = world.db.chunk_rows(docs["giant_abstract"])[0]
    assert json.loads(l1[7])["l1_source"] == "abstract_capped" and l1[1].startswith("A paper whose Abstract")
    # the two documents that were fine are exactly as they were
    for doc_id in (docs["fine"], docs["wide_chars"]):
        assert {p.id for p in world.doc_points(name, doc_id)} == world.old_points[doc_id]
        assert all("embed_fp" not in p.payload for p in world.doc_points(name, doc_id))
    # new points carry the fingerprint and the vector the service computes for their text
    fresh = [p for doc_id in rebuilt for p in world.doc_points(name, doc_id, with_vectors=True)
             if p.payload.get("embed_fp") == FP]
    assert fresh
    dense, _ = world.encode([p.payload["content"] for p in fresh])
    for point, expected in zip(fresh, dense):
        assert float(np.asarray(point.vector["dense"], dtype=np.float32) @ np.asarray(expected, dtype=np.float32)) >= 0.999
    note = world.store.get_subtask(ctx.task_id, "verify")["message"]
    assert "rebuilt" in note and "sample" in note

    sql_after, ids_after = world.db.digest(), world.all_ids(name)
    again = world.ctx("papers:rebuild", name)
    assert run_whole_pipeline(again, monkeypatch) == "completed"
    assert again.state_get("select_counts")["docs"] == 0 and shard_records(again) == []
    assert world.db.digest() == sql_after and world.all_ids(name) == ids_after


def test_write_puts_new_points_in_first_then_sql_then_deletes_the_old_points(world):
    name = world.collection()
    docs = world.seed_documents(name)
    ctx = world.ctx("papers:rebuild", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)
    sql_before = world.db.digest()
    plans = reembed_papers.read_plans(ctx)
    new_ids = {r["point_id"] for r in shard_records(ctx)}
    old_stale = set().union(*(world.old_stale[d] for d in plans))
    seen: list[tuple[str, bool, bool, bool]] = []

    def watch(message, level="info"):
        present = world.all_ids(name)
        seen.append((message, new_ids <= present, old_stale <= present, world.db.digest() == sql_before))

    run(spec["write"], ctx, watch)

    uploaded = [s for s in seen if s[0].startswith("write vectors_")][-1]        # the last shard: all new points are in
    sql_done = next(s for s in seen if "SQL" in s[0])
    deleted = next(s for s in seen if "stale points deleted" in s[0])
    assert uploaded[1:] == (True, True, True)       # new points in; old points still there; SQL untouched
    assert sql_done[1:] == (True, True, False)      # SQL replaced; old points still there
    assert deleted[1] and deleted[3] is False and not old_stale & world.all_ids(name)    # then the old points go
    assert seen.index(uploaded) < seen.index(sql_done) < seen.index(deleted)


def test_a_new_task_finishes_a_rebuild_that_stopped_after_its_points_were_written(world, monkeypatch):
    name = world.collection()
    docs = world.seed_documents(name)
    first = world.ctx("papers:rebuild", name)
    spec = stages(first)
    run(spec["select"], first)
    embed_locally(first)
    sql_before = world.db.digest()
    shards = len(list(first.vectors_dir.glob("vectors_*.parquet")))
    written: list[str] = []

    def stop_once_the_points_are_in(message, level="info"):
        if message.startswith("write vectors_"):
            written.append(message)
            if len(written) == shards:
                first.abort.set()                      # the next step, the SQL one, finds the task cancelled

    with pytest.raises(RuntimeError, match="cancelled"):
        run(spec["write"], first, stop_once_the_points_are_in)
    assert world.db.digest() == sql_before and not first.state_get("paper_finalized")
    in_between = world.all_ids(name)
    assert any(world.old_stale[d] <= in_between for d in (docs["long_chunk"], docs["giant_abstract"]))

    second = world.ctx("papers:rebuild", name)         # a fresh task: select again, the new points already exist
    assert run_whole_pipeline(second, monkeypatch) == "completed"

    assert second.state_get("select_counts")["docs"] == 3 and second.state_get("select_counts")["chunk"] == 0
    assert shard_records(second) == []                 # nothing left to embed
    for doc_id in (docs["long_chunk"], docs["giant_abstract"], docs["two_l1"]):
        assert {p.id for p in world.doc_points(name, doc_id)} == world.db.point_ids(doc_id)
        assert [r[2] for r in world.db.chunk_rows(doc_id)].count(1) == 1
    assert stages(second)["verify"].gate(second).ok


def test_every_write_step_can_run_again_without_changing_anything(world):
    name = world.collection()
    world.seed_documents(name)
    ctx = world.ctx("papers:rebuild", name)
    spec = stages(ctx)
    run(spec["select"], ctx)
    embed_locally(ctx)
    run(spec["write"], ctx)
    sql, ids = world.db.digest(), world.all_ids(name)
    payloads = {p.id: p.payload for p in world.client.retrieve(name, ids=sorted(ids), with_payload=True)}

    again = run(spec["write"], ctx)                                                       # every shard already written
    assert any("already written" in line for line in again)
    (ctx.select_dir / "write_done.txt").unlink()                                          # the upserts, once more
    ctx.state_set(paper_finalized=False)
    (ctx.select_dir / "paper_sql_done.txt").unlink()                                      # SQL plans, once more
    run(spec["write"], ctx)

    assert world.db.digest() == sql and world.all_ids(name) == ids
    assert {p.id: p.payload for p in world.client.retrieve(name, ids=sorted(ids), with_payload=True)} == payloads
    assert spec["verify"].gate(ctx).ok


# -------------------------------------------------------------------------------- 3. verify
def test_verify_names_a_missing_point_an_extra_point_a_second_l1_and_a_chunk_over_the_limit(world, monkeypatch):
    name = world.collection()
    docs = world.seed_documents(name)
    ctx = world.ctx("papers:rebuild", name)
    assert run_whole_pipeline(ctx, monkeypatch) == "completed"
    verify = stages(ctx)["verify"].gate
    assert verify(ctx).ok
    doc = docs["giant_abstract"]

    # a point of the document is gone
    victim = next(p for p in world.doc_points(name, doc, with_vectors=True) if p.payload["level"] == 2)
    world.client.delete(name, points_selector=[victim.id], wait=True)
    gate = verify(ctx)
    assert not gate.ok and "missing" in gate.reason and str(doc) in gate.reason
    world.client.upsert(name, points=[qm.PointStruct(id=victim.id, vector=victim.vector, payload=victim.payload)], wait=True)
    assert verify(ctx).ok

    # a point nobody's chunk explains
    extra = stored_point(987654321, dict(victim.payload, content="an orphan"))
    world.client.upsert(name, points=[extra], wait=True)
    gate = verify(ctx)
    assert not gate.ok and "extra" in gate.reason and "987654321" in gate.reason
    world.client.delete(name, points_selector=[987654321], wait=True)
    assert verify(ctx).ok

    # a second L1 row, and a chunk over the limit
    with world.db.connect() as conn:
        conn.execute("INSERT INTO chunks (doc_id, content, level, chunk_index, start_idx, end_idx, token_count, metadata_json) "
                     "VALUES (?, 'second summary', 1, 99, 0, 14, 14, '{}')", (doc,))
        world.client.upsert(name, points=[stored_point(chunk_id_int(chunk_id_str(doc, "second summary", 1, 0)),
                                                       dict(victim.payload, content="second summary", level=1))], wait=True)
    gate = verify(ctx)
    assert not gate.ok and "L1" in gate.reason and str(doc) in gate.reason
    with world.db.connect() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ? AND content = 'second summary'", (doc,))
    world.client.delete(name, points_selector=[chunk_id_int(chunk_id_str(doc, "second summary", 1, 0))], wait=True)
    assert verify(ctx).ok

    huge = text_of_tokens(world.tokenizer, 2500)
    with world.db.connect() as conn:
        conn.execute("INSERT INTO chunks (doc_id, content, level, chunk_index, start_idx, end_idx, token_count, metadata_json) "
                     "VALUES (?, ?, 2, 98, 0, ?, ?, '{}')", (doc, huge, len(huge), len(huge)))
    world.client.upsert(name, points=[stored_point(chunk_id_int(chunk_id_str(doc, huge, 2, 0)),
                                                   dict(victim.payload, content=huge, level=2))], wait=True)
    gate = verify(ctx)
    assert not gate.ok and "2048" in gate.reason and str(doc) in gate.reason


def test_verify_fails_while_a_document_could_not_be_rebuilt(world, monkeypatch):
    name = world.collection()
    world.seed_documents(name)
    world.seed(name, 6, "Markdown is gone", None,
               [{"content": text_of_tokens(world.tokenizer, 3000), "level": 2, "start_idx": 0, "end_idx": 1,
                 "metadata": {"section_name": "x"}}])
    ctx = world.ctx("papers:rebuild", name)

    status = run_whole_pipeline(ctx, monkeypatch)

    assert status == "failed"
    gate = stages(ctx)["verify"].gate(ctx)
    assert not gate.ok and "1 document" in gate.reason and "6" in gate.reason and "markdown" in gate.reason.lower()


# ---------------------------------------------------------------------- 4. arxiv and routing
def test_each_scope_rebuilds_only_the_documents_its_collection_serves(world, monkeypatch, tmp_path):
    write_layout(CorpusLayout(split=True, papers_legacy_mixed=False), tmp_path / "corpus_layout.json")
    papers_name, arxiv_name = world.collection(), world.collection()
    world.seed_documents(papers_name)
    tk = world.tokenizer
    title = "A preprint with a body for an abstract"
    body = text_of_tokens(tk, 4000)
    md = markdown_paper(tk, title, abstract=body, abstract_only=True)
    world.seed(arxiv_name, 8, title, md,
               [{"content": f"{title}\n\n{body}", "level": 1, "start_idx": 0, "end_idx": len(body),
                 "metadata": {"section_name": "title_summary", "title": title, "l1_source": "abstract"}}]
               + as_rows([c for c in expected_chunks(8, md, title) if c.level == 2]),
               venue_tier="preprint", stale_rows=(0,))

    arxiv = world.ctx("arxiv:rebuild", arxiv_name)
    papers = world.ctx("papers:rebuild", papers_name)
    run(stages(arxiv)["select"], arxiv)              # both scans see the database before either rebuilds anything
    run(stages(papers)["select"], papers)
    assert set(reembed_papers.read_plans(arxiv)) == {8}
    assert arxiv.state_get("select_counts")["foreign"] == 3                  # the three formal papers: papers:rebuild's
    assert set(reembed_papers.read_plans(papers)) == {1, 2, 3}
    assert papers.state_get("select_counts")["foreign"] == 1                 # the preprint: arxiv:rebuild's
    assert run_whole_pipeline(arxiv, monkeypatch) == "completed"             # select is already complete: the rest runs
    assert run_whole_pipeline(papers, monkeypatch) == "completed"
    points = world.doc_points(arxiv_name, 8)
    assert {p.id for p in points} == world.db.point_ids(8) and all(p.payload["venue_tier"] == "preprint" for p in points)
    assert world.db.chunk_rows(8)[0][2] == 1 and [r[2] for r in world.db.chunk_rows(8)].count(1) == 1


# ----------------------------------------------------------------------- 5. the memory guard
def test_the_guard_lets_a_select_through_while_memory_is_ample_and_between_the_two_bounds():
    ctx = ReembedContext(venue="papers:rebuild", year=0, store=None, task_id=1, scope_name="papers:rebuild",
                         mem_poll_s=0.05)
    messages: list[str] = []
    available = reembed_papers.available_memory_bytes()
    assert available > 0

    ctx.mem_pause_below, ctx.mem_resume_above = 1, 2
    started = time.monotonic()
    reembed_papers.wait_for_memory(ctx, messages.append)
    # between the bounds: below the resume level but above the pause level, a running scan is not stopped
    ctx.mem_pause_below, ctx.mem_resume_above = available // 2, available * 2 + (64 << 30)
    reembed_papers.wait_for_memory(ctx, messages.append)

    assert time.monotonic() - started < 2 and messages == []


def test_the_guard_pauses_below_the_low_bound_and_resumes_only_once_memory_is_back_above_the_high_one():
    ctx = ReembedContext(venue="papers:rebuild", year=0, store=None, task_id=1, scope_name="papers:rebuild",
                         mem_poll_s=0.05)
    messages: list[str] = []
    available = reembed_papers.available_memory_bytes()
    ctx.mem_pause_below, ctx.mem_resume_above = available + (8 << 30), available + (16 << 30)
    raised = threading.Event()

    def memory_comes_back():
        time.sleep(0.5)
        ctx.mem_pause_below = ctx.mem_resume_above = 0
        raised.set()

    threading.Thread(target=memory_comes_back, daemon=True).start()
    started = time.monotonic()
    reembed_papers.wait_for_memory(ctx, messages.append)

    assert raised.is_set() and time.monotonic() - started >= 0.4
    assert len(messages) == 2 and "pause" in messages[0] and "resum" in messages[1]
    assert "MemAvailable" in messages[0] and "GiB" in messages[0]


def test_a_task_cancelled_while_the_guard_holds_it_stops():
    ctx = ReembedContext(venue="papers:rebuild", year=0, store=None, task_id=1, scope_name="papers:rebuild",
                         mem_poll_s=0.05)
    ctx.mem_pause_below = ctx.mem_resume_above = reembed_papers.available_memory_bytes() + (64 << 30)
    threading.Timer(0.3, ctx.abort.set).start()

    with pytest.raises(RuntimeError, match="cancelled"):
        reembed_papers.wait_for_memory(ctx, lambda message: None)


def test_select_waits_for_memory_between_batches_and_carries_on_when_it_returns(world):
    name = world.collection()
    world.seed_documents(name)
    ctx = world.ctx("papers:rebuild", name, scan_batch=40)
    ctx.mem_pause_below = ctx.mem_resume_above = reembed_papers.available_memory_bytes() + (64 << 30)

    def memory_comes_back():
        time.sleep(0.6)
        ctx.mem_pause_below = ctx.mem_resume_above = 0

    threading.Thread(target=memory_comes_back, daemon=True).start()
    started = time.monotonic()
    lines = run(stages(ctx)["select"], ctx)

    assert time.monotonic() - started >= 0.5
    assert any("pause" in line for line in lines) and any("resum" in line for line in lines)
    assert ctx.state_get("select_complete") is True and set(reembed_papers.read_plans(ctx)) == {1, 2, 3}
