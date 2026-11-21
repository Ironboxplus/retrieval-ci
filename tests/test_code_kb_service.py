"""TDD for the server-side code KB service (app/services/code_kb).

The service stores AST-chunked code in the 4090 Qdrant `code_chunks` collection,
scoped per (owner_user_id, project[, host]). These tests inject a fake in-memory
vector store + a deterministic bag-of-words embed_fn so they need no GPU/Qdrant.
"""
from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest

from app.services.code_kb.schemas import (
    CodeChunkIn,
    CodeFileIn,
    CodeHit,
    CodeIndexRequest,
    CodeSearchRequest,
)
from app.services.code_kb.service import CodeKBService
from app.services.code_kb.vector_store import point_id


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
_VOCAB = ["qdrant", "upsert", "wait", "embed", "search", "cosine", "playbook",
          "journal", "chunk", "render", "react", "button", "vector", "index"]


def _bow(text: str) -> list[float]:
    t = text.lower()
    v = [float(t.count(w)) for w in _VOCAB]
    if not any(v):
        v = [1.0] * len(_VOCAB)
    return v


def _embed(texts):
    return [_bow(t) for t in texts]


def _sparse_embed(texts):
    """Deterministic fake sparse: one non-zero lexical weight per text."""
    return [{"1": float(len(t))} for t in texts]


class _PrioritySpyEmbedder:
    def __init__(self):
        self.priorities = []

    def embed_texts(self, texts, *, priority="interactive"):
        import numpy as np
        from core.indexing.embedders import EmbeddingResult

        self.priorities.append(priority)
        return EmbeddingResult(
            dense=np.asarray([_bow(text) for text in texts], dtype=np.float32),
            sparse=[{"1": 1.0} for _ in texts],
        )


class _SpyInnerStore:
    """Records which QdrantStore retriever CodeVectorStore dispatched to."""
    supports_sparse = True

    def __init__(self):
        self.calls = []

    def _fmt(self, tag):
        return [{"score": 1.0, "content": "x", "metadata": {"path": f"{tag}.py", "code": "c"}}]

    def search(self, query_embedding, top_k=10, filter_dict=None, **kw):
        self.calls.append("dense"); return self._fmt("dense")

    def hybrid_search(self, dense_vector, sparse_vector, top_k=10, prefetch_limit=100,
                      filter_dict=None, **kw):
        self.calls.append("hybrid"); return self._fmt("hybrid")

    def sparse_search(self, sparse_vector, top_k=10, filter_dict=None, **kw):
        self.calls.append("keyword"); return self._fmt("keyword")


class FakeCodeVectorStore:
    """In-memory stand-in for CodeVectorStore — same method surface."""

    supports_sparse = True

    def reusable_vectors(self, *, items, **_scope):
        """In-memory stores keep no vectors worth reusing; real Qdrant is tested separately."""
        return [None] * len(items)

    def __init__(self):
        self.points: list[dict] = []
        self.last_sparse_vectors = None   # sparse arg captured on the last upsert
        self.last_mode = None             # mode captured on the last search
        self.last_sparse_query = None     # sparse query vector captured on the last search

    @staticmethod
    def _project_matches(candidate: str, project) -> bool:
        """Mirror Qdrant ``MatchValue`` / ``MatchAny`` project filtering."""
        if project is None:
            return True
        if isinstance(project, str):
            return candidate == project
        return candidate in project

    def upsert(self, *, owner_user_id, project, host, items, dense_vectors, sparse_vectors=None):
        self.last_sparse_vectors = sparse_vectors
        for i, (it, vec) in enumerate(zip(items, dense_vectors)):
            pid = point_id(
                owner_user_id,
                project,
                it.rel_path,
                it.start_line,
                it.label,
                it.chunk_ordinal,
            )
            # Qdrant upsert semantics: a deterministic id overwrites in place, never dups.
            self.points = [p for p in self.points if p["id"] != pid]
            self.points.append({
                "id": pid,
                "owner_user_id": owner_user_id, "project": project, "host": host,
                "path": it.rel_path, "language": it.language,
                "start_line": it.start_line, "end_line": it.end_line,
                "label": it.label, "content": it.content, "code": it.code,
                "vector": vec,
                "sparse": sparse_vectors[i] if sparse_vectors else None,
            })
        return len(items)

    def list_point_ids(self, *, owner_user_id, project, path):
        return {p["id"] for p in self.points if p["owner_user_id"] == owner_user_id
                and p["project"] == project and p["path"] == path}

    def list_point_ids_for_paths(self, *, owner_user_id, project, paths=None):
        return {
            p["id"] for p in self.points
            if p["owner_user_id"] == owner_user_id
            and p["project"] == project
            and (paths is None or p["path"] in paths)
        }

    def delete_point_ids(self, ids):
        ids = set(ids)
        if not ids:
            return 0
        before = len(self.points)
        self.points = [p for p in self.points if p["id"] not in ids]
        return before - len(self.points)

    def delete_project(self, *, owner_user_id, project):
        # host is NOT a scoping key (W148): deleting a project drops every point
        # for (owner, project) regardless of which machine indexed it.
        before = len(self.points)
        self.points = [p for p in self.points if not (
            p["owner_user_id"] == owner_user_id and p["project"] == project)]
        return before - len(self.points)

    def list_paths(self, *, owner_user_id, project):
        return {p["path"] for p in self.points
                if p["owner_user_id"] == owner_user_id
                and self._project_matches(p["project"], project)}

    def list_owner_files(self, *, owner_user_id):
        return {(p["project"], p["path"]) for p in self.points
                if p["owner_user_id"] == owner_user_id}

    def list_owner_paths(self, *, owner_user_id):
        return {p["path"] for p in self.points if p["owner_user_id"] == owner_user_id}

    def delete_paths(self, *, owner_user_id, project, paths):
        if not paths:
            return 0
        before = len(self.points)
        self.points = [p for p in self.points if not (
            p["owner_user_id"] == owner_user_id and p["project"] == project
            and p["path"] in paths)]
        return before - len(self.points)

    def search(self, *, dense_vector, top_k, owner_user_id, project=None, language=None,
               sparse_vector=None, mode="dense", paths=None, excluded_projects=None):
        self.last_mode = mode
        self.last_sparse_query = sparse_vector

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0
        # Mirror the real store's MatchAny push-down on `path` (path_glob resolution),
        # so top_k fills only from matching files — no post-filter drop.
        allowed = set(paths) if paths is not None else None
        excluded = set(excluded_projects) if excluded_projects else None
        cands = [p for p in self.points
                 if p["owner_user_id"] == owner_user_id
                 and self._project_matches(p["project"], project)
                 and (language is None or p["language"] == language)
                 and (allowed is None or p["path"] in allowed)
                 and (excluded is None or p["project"] not in excluded)]
        scored = sorted(((cos(dense_vector, p["vector"]), p) for p in cands),
                        key=lambda x: x[0], reverse=True)
        return [{"score": s, **p} for s, p in scored[:top_k]]

    def count(self, *, owner_user_id=None, project=None):
        return len([p for p in self.points
                    if (owner_user_id is None or p["owner_user_id"] == owner_user_id)
                    and (project is None or p["project"] == project)])


def _svc(store=None):
    return CodeKBService(embed_fn=_embed, vector_store=store or FakeCodeVectorStore())


def _req(project="proj", host="host1", files=None):
    return CodeIndexRequest(project=project, host=host, root_dir="/repo", files=files or [])


def _file(rel_path, chunks, language="python"):
    return CodeFileIn(rel_path=rel_path, language=language, chunks=chunks)


def _chunk(content, code=None, start=1, end=10, label="fn"):
    return CodeChunkIn(start_line=start, end_line=end, label=label,
                       content=content, code=code or content)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_index_then_search_returns_relevant_chunk():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    req = _req(files=[
        _file("core/store.py", [_chunk("qdrant upsert wait vector", label="upsert_batch")]),
        _file("ui/App.tsx", [_chunk("react render button", label="App")], language="typescript"),
    ])
    resp = svc.index(req, owner_user_id="u1")
    assert resp.indexed == 2 and resp.files == 2

    hits = svc.search(CodeSearchRequest(query="how to upsert vectors into qdrant", top_k=3),
                      owner_user_id="u1").hits
    assert hits, "expected at least one hit"
    assert hits[0].path == "core/store.py"
    assert hits[0].label == "upsert_batch"
    assert hits[0].start_line == 1 and hits[0].code


def test_search_reports_per_phase_timings_ms():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("core/store.py", [_chunk("qdrant upsert wait vector", label="upsert_batch")]),
    ]), owner_user_id="u1")

    resp = svc.search(
        CodeSearchRequest(query="how to upsert vectors into qdrant", top_k=3),
        owner_user_id="u1")

    t = resp.timings_ms
    assert t.embed is not None and t.embed >= 0
    assert t.retrieve is not None and t.retrieve >= 0
    assert t.rerank is None, "rerank=false must leave rerank timing null, not 0"
    assert t.fuse is None, "a single-leg search never fuses"
    assert t.total is not None
    measured_sum = t.embed + t.retrieve
    assert t.total >= measured_sum * 0.9


def test_search_with_rerank_reports_rerank_timing():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("core/store.py", [_chunk("qdrant upsert wait vector", label="upsert_batch")]),
    ]), owner_user_id="u1")

    resp = svc.search(
        CodeSearchRequest(query="how to upsert vectors into qdrant", top_k=3, rerank=True),
        owner_user_id="u1")

    t = resp.timings_ms
    assert t.rerank is not None and t.rerank >= 0
    assert t.total >= (t.embed + t.retrieve + t.rerank) * 0.9


def test_empty_query_search_still_reports_total_timing():
    resp = _svc().search(CodeSearchRequest(query="   "), owner_user_id="u1")
    assert resp.timings_ms.total is not None and resp.timings_ms.total >= 0
    assert resp.timings_ms.embed is None
    assert resp.timings_ms.retrieve is None


def test_search_reads_previous_snapshot_while_new_index_is_embedding():
    """Embedding a refresh can take minutes; search must not hold behind that work.
    Until mutation starts, callers should keep reading the previous complete snapshot."""
    store = FakeCodeVectorStore()
    embed_started = threading.Event()
    release_embed = threading.Event()

    def blocking_embed(texts):
        if any("NEW_SNAPSHOT" in text for text in texts):
            embed_started.set()
            assert release_embed.wait(5), "test did not release the blocked embed"
        return _embed(texts)

    svc = CodeKBService(embed_fn=blocking_embed, vector_store=store)
    svc.index(_req(files=[
        _file("service.py", [_chunk("qdrant OLD_SNAPSHOT", label="old")]),
    ]), owner_user_id="u1")

    index_thread = threading.Thread(
        target=lambda: svc.index(_req(files=[
            _file("service.py", [_chunk("qdrant NEW_SNAPSHOT", label="new")]),
        ]), owner_user_id="u1"),
        daemon=True,
    )
    index_thread.start()
    assert embed_started.wait(1), "index did not reach the blocking embed"

    search_done = threading.Event()
    search_result = []

    def run_search():
        search_result.extend(svc.search(
            CodeSearchRequest(query="qdrant OLD_SNAPSHOT", project="proj", top_k=3),
            owner_user_id="u1").hits)
        search_done.set()

    search_thread = threading.Thread(target=run_search, daemon=True)
    search_thread.start()
    try:
        assert search_done.wait(1), "search blocked behind the indexing embed"
        assert search_result and "OLD_SNAPSHOT" in search_result[0].code
    finally:
        release_embed.set()
        index_thread.join(timeout=5)
        search_thread.join(timeout=5)


def test_shared_embedder_marks_index_batches_background_and_queries_interactive(monkeypatch):
    """Code KB must tell the shared embedder scheduler which work may yield."""
    embedder = _PrioritySpyEmbedder()
    monkeypatch.setattr("core.indexing.embedders.get_embedder", lambda: embedder)
    store = FakeCodeVectorStore()
    svc = CodeKBService(vector_store=store)

    svc.index(_req(files=[_file("service.py", [
        _chunk("qdrant indexed code", label="indexed_function"),
    ])]), owner_user_id="u1")
    svc.search(CodeSearchRequest(
        query="indexed_function", project="proj", mode="keyword"), owner_user_id="u1")

    assert embedder.priorities[0] == "background"
    assert embedder.priorities[-1] == "interactive"


def test_default_dense_with_custom_sparse_still_marks_index_background(monkeypatch):
    embedder = _PrioritySpyEmbedder()
    monkeypatch.setattr("core.indexing.embedders.get_embedder", lambda: embedder)
    svc = CodeKBService(
        sparse_embed_fn=_sparse_embed, vector_store=FakeCodeVectorStore())

    svc.index(_req(files=[_file("service.py", [
        _chunk("qdrant indexed code", label="indexed_function"),
    ])]), owner_user_id="u1")

    assert embedder.priorities == ["background"]


def test_reindex_replaces_stale_chunks():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="one"),
                                          _chunk("embed", label="two")])]),
              owner_user_id="u1")
    assert store.count(owner_user_id="u1", project="proj") == 2
    # Re-index the same project with a single chunk → the stale one must be gone.
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="one")])]),
              owner_user_id="u1")
    assert store.count(owner_user_id="u1", project="proj") == 1


def test_full_reindex_upsert_failure_preserves_previous_complete_snapshot():
    class PartiallyFailingStore(FakeCodeVectorStore):
        fail_upsert = False

        def upsert(self, **kwargs):
            if not self.fail_upsert:
                return super().upsert(**kwargs)
            partial = dict(kwargs)
            partial["items"] = kwargs["items"][:1]
            partial["dense_vectors"] = kwargs["dense_vectors"][:1]
            if kwargs.get("sparse_vectors") is not None:
                partial["sparse_vectors"] = kwargs["sparse_vectors"][:1]
            super().upsert(**partial)
            raise RuntimeError("partial qdrant upsert failure")

    store = PartiallyFailingStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("old_a.py", [_chunk("qdrant old a", label="old_a")]),
        _file("old_b.py", [_chunk("qdrant old b", label="old_b")]),
    ]), owner_user_id="u1")

    store.fail_upsert = True
    with pytest.raises(RuntimeError, match="partial qdrant upsert failure"):
        svc.index(_req(files=[
            _file("new.py", [_chunk("qdrant new", label="new")]),
        ]), owner_user_id="u1")

    assert {"old_a.py", "old_b.py"}.issubset(
        store.list_paths(owner_user_id="u1", project="proj"))


def test_reindex_from_different_host_replaces_not_duplicates():
    # W148: host is display-only, not a scoping key. Re-indexing the same project
    # from a renamed/second machine must REPLACE (not co-exist as duplicates).
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(host="laptop", files=[_file("a.py", [_chunk("qdrant upsert", label="one")])]),
              owner_user_id="u1")
    assert store.count(owner_user_id="u1", project="proj") == 1
    svc.index(_req(host="desktop", files=[_file("a.py", [_chunk("qdrant upsert", label="one")])]),
              owner_user_id="u1")
    assert store.count(owner_user_id="u1", project="proj") == 1  # not 2
    hits = svc.search(CodeSearchRequest(query="qdrant upsert", top_k=5), owner_user_id="u1").hits
    assert hits and hits[0].host == "desktop"  # latest machine wins


def test_empty_index_does_not_wipe_existing():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="one")])]), owner_user_id="u1")
    assert store.count(owner_user_id="u1", project="proj") == 1
    # An empty enumeration must NOT delete the existing index.
    resp = svc.index(_req(files=[]), owner_user_id="u1")
    assert resp.indexed == 0
    assert store.count(owner_user_id="u1", project="proj") == 1


def test_search_scoped_to_owner():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant upsert", label="secret")])]),
              owner_user_id="u1")
    # u2 must not see u1's code.
    hits = svc.search(CodeSearchRequest(query="qdrant upsert", top_k=5),
                      owner_user_id="u2").hits
    assert hits == []


def test_search_project_filter():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(project="p1", files=[_file("a.py", [_chunk("qdrant upsert", label="a")])]),
              owner_user_id="u1")
    svc.index(_req(project="p2", files=[_file("b.py", [_chunk("qdrant upsert", label="b")])]),
              owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="qdrant upsert", project="p2", top_k=5),
                      owner_user_id="u1").hits
    assert hits and all(h.label == "b" for h in hits)


def test_search_language_filter():
    # Wave-1 #3: language filter keeps a config-file match from beating real code.
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("conf/app.yml", [_chunk("bandwidth rate limiting config", label="")], language="yaml"),
        _file("src/limiter.rs", [_chunk("bandwidth rate limiting token bucket", label="Bucket")],
              language="rust"),
    ]), owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="bandwidth rate limiting", language="rust", top_k=5),
                      owner_user_id="u1").hits
    assert hits and all(h.language == "rust" for h in hits)
    assert hits[0].path == "src/limiter.rs"


def test_search_path_glob_filter():
    # Wave-1 #3: restrict to a subtree (prefix) or a glob like "*.rs".
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("easytier/src/peers/conn.rs", [_chunk("select connection recv limiter", label="sel")]),
        _file("easytier/src/tunnel/mod.rs", [_chunk("select connection recv limiter", label="t")]),
        _file("other/util.go", [_chunk("select connection recv limiter", label="u")]),
    ]), owner_user_id="u1")
    prefix = svc.search(CodeSearchRequest(query="select connection recv limiter",
                                          path_glob="easytier/src/peers/", top_k=5),
                        owner_user_id="u1").hits
    assert prefix and all(h.path.startswith("easytier/src/peers/") for h in prefix)
    globbed = svc.search(CodeSearchRequest(query="select connection recv limiter",
                                           path_glob="*.go", top_k=5),
                         owner_user_id="u1").hits
    assert globbed and all(h.path.endswith(".go") for h in globbed)


def test_windows_scope_and_path_glob_are_case_insensitive_with_backslashes():
    """A Windows caller may change both path casing and separators without splitting the
    project or making its subtree filter silently miss the indexed Go file."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    typed_index_scope = r"E:\JETB\Pythons\Playground\Octopus\AxonHub\LLM"
    svc.index(_req(project=typed_index_scope, files=[
        _file(r"Conversion\Planner\Compact.go", [
            _chunk("RequestTypeCompact request local session", label="RequestTypeCompact")],
              language="go"),
    ]), owner_user_id="u1")

    response = svc.search(CodeSearchRequest(
        query="RequestTypeCompact", project="e:/jetb/pythons/playground/octopus/axonhub/llm",
        path_glob="conversion\\planner\\", top_k=5), owner_user_id="u1")

    assert response.total == 1
    assert response.hits[0].path == "Conversion/Planner/Compact.go"
    assert response.hits[0].project == "E:/jetb/pythons/playground/octopus/axonhub/llm"


def test_windows_search_includes_legacy_case_variant_scope_without_reindex():
    """A pre-0.4.5 vector scope remains searchable after clients canonicalise NTFS paths.

    This models the Axon failure: the catalogue knows an old mixed-case scope, but a
    new client asks for the same Windows directory using a normalized spelling. The
    compatibility read must use one MatchAny project filter and retain the detailed
    legacy-scope diagnostic rather than claiming the scope is absent.
    """
    store = FakeCodeVectorStore()
    svc = _svc(store)
    legacy = "E:/JETB/Pythons/Playground/Octopus/AxonHub/LLM"
    canonical = "E:/jetb/pythons/playground/octopus/axonhub/llm"
    store.points.append({
        "id": 1,
        "owner_user_id": "u1",
        "project": legacy,
        "host": "DESKTOP-OTHER",
        "path": "Conversion/Planner/Compact.go",
        "language": "go",
        "start_line": 1,
        "end_line": 8,
        "label": "RequestTypeCompact",
        "content": "RequestTypeCompact request local session",
        "code": "func convertCompact() {}",
        "vector": _bow("RequestTypeCompact request local session"),
        "sparse": {"1": 1.0},
    })

    response = svc.search(
        CodeSearchRequest(
            query="RequestTypeCompact",
            project=canonical,
            mode="dense",
            path_glob="conversion\\planner\\",
            top_k=5,
        ),
        owner_user_id="u1",
        project_aliases=[canonical, legacy],
    )

    assert response.total == 1
    assert response.hits[0].project == legacy
    assert response.scope_aliases == [legacy]


def test_path_glob_reads_legacy_stored_leading_slash_without_raising():
    """A pre-normalisation Qdrant path is still relative for matching purposes."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    store.points.append({
        "id": 1, "owner_user_id": "u1", "project": "proj", "host": "legacy-host",
        "path": "/conversion/compact.go", "language": "go",
        "start_line": 1, "end_line": 4, "label": "compact",
        "content": "RequestTypeCompact marker", "code": "func compact() {}",
        "vector": _bow("RequestTypeCompact marker"), "sparse": None,
    })

    response = svc.search(CodeSearchRequest(
        query="RequestTypeCompact", project="proj", mode="dense",
        path_glob="conversion/", top_k=5,
    ), owner_user_id="u1")

    assert response.total == 1
    assert response.hits[0].path == "/conversion/compact.go"


def test_search_path_glob_does_not_drop_topk_beyond_fetch_cap():
    """Push-down (not post-filter): a matching file ranked FAR below the over-fetch cap
    must still come back. The old design fetched only the top MAX_FETCH_K by score then
    post-filtered — so a narrow subtree competing against a big repo whose non-matching
    files outrank it beyond the cap silently returned nothing (dropped top_k). Resolving
    the match-set and pushing it down as a MatchAny on `path` fills top_k only from
    matching files, so rank position can never hide a match."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    # 250 non-matching files (all identical score) that occupy every over-fetch slot,
    # plus one matching file appended last so it ranks dead last.
    files = [_file(f"other/f{i}.py", [_chunk("plain filler body", label=f"n{i}")])
             for i in range(250)]
    files.append(_file("target/deep/hit.py", [_chunk("plain filler body", label="hit")]))
    svc.index(_req(files=files), owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="plain filler body", path_glob="target/", top_k=5),
                      owner_user_id="u1").hits
    assert {h.path for h in hits} == {"target/deep/hit.py"}


def test_search_path_glob_pushdown_all_modes():
    """dense / hybrid / keyword must all honor path_glob via the same push-down, so none
    of them can drop a matching hit that ranks below the fetch cap."""
    for mode in ("dense", "hybrid", "keyword"):
        store = FakeCodeVectorStore()
        svc = CodeKBService(embed_fn=_embed, sparse_embed_fn=_sparse_embed, vector_store=store)
        svc.index(_req(files=[
            _file("keep/a.py", [_chunk("bandwidth limiter marker", label="a")]),
            _file("skip/b.py", [_chunk("bandwidth limiter marker", label="b")]),
        ]), owner_user_id="u1")
        hits = svc.search(CodeSearchRequest(query="bandwidth limiter marker", mode=mode,
                                            path_glob="keep/", top_k=5), owner_user_id="u1").hits
        assert {h.path for h in hits} == {"keep/a.py"}, mode


def test_search_path_glob_cross_project_matches_by_filename():
    """project=None (search across all owner projects) + path_glob resolves matches by
    rel_path across every project — a shared filename in two repos is intentionally matched
    in both. Locks this behavior so it isn't silently broken later."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    for proj in ("p1", "p2"):
        svc.index(_req(project=proj, files=[
            _file("src/main.py", [_chunk("qdrant upsert vector", label=f"{proj}")]),
            _file("docs/readme.md", [_chunk("qdrant upsert vector", label=f"{proj}-doc")]),
        ]), owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="qdrant upsert vector",
                                        path_glob="src/main.py", top_k=10),
                      owner_user_id="u1").hits
    assert {h.project for h in hits} == {"p1", "p2"}
    assert all(h.path == "src/main.py" for h in hits)


def test_search_path_glob_single_file_is_exact():
    """A single-file path_glob must not leak into a same-stem sibling
    ('src/main.rs' must not match 'src/main.rs.bak')."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("src/main.rs", [_chunk("token bucket limiter", label="a")]),
        _file("src/main.rs.bak", [_chunk("token bucket limiter", label="b")]),
    ]), owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="token bucket limiter", project="proj",
                                        path_glob="src/main.rs", top_k=5),
                      owner_user_id="u1").hits
    assert {h.path for h in hits} == {"src/main.rs"}


def test_searches_in_same_project_are_not_serialized():
    class ConcurrentSearchStore(FakeCodeVectorStore):
        def __init__(self):
            super().__init__()
            self.first_entered = threading.Event()
            self.second_entered = threading.Event()
            self.release_first = threading.Event()
            self.call_count = 0
            self.call_lock = threading.Lock()

        def search(self, **kwargs):
            with self.call_lock:
                self.call_count += 1
                call_number = self.call_count
            if call_number == 1:
                self.first_entered.set()
                assert self.release_first.wait(5)
            else:
                self.second_entered.set()
            return super().search(**kwargs)

    store = ConcurrentSearchStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("service.py", [_chunk("qdrant concurrent search", label="search")]),
    ]), owner_user_id="u1")

    first = threading.Thread(target=lambda: svc.search(
        CodeSearchRequest(query="qdrant", project="proj"), owner_user_id="u1"))
    second = threading.Thread(target=lambda: svc.search(
        CodeSearchRequest(query="qdrant", project="proj"), owner_user_id="u1"))
    first.start()
    assert store.first_entered.wait(1)
    second.start()
    try:
        assert store.second_entered.wait(1), "same-project searches were serialized"
    finally:
        store.release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)


def test_search_max_per_file_caps_domination():
    # Wave-1 #6: one file's many chunks must not monopolise the top-k.
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[
        _file("token_bucket.rs", [_chunk(f"token bucket test case {i}", label=f"t{i}", start=i)
                                  for i in range(1, 6)]),
        _file("select_conn.rs", [_chunk("token bucket used by select conn", label="sel")]),
    ]), owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="token bucket", max_per_file=2, top_k=4),
                      owner_user_id="u1").hits
    from collections import Counter
    counts = Counter(h.path for h in hits)
    assert all(n <= 2 for n in counts.values()), counts
    assert "select_conn.rs" in counts, "capping one file must let others surface"


def test_max_per_file_capped_per_project_not_globally():
    # SF1: two projects share the rel_path 'main.py'. Searching across all projects
    # (project=None) with max_per_file=2 must give EACH project's main.py its own cap
    # bucket (4 hits total), not collapse them into one 'main.py' bucket (2 hits) —
    # which is exactly what a path-only key would wrongly do.
    from collections import Counter
    store = FakeCodeVectorStore()
    svc = _svc(store)
    for proj in ("p1", "p2"):
        svc.index(_req(project=proj, files=[
            _file("main.py", [_chunk("qdrant upsert vector", label=f"{proj}-{i}", start=i)
                              for i in range(1, 4)])]), owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="qdrant upsert vector", max_per_file=2, top_k=10),
                      owner_user_id="u1").hits
    by_project = Counter(h.project for h in hits)
    assert by_project["p1"] == 2 and by_project["p2"] == 2, by_project
    assert len(hits) == 4


def test_index_stores_sparse_vectors_when_available():
    # Wave-2 hybrid: a sparse_embed_fn means sparse vectors get persisted alongside dense.
    store = FakeCodeVectorStore()
    svc = CodeKBService(embed_fn=_embed, sparse_embed_fn=_sparse_embed, vector_store=store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant upsert", label="one"),
                                         _chunk("embed vector", label="two")])]),
              owner_user_id="u1")
    assert store.last_sparse_vectors is not None
    assert len(store.last_sparse_vectors) == 2


def test_index_omits_sparse_when_no_sparse_fn():
    # Dense-only embedder → no sparse arg (hybrid transparently degrades later).
    store = FakeCodeVectorStore()
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant upsert", label="one")])]),
              owner_user_id="u1")
    assert store.last_sparse_vectors is None


def test_hybrid_mode_sends_sparse_query():
    store = FakeCodeVectorStore()
    svc = CodeKBService(embed_fn=_embed, sparse_embed_fn=_sparse_embed, vector_store=store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant upsert", label="one")])]),
              owner_user_id="u1")
    svc.search(CodeSearchRequest(query="qdrant upsert", mode="hybrid", top_k=3),
               owner_user_id="u1")
    assert store.last_mode == "hybrid"
    assert store.last_sparse_query is not None


def test_keyword_mode_routes_sparse():
    store = FakeCodeVectorStore()
    svc = CodeKBService(embed_fn=_embed, sparse_embed_fn=_sparse_embed, vector_store=store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant upsert", label="one")])]),
              owner_user_id="u1")
    svc.search(CodeSearchRequest(query="qdrant", mode="keyword", top_k=3), owner_user_id="u1")
    assert store.last_mode == "keyword"


def test_mode_degrades_to_dense_without_sparse():
    # No sparse_embed_fn → even mode=hybrid must fall back to dense (never send None sparse).
    store = FakeCodeVectorStore()
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant upsert", label="one")])]),
              owner_user_id="u1")
    svc.search(CodeSearchRequest(query="qdrant upsert", mode="hybrid", top_k=3),
               owner_user_id="u1")
    assert store.last_mode == "dense"
    assert store.last_sparse_query is None


def test_keyword_without_sparse_returns_an_explicit_unavailable_result_not_dense_noise():
    """Exact symbol search must never silently return a semantic near-neighbour.

    This reproduces the Axon failure mode: a caller asks for ``RequestTypeCompact`` in
    keyword mode, sparse retrieval is unavailable, and the old code quietly ran dense
    retrieval instead, producing unrelated low-score files.
    """
    store = FakeCodeVectorStore()
    store.supports_sparse = False
    svc = CodeKBService(embed_fn=_embed, vector_store=store)
    svc.index(_req(files=[
        _file("tools/main.go", [_chunk("unrelated request launcher", label="main")], language="go"),
    ]), owner_user_id="u1")

    response = svc.search(CodeSearchRequest(
        query="RequestTypeCompact", mode="keyword", top_k=5), owner_user_id="u1")

    assert response.hits == []
    assert response.total == 0
    assert response.requested_mode == "keyword"
    assert response.effective_mode == "unavailable"
    assert response.notice is not None and "does not substitute dense" in response.notice


def test_rerank_reorders_by_cross_encoder():
    # Wave-2: rerank=True rescores with the cross-encoder; a lower dense hit with the
    # exact target wins. Inject a fake reranker scoring by presence of "TARGET".
    class FakeReranker:
        def score_pairs(self, query, texts):
            return [1.0 if "TARGET" in t else 0.0 for t in texts]

    store = FakeCodeVectorStore()
    svc = CodeKBService(embed_fn=_embed, vector_store=store, reranker=FakeReranker())
    svc.index(_req(files=[
        _file("hot.py", [_chunk("qdrant upsert vector", code="def a(): TARGET", label="hit")]),
        _file("cold.py", [_chunk("qdrant upsert vector wait embed", code="def b(): pass", label="miss")]),
    ]), owner_user_id="u1")
    hits = svc.search(CodeSearchRequest(query="qdrant upsert", rerank=True, top_k=5),
                      owner_user_id="u1").hits
    assert hits[0].path == "hot.py", "reranked TARGET chunk must rise to the top"
    assert hits[0].score == 1.0


@pytest.mark.parametrize(
    "invalid_scores",
    [
        None,
        0.5,
        object(),
        "0.5",
        [0.5],
        ["not-a-score", "still-not-a-score"],
        [math.nan, 0.5],
        [math.inf, 0.5],
        [-math.inf, 0.5],
    ],
)
def test_rerank_malformed_scores_fail_open_without_mutating_retrieval_order(invalid_scores):
    """A malformed cross-encoder response must not make a partial rerank look valid."""
    class BadReranker:
        def score_pairs(self, _query, _texts):
            return invalid_scores

    hits = [
        CodeHit(path="first.go", start_line=1, end_line=2, label="first",
                score=0.9, code="first"),
        CodeHit(path="second.go", start_line=3, end_line=4, label="second",
                score=0.4, code="second"),
    ]
    service = CodeKBService(embed_fn=_embed, vector_store=FakeCodeVectorStore(),
                            reranker=BadReranker())

    out = service._rerank("secret-query-must-not-appear-in-logs", hits)

    assert out == hits
    assert [(hit.path, hit.score) for hit in out] == [
        ("first.go", 0.9), ("second.go", 0.4),
    ]


def test_rerank_failure_does_not_log_the_query_text(caplog):
    class ExplodingReranker:
        def score_pairs(self, query, _texts):
            raise RuntimeError(f"bad reranker input: {query}")

    hits = [CodeHit(path="a.go", start_line=1, end_line=2, score=0.6, code="a")]
    service = CodeKBService(embed_fn=_embed, vector_store=FakeCodeVectorStore(),
                            reranker=ExplodingReranker())

    assert service._rerank("private-query-token", hits) == hits
    assert "private-query-token" not in caplog.text


def test_rerank_score_tie_keeps_the_higher_retrieval_score_first():
    class TieReranker:
        def score_pairs(self, _query, _texts):
            return [0.75, 0.75]

    lower_retrieval = CodeHit(path="generated.pb.go", start_line=1, end_line=2,
                              label="Generated", score=0.4, code="SendBPS")
    higher_retrieval = CodeHit(path="hysteria.go", start_line=1, end_line=2,
                               label="HysteriaCCRX", score=0.9, code="SendBPS")
    service = CodeKBService(embed_fn=_embed, vector_store=FakeCodeVectorStore(),
                            reranker=TieReranker())

    reranked = service._rerank("Hysteria-CC-RX SendBPS", [lower_retrieval, higher_retrieval])

    assert [hit.path for hit in reranked] == ["hysteria.go", "generated.pb.go"]


def test_equal_retrieval_scores_prefer_exact_query_terms_in_label_then_stable_location():
    """Equal Qdrant/RRF scores must not let generated protobuf order drift to first."""
    target = {
        "score": 0.5, "path": "experimental/Hysteria-CC-RX/config.go",
        "start_line": 12, "end_line": 20, "label": "Hysteria-CC-RX",
        "language": "go", "project": "sing-box", "host": "h",
        "code": "func SendBPS() {}",
    }
    generated = {
        "score": 0.5, "path": "experimental/proto/generated.pb.go",
        "start_line": 1, "end_line": 8, "label": "GeneratedMessage",
        "language": "go", "project": "sing-box", "host": "h",
        "code": "func SendBPS() {}",
    }

    class TieStore:
        supports_sparse = False

        def __init__(self, raw):
            self.raw = raw

        def search(self, **_kwargs):
            return self.raw

    query = "Hysteria-CC-RX SendBPS"
    first = CodeKBService(embed_fn=_embed, vector_store=TieStore([generated, target])).search(
        CodeSearchRequest(query=query, project="sing-box", mode="dense", top_k=2),
        owner_user_id="u1",
    )
    second = CodeKBService(embed_fn=_embed, vector_store=TieStore([target, generated])).search(
        CodeSearchRequest(query=query, project="sing-box", mode="dense", top_k=2),
        owner_user_id="u1",
    )

    assert [hit.path for hit in first.hits] == [
        "experimental/Hysteria-CC-RX/config.go", "experimental/proto/generated.pb.go",
    ]
    assert [hit.path for hit in second.hits] == [hit.path for hit in first.hits]


def test_vector_store_routes_modes_to_qdrant_retrievers():
    # CodeVectorStore.search dispatches dense/hybrid/keyword to the right QdrantStore call.
    from app.services.code_kb.vector_store import CodeVectorStore
    inner = _SpyInnerStore()
    cvs = CodeVectorStore(store=inner)
    cvs.search(dense_vector=[0.1], top_k=3, owner_user_id="u1", mode="dense")
    cvs.search(dense_vector=[0.1], top_k=3, owner_user_id="u1",
               sparse_vector={"1": 2.0}, mode="hybrid")
    cvs.search(dense_vector=[0.1], top_k=3, owner_user_id="u1",
               sparse_vector={"1": 2.0}, mode="keyword")
    assert inner.calls == ["dense", "hybrid", "keyword"]


def test_vector_store_lists_unique_paths_with_qdrant_facet():
    from app.services.code_kb.vector_store import CodeVectorStore

    class FacetClient:
        def __init__(self):
            self.calls = []

        def facet(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(hits=[
                SimpleNamespace(value="src/a.py"),
                SimpleNamespace(value="src/b.py"),
            ])

        def scroll(self, *_args, **_kwargs):
            raise AssertionError("path enumeration must not scroll code chunks")

    class FacetStore:
        supports_sparse = True

        def __init__(self):
            self.client = FacetClient()

        @staticmethod
        def _build_filter(filters):
            return filters

    inner = FacetStore()
    store = CodeVectorStore(store=inner)

    assert store.list_paths(owner_user_id="u1", project="proj") == {
        "src/a.py", "src/b.py",
    }
    assert inner.client.calls[0]["key"] == "path"
    assert inner.client.calls[0]["exact"] is True


def test_vector_store_lists_owner_paths_with_one_path_facet():
    from app.services.code_kb.vector_store import CodeVectorStore

    class FacetClient:
        def facet(self, **kwargs):
            assert kwargs["facet_filter"] == {"owner_user_id": "u1"}
            return SimpleNamespace(hits=[SimpleNamespace(value="shared/main.py")])

        def scroll(self, *_args, **_kwargs):
            raise AssertionError("cross-project search must not scan owner chunks")

    inner = SimpleNamespace(
        supports_sparse=True,
        client=FacetClient(),
        _build_filter=lambda filters: filters,
    )
    store = CodeVectorStore(store=inner)

    assert store.list_owner_paths(owner_user_id="u1") == {"shared/main.py"}


def test_vector_store_lists_owner_files_with_facets_not_chunk_scroll():
    from app.services.code_kb.vector_store import CodeVectorStore

    class FacetClient:
        def facet(self, **kwargs):
            if kwargs["key"] == "project":
                return SimpleNamespace(hits=[
                    SimpleNamespace(value="/repo"),
                    SimpleNamespace(value="/repo/child"),
                ])
            project = kwargs["facet_filter"]["project"]
            paths = {
                "/repo": ["child/shared.py", "parent.py"],
                "/repo/child": ["shared.py", "child.py"],
            }[project]
            return SimpleNamespace(
                hits=[SimpleNamespace(value=path) for path in paths])

        def scroll(self, *_args, **_kwargs):
            raise AssertionError("fan-out discovery must not scan owner chunks")

    inner = SimpleNamespace(
        supports_sparse=True,
        client=FacetClient(),
        _build_filter=lambda filters: filters,
    )
    store = CodeVectorStore(store=inner)

    assert store.list_owner_files(owner_user_id="u1") == {
        ("/repo", "child/shared.py"),
        ("/repo", "parent.py"),
        ("/repo/child", "shared.py"),
        ("/repo/child", "child.py"),
    }


def test_facet_failure_does_not_fall_back_to_chunk_scroll():
    from app.services.code_kb.vector_store import CodeVectorStore

    class BrokenFacetClient:
        def facet(self, **_kwargs):
            raise RuntimeError("path payload index unavailable")

        def scroll(self, *_args, **_kwargs):
            raise AssertionError("chunk scroll fallback must stay disabled")

    inner = SimpleNamespace(
        supports_sparse=True,
        client=BrokenFacetClient(),
        _build_filter=lambda filters: filters,
    )
    store = CodeVectorStore(store=inner)

    with pytest.raises(RuntimeError, match="payload index unavailable"):
        store.list_paths(owner_user_id="u1", project="proj")


def test_transient_facet_failure_is_retried_without_chunk_scroll(monkeypatch):
    from app.services.code_kb import vector_store as vector_store_module
    from app.services.code_kb.vector_store import CodeVectorStore

    class RecoveringFacetClient:
        def __init__(self):
            self.calls = 0

        def facet(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary qdrant timeout")
            return SimpleNamespace(hits=[SimpleNamespace(value="src/recovered.py")])

        def scroll(self, *_args, **_kwargs):
            raise AssertionError("chunk scroll fallback must stay disabled")

    monkeypatch.setattr(vector_store_module.time, "sleep", lambda _seconds: None)
    client = RecoveringFacetClient()
    inner = SimpleNamespace(
        supports_sparse=True,
        client=client,
        _build_filter=lambda filters: filters,
    )
    store = CodeVectorStore(store=inner)

    assert store.list_paths(owner_user_id="u1", project="proj") == {
        "src/recovered.py",
    }
    assert client.calls == 2


def test_vector_store_creates_only_missing_search_payload_indexes():
    from app.services.code_kb.vector_store import CodeVectorStore

    class IndexClient:
        def __init__(self):
            self.created = []

        def get_collection(self, _collection_name):
            return SimpleNamespace(payload_schema={"owner_user_id": {}})

        def create_payload_index(self, **kwargs):
            self.created.append(kwargs["field_name"])

    inner = SimpleNamespace(supports_sparse=True, client=IndexClient())
    store = CodeVectorStore(store=inner)
    store.ensure_search_payload_indexes()

    assert inner.client.created == ["project", "path", "language"]


def test_payload_index_failure_does_not_skip_later_fields(monkeypatch):
    from app.services.code_kb import vector_store as vector_store_module
    from app.services.code_kb.vector_store import CodeVectorStore

    class IndexClient:
        def __init__(self):
            self.attempted = []

        def get_collection(self, _collection_name):
            return SimpleNamespace(payload_schema={})

        def create_payload_index(self, **kwargs):
            field_name = kwargs["field_name"]
            self.attempted.append(field_name)
            if field_name == "project":
                raise RuntimeError("project index unavailable")

    monkeypatch.setattr(
        vector_store_module, "_with_retry", lambda _what, operation: operation())
    inner = SimpleNamespace(supports_sparse=True, client=IndexClient())
    store = CodeVectorStore(store=inner)
    store.ensure_search_payload_indexes()

    assert inner.client.attempted == ["owner_user_id", "project", "path", "language"]
    assert store._missing_search_indexes == {"project"}


def test_missing_path_index_is_retried_before_facet(monkeypatch):
    from app.services.code_kb import vector_store as vector_store_module
    from app.services.code_kb.vector_store import CodeVectorStore

    class RecoveringClient:
        def __init__(self):
            self.schema = {
                "owner_user_id": {}, "project": {}, "language": {},
            }
            self.created = []

        def get_collection(self, _collection_name):
            return SimpleNamespace(payload_schema=self.schema)

        def create_payload_index(self, **kwargs):
            field_name = kwargs["field_name"]
            self.created.append(field_name)
            self.schema[field_name] = {}

        def facet(self, **_kwargs):
            assert "path" in self.schema
            return SimpleNamespace(hits=[SimpleNamespace(value="src/recovered.py")])

    monkeypatch.setattr(
        vector_store_module, "_with_retry", lambda _what, operation: operation())
    client = RecoveringClient()
    inner = SimpleNamespace(
        supports_sparse=True, client=client, _build_filter=lambda filters: filters)
    store = CodeVectorStore(store=inner)
    store._missing_search_indexes = {"path"}
    store._next_payload_index_retry_at = 0.0

    assert store.list_paths(owner_user_id="u1", project="proj") == {
        "src/recovered.py",
    }
    assert client.created == ["path"]


# --------------------------------------------------------------------------- #
# Incremental indexing (W157-inc) — the invariant: incremental result MUST equal
# a full reindex of the current tree (the "no drift after append" the user flagged).
# --------------------------------------------------------------------------- #
def _ireq(project="proj", host="h1", files=None, manifest=None, removed_paths=None):
    return CodeIndexRequest(project=project, host=host, root_dir="/repo",
                            files=files or [], incremental=True, manifest=manifest,
                            removed_paths=removed_paths or [])


def _point_key(p):
    # Mirrors vector_store.point_id inputs: identity of a stored chunk.
    return (p["path"], p["start_line"], p["label"])


def test_incremental_upsert_failure_preserves_previous_file_snapshot():
    class PartiallyFailingStore(FakeCodeVectorStore):
        fail_upsert = False

        def upsert(self, **kwargs):
            if not self.fail_upsert:
                return super().upsert(**kwargs)
            partial = dict(kwargs)
            partial["items"] = kwargs["items"][:1]
            partial["dense_vectors"] = kwargs["dense_vectors"][:1]
            if kwargs.get("sparse_vectors") is not None:
                partial["sparse_vectors"] = kwargs["sparse_vectors"][:1]
            super().upsert(**partial)
            raise RuntimeError("partial qdrant upsert failure")

    store = PartiallyFailingStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [
        _chunk("qdrant old one", label="old_one", start=1),
        _chunk("qdrant old two", label="old_two", start=20),
    ])]), owner_user_id="u1")

    store.fail_upsert = True
    with pytest.raises(RuntimeError, match="partial qdrant upsert failure"):
        svc.index(_ireq(files=[_file("a.py", [
            _chunk("qdrant new", label="new", start=40),
        ])], manifest=["a.py"]), owner_user_id="u1")

    labels = {
        point["label"] for point in store.points
        if point["owner_user_id"] == "u1"
        and point["project"] == "proj"
        and point["path"] == "a.py"
    }
    assert {"old_one", "old_two"}.issubset(labels)


def _snapshot(store, owner="u1", project="proj"):
    return {(_point_key(p), p["code"]) for p in store.points
            if p["owner_user_id"] == owner and p["project"] == project}


def test_incremental_adds_new_file_without_wiping():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="a")])]), owner_user_id="u1")
    svc.index(_ireq(files=[_file("b.py", [_chunk("embed", label="b")])],
                    manifest=["a.py", "b.py"]), owner_user_id="u1")
    paths = {p["path"] for p in store.points}
    assert paths == {"a.py", "b.py"}, "incremental add must not wipe the untouched file"


def test_incremental_replaces_changed_file_leaving_no_orphans():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    # first index: file with 3 chunks at lines 1/2/3
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant one", start=1, label="one"),
                                         _chunk("qdrant two", start=2, label="two"),
                                         _chunk("qdrant three", start=3, label="three")])]),
              owner_user_id="u1")
    assert store.count(owner_user_id="u1", project="proj") == 3
    # file shrank to a single chunk → old line-2/3 points must be purged, not orphaned
    svc.index(_ireq(files=[_file("a.py", [_chunk("qdrant one", start=1, label="one")])],
                    manifest=["a.py"]), owner_user_id="u1")
    rows = [p for p in store.points if p["path"] == "a.py"]
    assert len(rows) == 1 and rows[0]["label"] == "one", "shrunk file must leave no orphan chunks"


def test_incremental_removes_deleted_file_via_manifest():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="a")]),
                          _file("b.py", [_chunk("embed", label="b")])]), owner_user_id="u1")
    # b.py deleted from the repo → manifest omits it → its points must go
    svc.index(_ireq(files=[], manifest=["a.py"]), owner_user_id="u1")
    assert {p["path"] for p in store.points} == {"a.py"}


def test_incremental_removes_explicit_paths_without_full_manifest():
    """Filesystem watchers know exactly which path disappeared. They must not have to
    enumerate and hash the whole repository merely to delete that one file."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="a")]),
                          _file("b.py", [_chunk("embed", label="b")])]), owner_user_id="u1")

    svc.index(_ireq(removed_paths=["b.py"]), owner_user_id="u1")

    assert {p["path"] for p in store.points} == {"a.py"}


def test_incremental_no_manifest_is_additive_only():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="a")]),
                          _file("b.py", [_chunk("embed", label="b")])]), owner_user_id="u1")
    # No manifest → we must NOT infer deletions (can't tell removed from not-sent).
    svc.index(_ireq(files=[_file("a.py", [_chunk("qdrant changed", label="a")])],
                    manifest=None), owner_user_id="u1")
    assert {p["path"] for p in store.points} == {"a.py", "b.py"}


def test_incremental_empty_and_no_manifest_is_noop():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="a")])]), owner_user_id="u1")
    resp = svc.index(_ireq(files=[], manifest=None), owner_user_id="u1")
    assert resp.indexed == 0
    assert store.count(owner_user_id="u1", project="proj") == 1  # nothing wiped


def test_incremental_result_equivalent_to_full_reindex():
    """The anchor invariant: after a sequence of incremental edits, the point set must
    be byte-identical to a from-scratch full index of the final tree."""
    # Final tree: a.py (edited), c.py (added); b.py (deleted). Start tree: a.py, b.py.
    start = [_file("a.py", [_chunk("qdrant one", start=1, label="one"),
                            _chunk("qdrant two", start=2, label="two")]),
             _file("b.py", [_chunk("embed old", start=1, label="b")])]
    final_a = _file("a.py", [_chunk("qdrant one edited", start=1, label="one")])  # shrank + changed
    final_c = _file("c.py", [_chunk("render button", start=1, label="c")])        # new

    # Path 1: incremental from the start tree to the final tree.
    inc_store = FakeCodeVectorStore()
    inc_svc = _svc(inc_store)
    inc_svc.index(_req(files=start), owner_user_id="u1")
    inc_svc.index(_ireq(files=[final_a, final_c], manifest=["a.py", "c.py"]), owner_user_id="u1")

    # Path 2: a from-scratch full index of the final tree.
    full_store = FakeCodeVectorStore()
    full_svc = _svc(full_store)
    full_svc.index(_req(files=[final_a, final_c]), owner_user_id="u1")

    assert _snapshot(inc_store) == _snapshot(full_store), "incremental must equal full reindex"


def test_incremental_refuses_disjoint_manifest():
    # Delete-safety valve: a manifest with no overlap = broken walk / wrong scope → refuse.
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="a")]),
                          _file("b.py", [_chunk("embed", label="b")])]), owner_user_id="u1")
    with pytest.raises(ValueError, match="disjoint"):
        svc.index(_ireq(files=[_file("x.py", [_chunk("render", label="x")])],
                        manifest=["x.py", "y.py"]), owner_user_id="u1")
    # nothing deleted — old index intact
    assert {p["path"] for p in store.points} == {"a.py", "b.py"}


def test_incremental_reports_stale_prune_failure_without_dropping_old_chunks():
    # Upsert-first keeps the previous complete file available. A failed stale-id prune
    # still fails the job loudly, but it leaves extra old chunks instead of data loss.
    class DeleteBoomStore(FakeCodeVectorStore):
        fail_delete = False

        def delete_point_ids(self, ids):
            if self.fail_delete and ids:
                raise RuntimeError("qdrant delete timed out")
            return super().delete_point_ids(ids)

    store = DeleteBoomStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant one", start=1, label="one"),
                                         _chunk("qdrant two", start=2, label="two")])]),
              owner_user_id="u1")
    store.fail_delete = True
    with pytest.raises(RuntimeError, match="delete timed out"):
        svc.index(_ireq(files=[_file("a.py", [_chunk("qdrant one", start=1, label="one")])],
                        manifest=["a.py"]), owner_user_id="u1")
    # The old extra chunk remains searchable; a retry can safely finish pruning it.
    rows = [p for p in store.points if p["path"] == "a.py"]
    assert len(rows) == 2


def test_full_index_reports_stale_prune_failure_without_emptying_previous_snapshot():
    class DeleteBoomStore(FakeCodeVectorStore):
        fail_delete = False

        def delete_point_ids(self, ids):
            if self.fail_delete and ids:
                raise RuntimeError("qdrant stale prune failed")
            return super().delete_point_ids(ids)

    store = DeleteBoomStore()
    svc = _svc(store)
    svc.index(_req(files=[_file("a.py", [_chunk("qdrant", label="a")])]), owner_user_id="u1")
    store.fail_delete = True

    with pytest.raises(RuntimeError, match="stale prune failed"):
        svc.index(_req(files=[_file("b.py", [_chunk("qdrant changed", label="b")])]),
                  owner_user_id="u1")
    assert store.list_paths(owner_user_id="u1", project="proj") == {"a.py", "b.py"}


def test_vector_store_delete_retries_then_raises(monkeypatch):
    # Fix A at the store layer: transient errors retried, permanent failure RAISES (not 0).
    from app.services.code_kb import vector_store as vs

    monkeypatch.setattr(vs.time, "sleep", lambda *_: None)  # no real backoff sleeps

    class Boom:
        def count(self, *a, **k):
            raise RuntimeError("down")

    class FakeInner:
        supports_sparse = True
        client = Boom()

        def _build_filter(self, d):
            return d

    cvs = vs.CodeVectorStore(store=FakeInner())
    with pytest.raises(RuntimeError, match="down"):
        cvs.delete_paths(owner_user_id="u1", project="p", paths={"a.py"})


def test_delete_project():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(project="p1", files=[_file("a.py", [_chunk("qdrant", label="a")])]),
              owner_user_id="u1")
    resp = svc.delete_project("p1", owner_user_id="u1")
    assert resp.deleted >= 1
    assert store.count(owner_user_id="u1", project="p1") == 0


def test_owner_is_never_taken_from_request_body():
    """Owner must come only from the resolved kwarg, never a body field."""
    assert "owner_user_id" not in CodeIndexRequest.model_fields
    assert "owner_user_id" not in CodeSearchRequest.model_fields


# --------------------------------------------------------------------------- #
# 方案2: cross-scope write-time fan-out sync + duplicate stats
# --------------------------------------------------------------------------- #
def test_fan_out_refreshes_same_file_in_other_scope_both_complete():
    """Indexing a file that also lives in another scope (same owner) refreshes that
    scope's copy to the latest content, reports the count, and leaves BOTH scopes
    complete (no files added or dropped on either side)."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    # Child scope /proj/sub already indexed with x.py (OLD) + its own y.py.
    svc.index(_req(project="/proj/sub", files=[
        _file("x.py", [_chunk("qdrant OLD", label="x")]),
        _file("y.py", [_chunk("embed onlychild", label="y")]),
    ]), owner_user_id="u1")
    # Index the parent /proj: sub/x.py is the SAME physical file (NEW content) + top.py.
    resp = svc.index(_req(project="/proj", files=[
        _file("sub/x.py", [_chunk("qdrant NEW", label="x")]),
        _file("top.py", [_chunk("render parentonly", label="top")]),
    ]), owner_user_id="u1")

    assert resp.duplicates_updated == 1  # only sub/x.py == /proj/sub/x.py matched
    # Parent scope complete.
    parent = {p["path"] for p in store.points if p["project"] == "/proj"}
    assert parent == {"sub/x.py", "top.py"}
    # Child scope STILL complete; shared file refreshed; child-only file untouched;
    # parent-only file NOT injected.
    child = {p["path"]: p for p in store.points if p["project"] == "/proj/sub"}
    assert set(child) == {"x.py", "y.py"}, "child must stay complete (no drop, no foreign add)"
    assert "NEW" in child["x.py"]["content"], "shared file must be refreshed to latest in child"
    assert "onlychild" in child["y.py"]["content"], "child-only file untouched"


def test_fan_out_does_not_resurrect_target_scope_deleted_after_discovery():
    class BlockingOwnerSnapshotStore(FakeCodeVectorStore):
        def __init__(self):
            super().__init__()
            self.block_owner_snapshot = False
            self.owner_snapshot_captured = threading.Event()
            self.release_owner_snapshot = threading.Event()

        def list_owner_files(self, *, owner_user_id):
            snapshot = super().list_owner_files(owner_user_id=owner_user_id)
            if self.block_owner_snapshot:
                self.owner_snapshot_captured.set()
                assert self.release_owner_snapshot.wait(5)
            return snapshot

    store = BlockingOwnerSnapshotStore()
    svc = _svc(store)
    svc.index(_req(project="/proj/sub", files=[
        _file("x.py", [_chunk("qdrant OLD", label="x")]),
    ]), owner_user_id="u1")

    store.block_owner_snapshot = True
    parent_thread = threading.Thread(target=lambda: svc.index(_req(project="/proj", files=[
        _file("sub/x.py", [_chunk("qdrant NEW", label="x")]),
    ]), owner_user_id="u1"))
    parent_thread.start()
    assert store.owner_snapshot_captured.wait(2)

    svc.delete_project("/proj/sub", owner_user_id="u1")
    store.release_owner_snapshot.set()
    parent_thread.join(timeout=5)

    assert store.count(owner_user_id="u1", project="/proj/sub") == 0


def test_fan_out_zero_when_no_other_scope_shares_file():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(project="/a", files=[_file("a.py", [_chunk("qdrant", label="a")])]),
              owner_user_id="u1")
    resp = svc.index(_req(project="/b", files=[_file("b.py", [_chunk("embed", label="b")])]),
                     owner_user_id="u1")
    assert resp.duplicates_updated == 0  # /a/a.py and /b/b.py are different files


def test_fan_out_upsert_failure_leaves_other_scope_complete():
    """CRITICAL (review #1): if syncing the other scope fails mid-op, that scope must stay
    COMPLETE (stale is fine, empty is not) — the "两方都要全" red line. Upsert-first ordering
    guarantees a failed sync deletes nothing."""
    class UpsertFailsForProject(FakeCodeVectorStore):
        def __init__(self, fail_project):
            super().__init__()
            self._fail = fail_project
        def upsert(self, *, owner_user_id, project, host, items, dense_vectors, sparse_vectors=None):
            if project == self._fail:
                raise RuntimeError("qdrant upsert timed out")
            return super().upsert(owner_user_id=owner_user_id, project=project, host=host,
                                  items=items, dense_vectors=dense_vectors, sparse_vectors=sparse_vectors)

    store = UpsertFailsForProject(fail_project=None)
    svc = _svc(store)
    # Populate both scopes normally (no failure yet).
    svc.index(_req(project="/proj/sub", files=[_file("x.py", [_chunk("qdrant OLD", label="x")])]),
              owner_user_id="u1")
    svc.index(_req(project="/proj", files=[_file("top.py", [_chunk("render top", label="top")])]),
              owner_user_id="u1")
    # Now make fan-out's upsert into the child scope fail, and re-index the parent's sub/x.py.
    store._fail = "/proj/sub"
    resp = svc.index(_req(project="/proj", files=[
        _file("sub/x.py", [_chunk("qdrant NEW", label="x")]),
        _file("top.py", [_chunk("render top", label="top")]),
    ]), owner_user_id="u1")
    # Fan-out swallowed the failure (best-effort) — primary index still succeeded.
    assert resp.duplicates_updated == 0
    # Child scope must NOT be emptied: x.py is still present (with its OLD content — stale, complete).
    child = {p["path"]: p for p in store.points if p["project"] == "/proj/sub"}
    assert "x.py" in child, "failed fan-out must never delete the other scope's file"
    assert "OLD" in child["x.py"]["content"], "old content preserved when sync fails"


def test_full_reindex_fan_out_removes_deleted_file_from_nested_scope():
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(project="/repo/child", files=[
        _file("a.py", [_chunk("qdrant child old", label="child_old")]),
    ]), owner_user_id="u1")
    svc.index(_req(project="/repo", files=[
        _file("child/a.py", [_chunk("qdrant parent old", label="parent_old")]),
        _file("keep.py", [_chunk("qdrant keep", label="keep")]),
    ]), owner_user_id="u1")

    svc.index(_req(project="/repo", files=[
        _file("keep.py", [_chunk("qdrant keep refreshed", label="keep")]),
    ]), owner_user_id="u1")

    assert "a.py" not in store.list_paths(
        owner_user_id="u1", project="/repo/child")


def test_fan_out_is_owner_isolated():
    """A file with the same abspath under a DIFFERENT owner is never touched."""
    store = FakeCodeVectorStore()
    svc = _svc(store)
    svc.index(_req(project="/proj/sub", files=[_file("x.py", [_chunk("qdrant OLD", label="x")])]),
              owner_user_id="u2")  # someone else's identical path
    resp = svc.index(_req(project="/proj", files=[_file("sub/x.py", [_chunk("qdrant NEW", label="x")])]),
                     owner_user_id="u1")
    assert resp.duplicates_updated == 0
    other = [p for p in store.points if p["owner_user_id"] == "u2"]
    assert other and "OLD" in other[0]["content"], "another owner's copy must be untouched"
