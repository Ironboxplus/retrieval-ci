"""Integration tests for the drop-in ACE MCP tools.

These exercise the @mcp.tool() functions directly against a PlaybookStore backed by
a deterministic fake embedder, so no backend / GPU is touched. They verify the
ACE loop: reflect (curate) -> context/enhance/generate (retrieve & ground).
"""
from __future__ import annotations

import asyncio
import hashlib
import math

import mcp_server.server as server
from mcp_server.playbook import PlaybookStore

_DIM = 48


async def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for w in t.lower().replace("_", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v] if n else v)
    return out


def _use_fresh_playbook(tmp_path):
    server._playbook = PlaybookStore(db_path=str(tmp_path / "pb.db"), embed_fn=_fake_embed)


def test_reflect_then_context_and_view(tmp_path):
    _use_fresh_playbook(tmp_path)

    async def run():
        out = await server.ace_smart_reflect(
            "Always wait=True on Qdrant upsert or batches drop silently", domain="qdrant")
        assert "added new" in out

        # a near-duplicate reinforces the existing bullet (grow-and-refine)
        out2 = await server.ace_smart_reflect(
            "Always wait=True on Qdrant upsert or batches drop silently", domain="qdrant")
        assert "reinforced existing" in out2

        ctx = await server.ace_context_aware("qdrant upsert reliability", domain="qdrant")
        assert "wait=True" in ctx

        view = await server.ace_playbook()
        assert "Playbook (1 bullets)" in view  # dedup kept it at one

    asyncio.run(run())


def test_enhance_and_generate_attach_guidelines(tmp_path):
    _use_fresh_playbook(tmp_path)

    async def run():
        await server.ace_smart_reflect("Use astchunk for AST-aware code chunking",
                                       section="best_practices")
        enhanced = await server.ace_enhance_prompt("chunk this repo's code")
        assert "Playbook guidelines" in enhanced
        assert "astchunk" in enhanced

        gen = await server.ace_smart_generate("how should I chunk code?")
        assert "astchunk" in gen
        assert "enhanced prompt" in gen

    asyncio.run(run())


def test_enhance_prompt_is_noop_when_playbook_empty(tmp_path):
    _use_fresh_playbook(tmp_path)

    async def run():
        assert await server.ace_enhance_prompt("anything") == "anything"

    asyncio.run(run())


def test_reflect_rejects_empty(tmp_path):
    _use_fresh_playbook(tmp_path)

    async def run():
        assert "Nothing to reflect" in await server.ace_smart_reflect("   ")

    asyncio.run(run())


class _FakePlaybookClient:
    """Stub RetrievalClient exposing only the playbook methods ServerPlaybook needs."""

    def __init__(self):
        self.records = []
        self.bullets = []

    async def playbook_record(self, *, project_slug, content, section="insights", domain=None):
        self.records.append({"project_slug": project_slug, "content": content,
                             "section": section, "domain": domain})
        row = {"id": len(self.records), "content": content, "section": section,
               "domain": domain, "helpful": 0, "harmful": 0, "deduped": False}
        self.bullets.append(row)
        return row

    async def playbook_search(self, *, project_slug, query, top_k=5, domain=None):
        hits = [dict(b, score=1.0) for b in self.bullets if query.split()[0] in b["content"]]
        return {"bullets": hits[:top_k], "total": len(hits[:top_k])}

    async def playbook_list(self, *, project_slug=None, domain=None, limit=200):
        return {"bullets": self.bullets[:limit], "total": len(self.bullets)}


def test_ace_tools_use_server_playbook_when_no_local(monkeypatch):
    """With no injected local store, ACE tools must go through the SERVER playbook
    (so the admin console reflects real usage)."""
    server._playbook = None
    fake = _FakePlaybookClient()
    monkeypatch.setattr(server, "_client", fake)

    async def run():
        out = await server.ace_smart_reflect("qdrant upsert needs wait=True", domain="qdrant")
        assert "added new" in out
        # playbook scope is FORCED to the cwd abspath — never a hand-typed project name.
        assert fake.records and fake.records[0]["project_slug"] == server._scope_path(".")

        ctx = await server.ace_context_aware("qdrant reliability")
        assert "wait=True" in ctx

        view = await server.ace_playbook()
        assert "Playbook (1 bullets)" in view

    asyncio.run(run())
    server._playbook = None  # don't leak the None override into other tests
