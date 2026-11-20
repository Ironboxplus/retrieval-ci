# Retrieval MCP Server

> Platform architecture: [../overview.md](../overview.md). PyPI distribution: [../mcp_plugin/README.md](../mcp_plugin/README.md). Current stable client is `0.5.14` (2026-11-15), 35 tools: `0.5.14` runs automatic indexing at most once per 5 minutes per scope (changes in between go out as one batch; explicit `index_code` is never throttled); `0.5.13` (2026-11-13) makes `index_code` index documents too (they go into the journal and are searched with `journal_search`; `journal_index_dir` is deprecated); `0.5.12` shows the server's one-time duplicate-code summary after indexing; `0.5.11` makes the initialize instructions a rule (first search must be a Retrieval call); `0.5.7` restores the plain file SHA-256 as the incremental identity (re-index once with a full `index_code`; unchanged chunks keep their stored vectors); `0.5.6` fixes code chunking (chonkie byte offsets shifted chunks after non-ASCII text; sources over 384K characters were one chunk); `0.5.4` bounds `index_inventory` to one line per project, at most `limit` (default 40), with a `project` path filter; `0.5.3` reuses exact private vectors during public import (same tool surface); `0.5.2` adds `import_public_code_repo` (upload a clean local clone as a git bundle; the server builds the mirror and indexes it); `0.5.1` only makes rejected requests carry the backend's reason (and lists valid public `source_ids`/`domains` when one is unknown); `0.5.0` folds `search_all_code` and `search_public_code` into `search_code` (`repo_root` list or `"*"`, `sources="private"|"public"|"all"`; breaking) and rewrites the code/journal search descriptions around their advantages over rg/grep. `0.4.19` publishes text-only results (no automatic `{"result": string}` outputSchema, no duplicate structured copy of each result), makes sibling tools describe only their differences and drops Args notes that repeat the schema (47,883 → 37,417 characters, 21 description digests). `0.4.18` only compacts the advertised surface (no schema titles, the list-query contract on `query`, trivial examples removed; −5,548 characters). `0.4.17` is the `0.4.16` surface plus `search_arxiv`, which scopes `/api/search` to `venue_tier="preprint"` (arXiv) now that `search_papers` leaves preprints out. `0.4.16` kept the `0.4.15` surface and treats `cancelled` as a finished code index job state. `0.4.15` added `search_all_code` (35 tools, same digest for the other 34) — one call that fuses the private Code KB and the curated Public Code corpus into a single RRF/reranked ranking, tagging each hit `[private]` or `[public:<source_id>@<rev>]`; the public leg still requires admin tier. `0.4.14` keeps the same 34 names/signatures and only makes uploaded documents reachable past search — `papers` gives upload/webpage payloads no `paper_id` key, so results now advertise `doc:<documents.id>`, which routes to the `doc_ids` filter and to `/api/documents/{id}/fulltext|markdown`. PyPI `0.4.11` is blocked because Python 3.12 renders its dynamically extended descriptions with a different daemon digest; do not use it. `0.4.12` retains Retrieval-first instructions and ordered multi-query input while canonicalizing descriptions before suffixing; Python 3.10--3.13 surface and Python 3.12 official-PyPI stdio gates passed. `_session_cwd` / `_session_abspath` / `_scope_path` / `_local_repo_root` in this file are the vendored twins of the standalone plugin's copies and are asserted byte-identical.

`0.5.3` (2026-10-08): `import_public_code_repo` automatically
sends the clone's absolute path as `private_project`. The backend checks the
authenticated creator's private index and reuses vectors only for chunks matching
the published Git revision exactly; missing, changed or incomplete vectors are
embedded normally. Job results expose `reused_chunk_count`, and private data remains
in its original scope. The advertised MCP tool surface is unchanged.

`0.5.4` (2026-10-09): `index_inventory(host, project, limit=40)` lists scopes needing
attention first, then the newest, one line each; `_format_inventory_summary` in
`backend.py` is the asserted byte-identical twin of the plugin's.

Exposes the Retrieval paper, Journal, Code KB, Index Catalog, ACE and artifact APIs
as **MCP tools**, so any MCP client —
Claude Code, Cursor, Codex, Claude Desktop — can search the corpus directly.

Corpus size is dynamic; use `corpus_stats` instead of copying a hard-coded paper/chunk count.

## Architecture: HTTP client over the existing backend

This server is a **standalone process** that calls the running FastAPI backend
over HTTP. It does **not** import `core/`, `app/`, or `main.py`, and it does **not**
load BGE-M3 — it reuses the warm model already serving `/api/search`. So nothing on
the GPU is duplicated and the core system is untouched.

```
MCP client ──stdio──▶ mcp_server.server ──HTTP──▶ FastAPI backend ──▶ Qdrant + BGE-M3
(Claude Code,                (this dir)            (:8000 / 4090)
 Cursor, …)
```

## Tools

The 0.4.13 candidate exposes 34 tools: 10 paper tools, 6 Journal tools, 5 ACE tools,
6 private Code KB tools, and 7 Public Code tools. 0.4.15 (the current surface,
also in 0.4.16) adds one more: `search_all_code`, which fuses the private Code KB and Public Code
tools into a single query. Paper artifacts are read-only and never mutate the paper
database or Qdrant collections.

### Paper retrieval (shared academic corpus, read-only)

| Tool | What it does |
|------|--------------|
| `search_papers(query, top_k, venue?, year?, title_only?, source_type?)` | Semantic hybrid search over the whole corpus; returns papers + paper_id + snippet. `source_type="upload"` restricts to uploaded documents (whole books), `"paper"` to the CS corpus, also `"webpage"`/`"review"` |
| `search_within_paper(paper_id, query, top_k)` | Every matching passage inside one paper, with scores |
| `get_paper_markdown(paper_id, save_path?, overwrite?)` | Return complete extracted Markdown verbatim, or save it on the MCP host |
| `download_paper_pdf(paper_id, save_path?, overwrite?)` | Stream the original PDF to an atomic local file; defaults to `<paper_id>.pdf` |
| `download_papers(paper_ids, save_dir?, artifacts?, overwrite?)` | Download PDF, Markdown, or both for 1-10 papers with bounded concurrency and per-artifact results |
| `match_document(text, top_k)` | Given a passage, find content-nearest papers (related-work discovery) |
| `match_paper(paper_id, top_k)` | Queue exact content-nearest neighbours of an indexed paper (self excluded) |
| `match_paper_status(task_id, cancel?)` | Poll/cancel the durable job; completed jobs return the full result |
| `list_conferences()` | Venues + CCF tier + years available (valid `venue` values) |
| `corpus_stats()` | Papers / chunks / vectors / embedding model |

### Journal work-memory (owner-scoped)

| Tool | What it does |
|------|--------------|
| `journal_record(...)` | Record an owner-scoped work-memory entry or a current file snapshot |
| `journal_search(...)` | Search owner-scoped Journal memory and indexed document chunks |
| `journal_recent(...)` | List recent Journal entries for a project scope |
| `journal_index_dir(...)` | Start a latest-wins asynchronous bulk index of a local document directory |
| `journal_index_status(job_id)` | Poll an asynchronous Journal directory-index job |
| `delete_journal_docs(dir)` | Explicitly remove batch-indexed Journal documents for one directory scope |

### ACE — evolving playbook (drop-in compatible with angry-robot-deals/ace-mcp)

A per-project playbook of reusable lessons (strategies, pitfalls, conventions)
that the agent curates over time — ACE's evolving *context* (arXiv:2510.04618).
The production server stores it centrally behind authenticated `/api/playbook/*`
routes, with gateway-owned user isolation. Embeddings and reranking reuse the
backend's existing model singletons, so no second model is loaded. ACE has no code
generator of its own: these tools curate and retrieve guidance; your agent does the
generating.

| Tool | What it does |
|------|--------------|
| `ace_smart_reflect(code, ...)` | Curate a transferable lesson into the playbook (Reflector→Curator). Near-duplicates reinforce the existing bullet (grow-and-refine), preventing context collapse |
| `ace_context_aware(query, domain?, top_k?)` | Retrieve the playbook bullets most relevant to what you're about to do |
| `ace_enhance_prompt(prompt, focus_area?)` | Return the prompt augmented with relevant playbook guidelines (no LLM call) |
| `ace_smart_generate(prompt, context?, auto_enhance?)` | Return playbook-grounded guidance + an enhanced prompt for **your** model to generate from |
| `ace_playbook(domain?, limit?)` | View the current project playbook with helpful/harmful counters |

### Code KB — search your own code (local chunking, server-side index)

Indexes a repo's code into the backend's shared `code_chunks` Qdrant collection and
searches it by meaning. Scanning, gitignore filtering, hashing, and AST chunking run
client-side, so the backend receives chunks rather than filesystem access. The
backend owns embedding, incremental reconciliation, jobs, catalog state, and vector
storage under the authenticated owner/host/absolute-scope key.

| Tool | What it does |
|------|--------------|
| `index_code(repo_root, incremental?, respect_gitignore?)` | Start asynchronous local preparation and server indexing for an absolute directory scope |
| `index_code_status(job_id?)` | Read local preparation plus backend embedding/upsert progress |
| `search_code(query, repo_root, ...)` | Hybrid/dense/keyword search with path, language, rerank, and per-file controls |
| `find_duplicate_code(repo_root, ...)` | File pairs sharing near-duplicate chunks, ranked by shared-chunk count |
| `delete_code(repo_root)` | Explicitly remove one code scope from vectors and the index catalog |
| `index_inventory()` | List indexed code/document scopes by owner, host, path, file count, and freshness |

### Public Code — shared curated repositories (admin-managed)

| Tool | What it does |
|------|--------------|
| `list_public_code_sources(...)` | List source/revision/coverage/update state |
| `add_public_code_source(...)` | Register an HTTPS upstream using profile references only |
| `check_public_code_updates(...)` | Queue remote revision checks |
| `index_public_code_sources(...)` | Queue mirror refresh, AST chunking, embedding, and promotion |
| `set_public_code_update_policy(...)` | Set manual or scheduled refresh policy |
| `public_code_job_status(...)` | Poll durable source/file/chunk progress |
| `search_code(sources="public", ...)` | Search one repository, several repositories, or whole domains |

## Prerequisites

- The Retrieval backend must be running and reachable.
- `mcp` and `httpx` installed in the same env (already present in the project venv).

## Run and singleton boundary

```bash
# Normal client entry: a thin stdio proxy that auto-starts/reuses the one local daemon.
retrieval-mcp

# Explicit direct FastMCP server for debugging/protocol compatibility only.
python -m mcp_server.server stdio-server

# The repo module's no-argument mode is also the thin proxy, not a second server.
python -m mcp_server.server
```

The daemon owns the full 33-tool FastMCP surface and is reachable only through
the local proxy/control plane. An ordinary client must keep its existing stdio
command configuration; it must not be changed to a visible URL or transport
configuration, and it does not need a manual `start` command.

| Env | Default | Meaning |
|-----|---------|---------|
| `RETRIEVAL_API_URL` | `http://127.0.0.1:8000` | backend base URL |
| `RETRIEVAL_API_KEY` | - | Optional Bearer key for authenticated gateway access |

## Wire it into a client

```json
{
  "mcpServers": {
    "retrieval": {
      "command": "retrieval-mcp",
      "args": []
    }
  }
}
```

Install `retrieval-mcp` in the client environment once. Existing `uvx` and repo
module command configurations remain compatible command-mode shims, but they add
a wrapper/import path per client and are not the preferred setup.

## Tests

```bash
python -m pytest tests/test_mcp_server.py tests/test_mcp_protocol.py \
  tests/test_playbook.py tests/test_ace_tools.py -q
```

All hermetic (no 4090 / BGE-M3 needed — embeds are faked/stubbed):
- `test_mcp_server.py` unit-tests the backend HTTP client via `httpx.MockTransport`.
- `test_mcp_protocol.py` launches the **real** MCP server as a subprocess and drives
  it with the MCP SDK over stdio against a stub backend (incl. a stub `/api/embed`) —
  the full client↔server↔backend chain over actual JSON-RPC, paper **and** ACE tools.
  (We don't ship a client of our own; the clients are Claude Code / Cursor / Codex.)
- `test_playbook.py` / `test_ace_tools.py` cover the playbook store and the ACE tools.

## Paper artifact transport

- `get_paper_markdown(..., save_path=None)` returns complete JSON-backed Markdown.
  With `save_path`, it streams the raw `/api/papers/{paper_id}/markdown` resource.
- Markdown and PDF saves use validated HTTP Range requests. Interrupted valid
  prefixes resume from the exact local byte offset, and the completed file is
  published atomically.
- `download_papers` accepts at most 10 paper IDs and reuses those per-artifact Range
  paths with at most two transfers in flight. Sanitized filename collisions receive
  stable digest suffixes; failures are reported per artifact.
- The separate authenticated HTTP `POST /api/papers/download-batch` endpoint is a
  one-shot ZIP convenience API. It accepts at most 10 papers, caps source artifacts
  at 1 GiB, allows two concurrent archive builds, and rejects Range because each
  POST creates a fresh archive.

On our own eval (`scripts/eval/code_embed_eval.py`, 2008 real repo chunks) BGE-M3 is
statistically **tied** with top code-specialized models (CodeRankEmbed, SFR-400M), so
the code KB reuses the shared BGE-M3 rather than paying for a second model.

## Roadmap

This server exposes the shared academic corpus, Journal, ACE playbook, Code KB, and
index inventory over the same authenticated REST surface. Production isolation is
owned by the gateway and backend identity layer; the MCP process contains no paper
database, Qdrant server, or embedding/reranker model.


当前发布 `retrieval-mcp==0.5.5`（2026-10-11）：`search_code` 新增 `show_line_numbers=True`，私有/公共/联合及批量查询默认逐行显示源文件行号；传 `False` 返回纯代码。前端默认独立行号栏，复制代码保留原始内容。该选项仅影响展示，服务 API、索引与 embedding 不变。
