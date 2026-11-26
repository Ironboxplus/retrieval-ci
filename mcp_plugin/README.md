# retrieval-mcp

Current release: **0.5.15** (2026-11-21), 35 tools. Platform architecture:
[../overview.md](../overview.md).

`0.5.15` (2026-11-21): every gitignore check fails closed. When git cannot answer (timeout, corrupt index, any error other than "not a git repository"), the watcher no longer treats every touched path as allowed: the batch becomes a reconcile that reports the failure and is retried after the interval, and an explicit `index_code` reports the error. A directory event lists its files with `git ls-files` instead of sending every file under it to one `git check-ignore`, and `check-ignore` runs in batches of 2,000 paths. A line window that is all blank no longer becomes an empty chunk (the backend rejected the whole batch with HTTP 422). A burst of 128,000 ignored files in one project had grown the daemon to 77 GB.

`0.5.14` (2026-11-15): automatic indexing (the filesystem watcher and auto-refresh) runs at most once per 5 minutes per scope; changes inside the interval go out as one batch, and explicit `index_code` calls are never throttled. The backend raises the floor of the policy's `min_check_interval_s` to 300 s.

`0.5.13` (2026-11-13): `index_code` also indexes documentation files (`.md` `.mdx` `.markdown` `.rst` `.adoc` `.txt`) in the same walk. They go through the Journal document pipeline and are searched with `journal_search`; auto-refresh keeps them fresh too. `journal_index_dir` is deprecated and will be removed in the next release. No backend change.

`0.5.12` (2026-11-09): after a code index changes files, the server checks the changed
files for duplicated code in the background; the next `search_code`, `journal_search` or
`index_code_status(job_id)` of that scope shows the summary once at the top (the server
decides who gets it; the client stores nothing). Needs the matching backend.

`0.5.11` (2026-11-09): the initialize instructions became a rule: the first search in a
task must be a Retrieval MCP call (`search_code` keyword mode replaces grep), shell search
tools come after it, with three named exceptions. Tools and schemas are unchanged.

`0.5.10` (2026-11-04): fewer whole-scope reads. Hash baselines use the lean inventory
(`lean=true`), an unchanged `journal_index_dir` returns before reporting, incremental
`index_code` sends its removals, and `index_inventory(project=)` filters on the server.

`0.5.9` (2026-11-04): opt-in sync of Claude Code / Codex conversations into Journal
(`RETRIEVAL_JOURNAL_CONVERSATIONS`); deja and the zstd decoder are provisioned by the
plugin itself. See "Claude Code / Codex conversations in Journal" below.

`0.5.7` (2026-10-13): the incremental-index file identity is the plain SHA-256 of
the file again. 0.5.6 had folded a hard-coded chunker generation into it, which
made every incremental pass re-embed whole projects. To replace chunks cut by an
older client, run `index_code` once (a full pass, the default); the backend keeps
the stored vectors of every chunk whose id and text are unchanged, so only chunks
that actually moved are embedded.

`0.5.6` (2026-10-13): code chunks are cut where they really are. chonkie's
CodeChunker reports UTF-8 byte offsets, and slicing the text with them shifted every
chunk after the first non-ASCII character (wrong text, wrong `path:line`); sources
over 384K characters were a single chunk. Large sources are now cut at blank lines
and chunked normally. Tool surface unchanged.

`0.5.2` adds `import_public_code_repo` (admin tier): it adds a repository you already
cloned to the curated Public Code corpus by uploading it as a `git bundle`, for
repositories the server cannot clone itself. Only a clean HEAD that is on an origin
branch is accepted; a `--depth 1` clone or a bare mirror is enough. The server builds
the mirror and chunks, embeds and indexes it; nothing is embedded locally. Poll the
returned handle with `public_code_job_status`.

`0.5.3`: importing a clone automatically passes
its absolute path as `private_project`. The backend reuses the importing admin's
private chunk vectors when their canonical payload exactly matches the published
Git revision. Missing, changed or incomplete dense/sparse vectors are embedded
normally. Job results expose `reused_chunk_count`; private data is retained and
private owner/project/host fields are never copied to public payloads. The MCP tool
names, arguments and advertised descriptions are unchanged.

`0.5.4`: `index_inventory` prints one line per project and at most `limit` (default
40) of them: scopes whose code snapshot needs attention first, then the most recently
indexed, with "Listing k of n" when more exist. `project=` filters by path fragment.
A root directory equal to the project key and a plain "completed" snapshot note are
no longer printed; 305 scopes had produced a 56,159-character result.

`0.5.1` fixes error reporting only (tool surface unchanged): a rejected backend request
now carries the backend's reason (`HTTP 422 from POST /api/public-code/search: Unknown
public-code source IDs: x`) instead of only the status line and an MDN link, and an
unknown `source_ids`/`domains` value in `search_code` comes back with the valid names.

`0.5.0` is a breaking change: `search_all_code` and `search_public_code` are removed
and folded into `search_code`. Where to search is now two arguments of one tool:
`repo_root` (one directory, a list, or `"*"` for every scope you own) and `sources`
(`"private"` default | `"public"` | `"all"`, the latter two admin tier), plus
`source_ids` / `domains` for the public side. Migration: `search_all_code(query,
repo_roots=R, include_public=True)` → `search_code(query, repo_root=R or "*",
sources="all")`; `search_public_code(query, ...)` → `search_code(query,
sources="public", ...)`. Behaviour and output of each route are unchanged; every
route now also takes a query list. Arguments that only fit the other side
(`source_ids` with `sources="private"`, `path_glob` with `sources="public"`) are an
explicit error. The `search_code` and `journal_search` descriptions and the server
instructions now state what they do better than rg/grep: keyword mode is the grep
replacement for exact lookups (ranked, returned with the surrounding function),
semantic modes find code/notes by meaning. `tools/list`: 37,417 → about 36K characters.

`0.4.19` keeps every tool name, signature and behaviour and trims `tools/list` from
47,883 to 37,417 characters (−30% since 0.4.17's 53,431). Tools no longer publish
FastMCP's automatic outputSchema (`{"result": string}` on all 36), so a call result is
sent once as text instead of also as `{"result": <same text>}`. Sibling tools now
describe only how they differ from the tool they extend: `search_all_code` from
`search_code`, `journal_index_dir` / `delete_journal_docs` from `index_code` /
`delete_code`, `search_arxiv` from `search_papers`, `get_paper_markdown` from
`search_within_paper`. Args prose no longer repeats `(required)` or a `(default X)`
the schema already carries. 21 description digests move; no inputSchema changes
(outputSchema is not part of the pinned surface).

`0.4.18` changes no tool name, signature or behaviour; it shrinks what every client
session loads from `tools/list` by 5,548 characters (53,431 → 47,883): pydantic's
generated schema `title`s are dropped (they only repeat parameter names), the
list-query contract moves from six appended description paragraphs to the `query`
parameter's own description plus one sentence in the server instructions, and
fifteen one-line examples that only restated a signature are removed. All 36
field digests move (every schema lost its titles).

`0.4.17` adds `search_arxiv`: semantic hybrid search over the arXiv CS preprints,
which `search_papers` now leaves out (the backend excludes `venue_tier=preprint`
unless a request is scoped to it). The tool sends the ordinary `/api/search` with
`venue_tier="preprint"`, so it needs no new gateway route; filters are `year` and
`top_k`, and like `search_papers` it accepts one query or an ordered list.
Results carry `arxiv-<arXiv id>` paper_ids that `search_within_paper` and
`get_paper_markdown` accept. Tool surface: 35 → 36 tools, the other 35 field
digests unchanged.

`0.4.16` keeps the 0.4.15 tool surface (same names, signatures and digests) and
fixes one client behaviour: a code index job can now be cancelled on the server,
and `index_code` (including its auto-refresh watcher) treats `cancelled` as a finished state
instead of polling until the wait timeout.

`0.4.15` adds `search_all_code`: one call that fuses your private Code KB and the
curated Public Code corpus (Reciprocal Rank Fusion by default, or cross-encoder
reranked with `rerank=true`), tagging each hit `[private]` or
`[public:<source_id>@<rev>]`. Public results still require admin tier — a
non-admin caller gets an explicit error, never a silent private-only downgrade.
`search_code` / `search_public_code` are unchanged and remain the right tool when
you only need one plane. v1 accepts a single query string only (no batch list).
Tool surface: 34 → 35 tools, same digest for the other 34
(`.handfree-output/regen_digests.py --expect search_all_code`).

`0.4.14` makes uploaded documents reachable past search. `papers` gives
upload/webpage payloads no `paper_id` key at all, so `search_papers` printed an
empty `paper_id:` line for a book and `search_within_paper` /
`get_paper_markdown` had nothing to pass — the book was findable but not
readable. Those results now advertise `doc:<documents.id>`, which routes to the
backend's `doc_ids` filter and to `/api/documents/{id}/fulltext|markdown`. A bare
all-digit id is accepted as the same thing: 0 of 106,156 production paper_ids are
all-digits, so the namespaces cannot collide. Because an uploaded book runs to
millions of characters, the inline (no `save_path`) form refuses above 400,000
bytes and says to use `save_path`; papers keep their uncapped behaviour. Tool
names and signatures are unchanged.

`0.4.12` is the cross-Python hotfix for 0.4.11. It canonicalizes each legacy tool
description before appending the multi-query suffix, so Python 3.10--3.13 advertise
the same tools/list surface and singleton daemon digest. `0.4.11` must not be used:
its Python 3.12 `uvx` path fails closed during daemon startup because description
indentation changes the surface digest.
`0.4.11` adds Retrieval-first initialize guidance and permits one MCP call to carry
multiple independent query strings for `search_code`, `journal_search`,
`search_public_code`, `search_papers`, and `search_within_paper`. A string remains
fully compatible; a list preserves input order (including duplicates), runs each
query with the existing filters/depth/rerank/formatter, and labels each result. The
batch has bounded in-flight concurrency only; it never shares or trims candidates.
`0.4.10` fixes a cross-project scope bug. The daemon is an account-level singleton,
but a relative scope (`scope_dir="."`, `repo_root="."`, `save_dir="."`) was resolved
against the DAEMON's own working directory — whichever project's shim started it. So
journal reads returned a foreign repo, `journal_record` wrote into that repo's
namespace, and downloads landed in the wrong project. The shim now sends its directory
in the authenticated raw handshake and the daemon binds it per session with a context
variable, so concurrent sessions from different projects stay independent.
`DAEMON_PROTOCOL_SCHEMA_VERSION` goes 1 → 2 so a fixed shim cannot reuse an unfixed
daemon. Tool surface is unchanged (33 tools, same digest).

Two changes in 0.4.9 over 0.4.8:

- `index_code` and `journal_index_dir` document the nested-scope guard's
  gitignore exception — a child scope indexing with `respect_gitignore=False`
  under a filtered parent now warns instead of being refused.
- Request bodies over 64 KiB are compressed with zstd level 3, which is what
  `index_code`/`index_dir` uploads are (a real 19.3 MiB body becomes 2.2 MiB in
  54 ms). This adds a `zstandard` dependency; it also makes httpx advertise zstd,
  so replies come back compressed with no further work.

  **It does not require a backend upgrade.** Compression is negotiated per
  request: the client learns a backend accepts compressed bodies from the
  `Accept-Encoding` *response* header (RFC 7694), and a backend that does not
  rejects the body with 400/415/422 — all decided before the route function runs,
  so the automatic uncompressed resend cannot duplicate an index submission.
  After one such rejection the client stops compressing for the session.

`find_duplicate_code` was added in 0.4.8 and its backend endpoint is live on `.88`.

The package declares `mcp>=1.29,<1.30`. In addition to the public SDK v1
`FastMCP` API, the raw JSONL daemon bridge uses the verified 1.29 low-level
`SessionMessage` and server-stream APIs. The narrow upper bound is therefore a
private-API compatibility contract, not merely a v2 exclusion; raising it needs
the daemon concurrency and clean-wheel MCP regressions first.

The daemon's 33-tool compatibility digest covers each tool name, full
`inputSchema`, and semantic description text. Description canonicalization uses
`inspect.cleandoc` only to remove the shared docstring margin emitted differently
by the supported Python 3.10/Pydantic and Windows renderers; it does not reduce
the fence to tool names or discard meaningful relative indentation.

Code indexing deliberately declares `chonkie[code]>=1.6,<1.7`,
`tree-sitter-language-pack>=1.8,<1.13`, and `tree-sitter>=0.23,<0.26`. The local
`CodeChunker` path is verified on Chonkie 1.6.x, the 1.12.x language-pack line,
and the native tree-sitter 0.25.x ABI. Chonkie's `code` extra only gives its
parser dependencies lower bounds; the explicit companion constraints prevent a
fresh install from silently crossing into a newer parser-pack or native ABI line.
The native constraint is intentional because local indexing executes native
parser code and its crash boundary must not drift. Raising any upper bound
requires a CodeChunker regression and a clean wheel-install MCP smoke; this is a
compatibility contract, not a package-cache workaround.

An **MCP server** for the Retrieval academic-paper API - semantic paper search,
document matching, ACE journal memory, and index inventory. Self-contained: it talks
to the backend over HTTP only (just `mcp` + `httpx`), so it installs anywhere with
`uvx` / `pip` - no repo checkout, no GPU, no models.

By default it targets the compute box on the lab LAN (`http://10.100.100.111:8000`),
which trusts LAN callers so **no key is needed**. Off-LAN, point `RETRIEVAL_API_URL` at
the public gateway (`https://retrieval.rnarket.com`) and set `RETRIEVAL_API_KEY` (`sk-...`).

## Tools

Every tool's full docstring (purpose + each argument with its default + an example)
is what your LLM sees - call them by name. Summary:

**Paper retrieval**

| Tool | What it does |
|------|--------------|
| `search_papers` | Semantic hybrid search over the live top-venue corpus (filters: `venue`, `year`, `title_only`) |
| `search_arxiv` | Semantic hybrid search over arXiv CS preprints, kept out of `search_papers` (filters: `year`) |
| `search_within_paper` | Every matching passage inside one paper |
| `get_paper_markdown` | Return one paper's complete Markdown, or Range-stream it to a caller-local file/directory |
| `download_paper_pdf` | Range-stream the original PDF to a caller-local path with validated resume and atomic publication |
| `download_papers` | Download `pdf`, `markdown`, or `both` for 1-10 paper IDs with two bounded concurrent transfers |
| `match_document` | Content-nearest papers to a passage |
| `match_paper` / `match_paper_status` | Submit and monitor/cancel a durable exact paper-neighbour job |
| `list_conferences` / `corpus_stats` | Venue registry / corpus size |

**Journal work-memory** (scoped to the current project by default)

| Tool | What it does |
|------|--------------|
| `journal_record` | Record a work note (`memory`) or a file's current content (`doc`, latest-wins) |
| `journal_search` | Search memory - `keyword` (FTS5, no embedding) or `hybrid`/`dense`/`sparse` |
| `journal_recent` | List recent entries |
| `journal_index_dir` | Batch-index a local dir's files into the journal (latest-wins per file) |
| `journal_index_status` | Poll an asynchronous Journal directory-index job |
| `delete_journal_docs` | Explicitly remove batch-indexed Journal documents for one directory scope |

**Code KB** (source stays local - only chunks are uploaded)

| Tool | What it does |
|------|--------------|
| `index_code` | AST-chunk a repo locally (40+ languages) and index it, scoped to you |
| `index_code_status` | Poll local preparation plus backend Code KB embedding/upsert progress |
| `search_code` | Keyword (grep replacement) or semantic code search with `path:line` citations over your scopes, the Public Code corpus (`sources="public"`), or both fused (`sources="all"`) |
| `find_duplicate_code` | Report file pairs in an indexed scope that duplicate each other |
| `delete_code` | Explicitly remove one Code KB scope from vectors and inventory |
| `index_inventory` | Your indexed-file tree: user -> host -> project -> dir -> file |

**Public Code** (admin-managed shared source corpus)

| Tool | What it does |
|------|--------------|
| `list_public_code_sources` | List configured sources, active revisions, coverage, profiles, and update state |
| `add_public_code_source` | Register an HTTPS Git upstream using server-side proxy/credential profile references |
| `check_public_code_updates` | Queue remote revision checks for selected sources or domains |
| `index_public_code_sources` | Queue bare mirror refresh, AST chunking, shared GPU embedding, and atomic promotion |
| `set_public_code_update_policy` | Change manual/scheduled update policy without exposing credentials |
| `public_code_job_status` | Poll durable source/file/chunk progress for checks and indexing |

Search the Public Code corpus with `search_code(sources="public", source_ids=..., domains=...)`.

**ACE playbook** (accumulated, curated lessons per project)

| Tool | What it does |
|------|--------------|
| `ace_context_aware` / `ace_playbook` | Retrieve relevant / list all curated bullets |
| `ace_enhance_prompt` / `ace_smart_generate` | Attach playbook lessons to a prompt (no LLM call) |
| `ace_smart_reflect` | Curate a transferable lesson into the playbook (grow-and-refine dedup) |

### Code KB language coverage

`index_code` chunks 40+ languages structurally via tree-sitter (chonkie CodeChunker): Python,
TypeScript/TSX/JS/JSX (React), Java, Kotlin (incl. Jetpack Compose `.kt`/`.kts`),
Swift, Go, Rust, C/C++, C#, Ruby, PHP, Lua, Scala, Dart, R, Julia, Elixir, Erlang,
Haskell, OCaml, SQL, GraphQL, Protobuf, HTML, CSS/SCSS (Tailwind = CSS classes),
Vue, Svelte, shell, PowerShell, Dockerfile, Terraform/HCL, CMake, YAML/JSON/TOML/XML,
and more. Grammarless config/text files fall back to line-window chunks; docs (`.md`)
and binaries are skipped (docs belong in the journal via `journal_index_dir`).

## Singleton behavior and command-mode MCP configuration

Use the same command-style stdio entry for Claude, Codex, Grok, and other MCP
clients. The normal no-argument invocation automatically starts or reuses one
per-user daemon and transparently bridges just this client's JSONL session. No
client URL, transport option, or manual `start` step is required.

Install the published package once and use the same command-style configuration
for every client. Existing version-pinned `uvx` configurations should pin
`retrieval-mcp==0.4.12`.

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

Existing `uvx` command configurations remain compatible. A once-installed
`retrieval-mcp` avoids a package-runner wrapper per client and is the recommended
singleton setup. `status` and `stop` are explicit local control commands;
`retrieval-mcp stdio-server` is only a direct debug/compatibility server and is
not the normal client configuration.

## Automatic code-index refresh

`index_code(...)` returns a local job ID before repository walking, hashing, AST chunking,
upload, GPU embedding, or Qdrant upsert completes. Poll that same ID with
`index_code_status()` through the `preparing`, backend queue, and terminal phases. A second
index request for the same scope reuses the active job instead of starting another scan.

With `auto_refresh=True`, the client then keeps one filesystem watcher for that absolute
repository path. Ordinary file events hash and chunk only the touched paths; ignore-rule
changes trigger a full reconcile. `search_code()` never waits for the watcher or indexing:
it returns the last completed snapshot and reports freshness separately.

Git repositories continue to honor Git's ignore rules by default. Any directory, including
non-Git projects, can add scope-relative patterns to `.retrievalignore`; callers can add
temporary patterns with `exclude_globs=["generated/**", "private.py"]`. Explicit
`delete_code()` cancels/fences stale refresh work and removes vectors, inventory, and the
saved refresh policy.

Windows drive and UNC scopes use `pathlib`'s Windows path semantics and one
case-insensitive canonical identity, while POSIX scope casing remains significant.
Historical mixed-case Windows scopes are included through bounded catalog aliases.
`path_glob` is always scope-relative; both slash styles are accepted, and `pathspec`
supplies basename, directory, `**`, and exact-file matching semantics.
Windows extended-length/device spellings (`\\?\`, `\\.\`) are rejected with guidance
to use the equivalent ordinary drive or UNC path, preventing duplicate scope identities.

## Claude Code / Codex conversations in Journal (opt-in)

Set `RETRIEVAL_JOURNAL_CONVERSATIONS=1` and `journal_search` / `journal_index_dir` start a
background sync of that scope's Claude Code and Codex conversations into Journal. Neither
tool waits for it; the output gets one extra line, e.g.
`Conversations: last sync 2 min ago — 3 segment(s) from 1 session(s) uploaded`.
Search them with `journal_search(path_glob=".conversations/")`; hits show as
`[doc/claude]` / `[doc/codex]`.

Parsing, project attribution and secret redaction are done by the
[deja-vu](https://github.com/vshulcz/deja-vu) CLI (MIT). Nothing needs installing: on the
first sync the plugin downloads the pinned deja release for your platform (Windows,
Linux, macOS; x64/arm64) from GitHub, refuses it unless its sha256 matches the hash
pinned in the plugin (copied from the release's `checksums.txt`), and keeps it in the
plugin state directory. Codex's `.jsonl.zst` rollouts are decompressed through a small
`zstd` shim backed by the plugin's own `zstandard` dependency (a real `zstd` on `PATH`
still takes precedence). A `deja` already on `PATH` is not used, so the JSON contract is
always the pinned version's. The plugin only reads deja's `--json` output and keeps its
own index under the plugin state directory, so your own deja index is untouched. The
first run downloads deja and builds that index (about two to three minutes for a large
history) while searches continue normally.

| Var | Default | Notes |
|-----|---------|-------|
| `RETRIEVAL_JOURNAL_CONVERSATIONS` | unset (off) | `1` = main sessions only; `agents` = also spawned sub-agent sessions; unset or `0` = off. |
| `RETRIEVAL_DEJA_BIN` | pinned download | Full path to a deja binary to use instead (offline machines). |

What is stored: `user` and `assistant` text in full, each shell command as one
`$ command` line (at most 200 characters), and `edit` / `wrote` as the file path only.
File lists, tool output and compaction summaries are dropped. A session becomes docs
titled `.conversations/<claude|codex>/<YYYY-MM-DD>_<session-id>/<NNN>.md`, cut into
segments of roughly 8,000 characters at a user turn; a finished segment never changes,
so a growing session re-uploads only its last segment. Secrets deja redacts are masked
before upload, plus `pypi-` tokens, which deja 0.23.5 misses.

Cost: each scope keeps a small state file in the plugin state directory. A sync with
nothing new sends no HTTP request at all, runs at most once per 60 seconds per scope,
and never calls inventory or stats endpoints. Deleting a transcript locally does not
remove its already uploaded segments, and a recursive `journal_index_dir` leaves
`.conversations/` alone.

## Config (env)

| Var | Default | Notes |
|-----|---------|-------|
| `RETRIEVAL_API_URL` | `http://10.100.100.111:8000` | LAN compute box (no key). Off-LAN, set to `https://retrieval.rnarket.com`. |
| `RETRIEVAL_API_KEY` | - | `sk-...` key for the gateway (create under `/auth/keys`). Required off-LAN. |
| Journal/code scope | current absolute directory path | The client derives scope from the directory you run or pass to each tool, so projects do not leak into each other. |

## Paper artifact transport

- `get_paper_markdown(..., save_path=None)` returns the complete JSON-backed Markdown
  to the MCP caller. With `save_path`, it instead uses the raw
  `GET /api/papers/{paper_id}/markdown` file route, so large Markdown is not buffered
  as JSON by the gateway or MCP process.
- Markdown files and PDFs use HTTP Range with validated `Content-Range`. If the
  connection closes after writing a valid prefix, the next request resumes from the
  exact local byte offset. Completed files are published atomically and are not
  overwritten unless `overwrite=True`.
- `download_papers` accepts at most 10 paper IDs, preflights all destination names,
  disambiguates sanitized filename collisions with a stable digest, and reuses the
  single-artifact Range paths with at most two concurrent transfers. Errors are
  reported per paper/artifact.
- The separate authenticated HTTP `POST /api/papers/download-batch` endpoint serves a
  one-shot ZIP for browser/API clients. It is capped at 10 papers and 1 GiB of source
  artifacts, allows at most two concurrent archive builds, and intentionally rejects
  Range because each generated archive is a new file.

## License

MIT


当前发布 `retrieval-mcp==0.5.5`（2026-10-11）：`search_code` 新增 `show_line_numbers=True`，私有/公共/联合及批量查询默认逐行显示源文件行号；传 `False` 返回纯代码。前端默认独立行号栏，复制代码保留原始内容。该选项仅影响展示，服务 API、索引与 embedding 不变。
