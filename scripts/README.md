# scripts/ — Operational scripts

> Current platform and safety rules: [../overview.md](../overview.md) and [../AGENTS.md](../AGENTS.md). Run `python scripts/retrival.py` to discover the live command registry; this README explains intent but the CLI is authoritative when they differ.

All non-test scripts go through a single CLI entry: [`retrival.py`](retrival.py).

```bash
python scripts/retrival.py                     # list groups
python scripts/retrival.py <group>             # list commands in group
python scripts/retrival.py <group> <cmd> ...   # passes remaining args through
```

Args after the command are forwarded verbatim to the underlying script.
Run `... <group> <cmd> --help` to see the underlying script's own help.

---

## Groups & commands

### `pipeline` — Paper collection / extraction / preindex (MD → SQL)

| Command | Script | Purpose |
|---|---|---|
| `preindex-md`      | [preindex_md_to_sql_only.py](preindex_md_to_sql_only.py) | **Stage F Step 1**: MD → SQL (CPU-only, bypass embed); WAL + 19× speedup |
| `preindex-corpus`  | [preindex_existing_extracted_corpus_papers.py](preindex_existing_extracted_corpus_papers.py) | Full preindex: MD → SQL + GPU embed → Qdrant |
| `rebuild-corpus`   | [rebuild_ccf_a_main_corpus.py](rebuild_ccf_a_main_corpus.py) | CCF-A corpus rebuild driver |
| `run-2025`         | [run_2025_pipeline.py](run_2025_pipeline.py) | 2025 batch: collect → download → extract → preindex |
| `download-subset`  | [download_subset.py](download_subset.py) | Per-conference subset download；可选 first-hop + SOCKS exit allowlist、真实 PDF 健康探测、不同出口重试、独立 `--pdf-root`、批次共享 `--max-bytes-per-second`（0=不限速）、manifest-safe（W17） |
| `download-arxiv`   | [download_via_arxiv.py](download_via_arxiv.py) | arxiv-based PDF download for missing papers |
| `download-missing` | [download_missing_pdfs.py](download_missing_pdfs.py) | Manifest-driven gap fill |
| `sync-manifest`    | [sync_manifest_pdf_md_paths.py](sync/sync_manifest_pdf_md_paths.py) | FS → manifest one-way sync (+ .bak) |

### `embed` — Remote BGE-M3 embed pipeline

Flow: `SQL chunks → jsonl.gz (dump) → kss 8× RTX 3090 (worker) → parquet → local Qdrant (ingest)`.

| Command | Script | Purpose |
|---|---|---|
| `dump`    | [remote_embed/dump_chunks.py](remote_embed/dump_chunks.py) | Dump SQL chunks → `chunks_NNNN.jsonl.gz` (100K rows/shard). Supports `--min-chunk-id` for incremental dump (Stage F) |
| `worker`  | [remote_embed/embed_worker.py](remote_embed/embed_worker.py) | Single-GPU BGE-M3 worker (run on kss). Reads rank/world via env |
| `ingest`  | [remote_embed/local_ingest.py](remote_embed/local_ingest.py) | Parquet → Qdrant upsert with W14 retry (wait=True + 1/2/4/8/16s backoff) |
| `ship`    | [remote_embed/ship.sh](remote_embed/ship.sh) | `code` \| `up` \| `down` \| `status` — code sync / chunk upload (scp, SMB-friendly) / vector download / kss state |
| `run-kss` | [remote_embed/run_embed_kss.sh](remote_embed/run_embed_kss.sh) | Spawn 8 tmux sessions (`embed_g0..g7`), one worker per GPU |

### `qdrant` — Qdrant ops

| Command | Script | Purpose |
|---|---|---|
| `health`            | [tools/qdrant_ops.py](../tools/qdrant_ops.py) `health`            | All-in-one probe: cluster / collections / meta / search |
| `segments`          | [tools/qdrant_ops.py](../tools/qdrant_ops.py) `segments`          | Per-segment payload index / vec-dense / vec-sparse sizes |
| `progress`          | [tools/qdrant_ops.py](../tools/qdrant_ops.py) `progress`          | Sample indexing rate → ETA |
| `search`            | [tools/qdrant_ops.py](../tools/qdrant_ops.py) `search`            | Single-vector latency probe |
| `start` / `stop`    | [tools/qdrant_ops.py](../tools/qdrant_ops.py) `start` / `stop`    | Qdrant server lifecycle (with `NO_PROXY` injection) |
| `recreate`          | [recreate_qdrant_collection.py](recreate_qdrant_collection.py)    | **One-shot** atomic collection (vectors + hnsw + quantization + 8 payload indexes) — avoids W5/W10 PATCH traps |
| `cleanup-orphans`   | [cleanup_qdrant_orphans.py](cleanup_qdrant_orphans.py)            | Drop Qdrant points whose `doc_id` not in SQL (`--execute` to commit) |
| `migrate-payload`   | [migrate_qdrant_payload.py](migrate_qdrant_payload.py)            | Old → new payload schema (`conference→venue`, `author_names`→`authors`) per-doc Filter |

### `db` — SQLite migrations

| Command | Script | Purpose |
|---|---|---|
| `migrate-papers-pk` | [migrate_papers_pk.py](migrate_papers_pk.py) | **Applied 2026-05-17** (Stage E): drop `papers.id`, lift `paper_id` to PRIMARY KEY NOT NULL, `doc_id` UNIQUE FK. dry-run / verify-only / skip-backup flags. Kept for rollback reference |

### `verify` — API & data source verification

| Command | Script | Purpose |
|---|---|---|
| `live-apis`     | [live_verify_all_apis.py](live_verify_all_apis.py)         | Full live API contract verification |
| `api-contracts` | [verify_live_api_contracts.py](verify_live_api_contracts.py) | JSON report mode |
| `2025-sources`  | [verify_2025_sources.py](verify_2025_sources.py)            | Conference source configuration sanity (23 cases incl. ICLR OpenReview) |
| `journal-gateway-smoke` | [smoke/smoke_journal_gateway.py](smoke/smoke_journal_gateway.py) | Real uvicorn compute+gateway journal smoke: sk-key auth, proxy, owner isolation, MCP; temp SQLite + in-memory vector store |
| `real-pdf-download-smoke` | [smoke/smoke_download_real_pdf.py](smoke/smoke_download_real_pdf.py) | 使用既有真实 PDF 启动真实 origin + forwarding proxy，验证并发聚合限速、字节一致性和临时产物清理；禁止生成占位 PDF |
| `journal-qdrant-smoke` | [smoke/smoke_journal_qdrant.py](smoke/smoke_journal_qdrant.py) | Temporary Qdrant server + `journal_smoke` collection; verifies JournalVectorStore upsert/search/delete without touching production `papers` |

### `test` — CLI test suites

These are *standalone* CLI runners, not pytest. For pytest see [tests/](../tests/).

| Command | Script | Coverage |
|---|---|---|
| `api`      | [test_api.py](test_api.py)           | 106 API tests |
| `e2e`      | [test_e2e.py](test_e2e.py)           | 24 end-to-end |
| `services` | [test_services.py](test_services.py) | 51 service-layer |
| `stress`   | [test_stress.py](test_stress.py)     | Load/stress |

### `ops` — Maintenance & monitoring

| Command | Script | Purpose |
|---|---|---|
| `cleanup-zombies` | [cleanup_zombie_tasks.py](cleanup_zombie_tasks.py)                  | Reap zombie task rows |
| `monitor-corpus`  | [monitor_ccf_a_main_corpus_pipeline.py](monitor_ccf_a_main_corpus_pipeline.py) | Corpus pipeline monitor |
| `watch-progress`  | [watch_progress.sh](watch_progress.sh)                              | 60s JSON progress sampler |

### `gen-public-openapi` — 公开 OpenAPI spec 生成

直接运行（未注册进 `retrival.py` dispatcher，单次执行）：

```bash
python scripts/gen_public_openapi.py
```

| 脚本 | 用途 |
|---|---|
| [gen_public_openapi.py](gen_public_openapi.py) | 从 `docs/openapi.json` 生成脱敏公开 spec → `frontend/public/openapi.json`，供 BWH `/docs/` Swagger UI 消费；剥除内部 admin 端点，各端点描述标「公网需登录·浏览不计费」或「公网需登录·扣费」（W127b） |

---

## Quick examples

```bash
# Stage F Step 2: incremental dump only new chunks since Stage C cutoff
python scripts/retrival.py embed dump --min-chunk-id 7791515 --start-shard 67

# Stage F Step 3a: upload to kss (uses scp because /mnt/lsb is SMB)
python scripts/retrival.py embed ship up

# Stage F Step 3b: spawn 8 GPU workers on kss
python scripts/retrival.py embed run-kss

# Stage F Step 5: ingest parquet back to Qdrant (with W14 retry)
python scripts/retrival.py embed ingest --max-points 100000   # smoke
python scripts/retrival.py embed ingest                       # full

# Qdrant diagnostics
python scripts/retrival.py qdrant health
python scripts/retrival.py qdrant segments
python scripts/retrival.py qdrant progress --samples 3 --interval 60

# Verify everything still works after a change
python scripts/retrival.py verify live-apis
python scripts/retrival.py test api
```

---

## [archive/](archive/) — Retired scripts (kept for reference, not deleted)

Moved out of the main directory to reduce noise. Subdirectories:

- **`chroma_whoosh/`** — Chroma / Whoosh era helpers (those vector stores are retired; replaced by Qdrant + SQLite FTS5)
- **`old_index/`** — Old index builders predating the BGE-M3 + Qdrant pipeline (`build_*_index.py`, `simple_index_builder.py`, `reset_index.py`, `warm_*`, ...)
- **`old_crawl/`** — Pre-`download_subset` crawl scripts (`direct_crawl.py`, `safe_crawl.py`, `download_pdfs.py`, `resume_download.py`, `run_crawl_task.py`)
- **`migrations_oneshot/`** — One-shot SQLite schema migrations that have already been applied to the production DB (`migrate_add_titles.py`, `migrate_add_source_type.py`, `migrate_add_heartbeat_columns.py`, `migrate_add_paper_id_column.py`, `d4_prune_payload.py`)
- **`old_powershell/`** — Pre-Python-pipeline PowerShell wrappers for the CCF-A corpus and old enterprise tests
- **`misc/`** — Citation / monitor / google-scholar / batch_convert_pdf and other one-off helpers no longer in active flows

> **Why kept**: rollback reference, historical context for `journal.md` cross-references. Safe to delete entirely once journal references are updated.

---

## Adding a new script

1. Drop the script under `scripts/` (or `scripts/remote_embed/` if it's an embed-pipeline component).
2. Register it in `COMMANDS` inside [`retrival.py`](retrival.py) with a kebab-case command name.
3. Add a row in the corresponding section above.
4. If the script is `.sh`, set `runner="sh"`; if `.py`, use `"py"` (the CLI picks `.venv/Scripts/python.exe` automatically when present).

Keep arg parsing inside the underlying script — `retrival.py` is a dispatcher, not a wrapper.
