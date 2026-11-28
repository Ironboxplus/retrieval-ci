# Remote BGE-M3 Embed Pipeline (kss 8× RTX 3090)

> Specialized bulk/recovery pipeline. Normal production indexing is server-side on the 4090; use this path only when a planned remote batch requires kss. Verify current GPU ownership and never transfer paper PDFs unnecessarily.

Track A: re-embed all SQLite chunks on remote 8-GPU box; ship vectors back; local upsert into Qdrant.
PDFs stay local. Only chunks (1.5GB gzip) go up; only vectors (~14GB) come back.

## Layout

```
scripts/remote_embed/
├── dump_chunks.py       # LOCAL: SQLite chunks → jsonl.gz shards
├── embed_worker.py      # KSS:   jsonl.gz → parquet (dense fp16 + sparse)
├── run_embed_kss.sh     # KSS:   split shards across 8 GPUs via tmux
├── local_ingest.py      # LOCAL: parquet → Qdrant upsert
└── ship.sh              # LOCAL: orchestrate rsync up/down + remote trigger
```

## Paths

| Side  | Path |
|-------|------|
| LOCAL | `data/remote_embed/chunks/chunks_NNNN.jsonl.gz` (in) |
| LOCAL | `data/remote_embed/vectors/vectors_NNNN.parquet` (out) |
| KSS   | `/mnt/lsb/retrival/chunks/` |
| KSS   | `/mnt/lsb/retrival/vectors/` |
| KSS   | `/mnt/nvme-disk/.cache/modelscope/BAAI/bge-m3` (model) |

## Flow

1. `python scripts/remote_embed/dump_chunks.py` — emits 67 jsonl.gz shards (~1.5GB).
2. `bash scripts/remote_embed/ship.sh up` — rsync shards → kss `/mnt/lsb/retrival/chunks/`.
3. `ssh kss "bash /home/kss/arc/retrival/scripts/remote_embed/run_embed_kss.sh"` — launch 8 tmux workers.
4. `bash scripts/remote_embed/ship.sh down` — rsync `vectors_*.parquet` back to local.
5. `python scripts/remote_embed/local_ingest.py` — Qdrant upsert.

## Resume

Every step is idempotent and shard-scoped: done.txt markers per shard, both at dump and upsert. Re-run safely.
