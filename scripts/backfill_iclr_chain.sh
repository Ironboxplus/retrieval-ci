#!/usr/bin/env bash
# Orchestrate ICLR backfill after marker MDs are ready on SMB.
#
# Pipeline (sequential, fail-fast):
#   1. sync_iclr_manifest_from_smb.py     manifest.markdown_path → SMB path
#   2. preindex_md_to_sql_only.py         MD → chunks → SQL (semantic chunker)
#   3. dump_chunks.py --min-chunk-id N    SQL → parquet shards
#   4. ship.sh up MIN_SHARD=N             local → kss SMB
#   5. (manual) run_embed_kss.sh on kss   bge-m3 GPU embed
#   6. ship.sh down                       kss SMB → local
#   7. local_ingest.py --min-shard N      parquet → Qdrant
#   8. verify via /api/search
#
# Run from project root. Use --dry-run to print commands without exec.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
PY="${PYTHON:-.venv/Scripts/python.exe}"

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY="echo + "
fi

echo "== ICLR backfill chain =="
echo "[$(date)] starting"

# 1. Sync manifest with SMB MD paths
echo
echo "--- Step 1/7: sync manifest.markdown_path from SMB ---"
${DRY} ${PY} scripts/sync_iclr_manifest_from_smb.py

# 2. Capture current max chunk_id for cursor-based dump
echo
echo "--- Step 2/7: capture cursor (SQL max chunk_id) ---"
CURSOR_BEFORE=$(${PY} -c "
import sqlite3
c = sqlite3.connect('data/db/retrieval.db')
print(c.execute('SELECT COALESCE(MAX(id),0) FROM chunks').fetchone()[0])
")
echo "cursor (pre-preindex max chunk_id) = ${CURSOR_BEFORE}"

# 3. Preindex (semantic chunker — uses bge-m3 for boundary detection)
echo
echo "--- Step 3/7: preindex (chunker_type=semantic, bge-m3 boundary detector) ---"
echo "NOTE: semantic chunker requires GPU access for bge-m3; preindex will fall back to CPU if no GPU"
${DRY} ${PY} scripts/preindex_md_to_sql_only.py \
  --chunker-type semantic \
  --commit-batch 20 \
  --checkpoint-interval 100

# 4. Dump new chunks (id > CURSOR_BEFORE) to parquet shards
CURSOR_AFTER=$(${PY} -c "
import sqlite3
c = sqlite3.connect('data/db/retrieval.db')
print(c.execute('SELECT COALESCE(MAX(id),0) FROM chunks').fetchone()[0])
")
NEW_CHUNKS=$((CURSOR_AFTER - CURSOR_BEFORE))
echo
echo "--- Step 4/7: dump ${NEW_CHUNKS} new chunks (id > ${CURSOR_BEFORE}) ---"
CURRENT_LAST_SHARD=$(${PY} -c "
from pathlib import Path
import re
shards = sorted(Path('data/remote_embed/chunks').glob('chunks_*.jsonl.gz'))
if not shards: print(0); exit()
m = re.match(r'chunks_(\d+).jsonl.gz', shards[-1].name)
print(int(m.group(1)) + 1 if m else 0)
")
echo "next shard index = ${CURRENT_LAST_SHARD}"
${DRY} ${PY} scripts/remote_embed/dump_chunks.py \
  --min-chunk-id $((CURSOR_BEFORE + 1)) \
  --start-shard ${CURRENT_LAST_SHARD}

# 5. Ship shards to kss
echo
echo "--- Step 5/7: ship new shards up to kss SMB ---"
${DRY} env MIN_SHARD=${CURRENT_LAST_SHARD} bash scripts/remote_embed/ship.sh up

# 6. Trigger kss embed (manual prompt — user runs this on kss)
echo
echo "--- Step 6/7: RUN ON KSS ---"
echo "  ssh kss"
echo "  cd /home/kss/arc/retrival"
echo "  MIN_SHARD=${CURRENT_LAST_SHARD} WORLD=8 bash scripts/remote_embed/run_embed_kss.sh"
echo "  # then wait for tmux embed_g0..g7 to finish"
echo "Press ENTER when kss embed is done, or Ctrl-C to abort..."
read -r _

# 7. Ship vectors back + ingest to Qdrant
echo
echo "--- Step 7/7: ship vectors back + ingest to Qdrant ---"
${DRY} env MIN_SHARD=${CURRENT_LAST_SHARD} bash scripts/remote_embed/ship.sh down
${DRY} ${PY} scripts/remote_embed/local_ingest.py --min-shard ${CURRENT_LAST_SHARD}

# 8. Verify
echo
echo "--- Verify ICLR is searchable ---"
${DRY} curl -s 'http://localhost:8000/api/search?q=in-context+learning&venue=ICLR&mode=hybrid&top_k=5' | head -200

echo
echo "[$(date)] done"
