#!/usr/bin/env bash
# Incremental ICLR backfill loop. Runs WHILE marker is still producing
# MDs on kss SMB. Each iteration:
#   1. sync manifest: pull any new MDs found on SMB into manifest.markdown_path
#   2. preindex: chunk only papers whose paper_id isn't in SQL yet
#   3. sleep then repeat
#
# Exits when marker is done (detected by per-rank done.txt totals summing
# to >= 8599) OR when --max-iter reached.
#
# Usage: bash scripts/incremental_backfill_loop.sh [interval=600] [max_iter=100]
set -euo pipefail

INTERVAL="${1:-600}"     # seconds between iterations (default 10 min)
MAX_ITER="${2:-100}"     # safety cap
TARGET=8599              # ICLR total PDFs

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
PY=".venv/Scripts/python.exe"

iter=0
while (( iter < MAX_ITER )); do
  iter=$((iter + 1))
  echo "=============================="
  echo "[$(date)] iter ${iter}/${MAX_ITER}"
  echo "=============================="

  # 1. Sync manifest with whatever marker has produced
  echo "--- sync manifest ---"
  ${PY} scripts/sync_iclr_manifest_from_smb.py 2>&1 | tail -3

  # 2. Run preindex (skips already-in-SQL papers)
  echo "--- preindex (canonical L1+L2 paper schema) ---"
  ${PY} scripts/preindex_md_to_sql_only.py --chunker-type section_l1 --checkpoint-interval 100 2>&1 | tail -5

  # 3. Check marker progress
  SMB_MDS=$(ssh -o ConnectTimeout=5 kss "find /mnt/lsb/retrival/papers/markdown/iclr -name '*.md' 2>/dev/null | wc -l" 2>&1 || echo 0)
  ICLR_IN_SQL=$(${PY} -c "
import sqlite3
c = sqlite3.connect('data/db/retrieval.db')
print(c.execute(\"SELECT COUNT(*) FROM papers WHERE paper_id LIKE 'iclr%'\").fetchone()[0])
")
  echo "[progress] smb_mds=${SMB_MDS} iclr_in_sql=${ICLR_IN_SQL} target=${TARGET}"

  if [[ "${SMB_MDS}" -ge "${TARGET}" && "${ICLR_IN_SQL}" -ge "${TARGET}" ]]; then
    echo "[done] both marker and preindex caught up to target ${TARGET}"
    break
  fi

  echo "--- sleep ${INTERVAL}s ---"
  sleep "${INTERVAL}"
done

echo
echo "[$(date)] loop exited after ${iter} iterations"
