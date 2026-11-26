#!/usr/bin/env bash
# Run 8 BGE-M3 embed workers on kss, one per GPU, in tmux sessions.
# Usage (on kss):
#   bash /home/kss/arc/retrival/scripts/remote_embed/run_embed_kss.sh
#
# Env overrides:
#   GPUS         "0 1 2 3 4 5 6 7" (default: all 8)
#   INPUT        /mnt/lsb/retrival/chunks
#   OUTPUT       /mnt/lsb/retrival/vectors
#   MODEL_DIR    /mnt/nvme-disk/.cache/modelscope/BAAI/bge-m3
#   VENV         /home/kss/arc/order/.venv
#   BATCH        64
#   WORK_DIR     /home/kss/arc/retrival
set -euo pipefail

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
INPUT="${INPUT:-/mnt/lsb/retrival/chunks}"
OUTPUT="${OUTPUT:-/mnt/lsb/retrival/vectors}"
BATCH="${BATCH:-64}"
# Self-locate the repo from this script's path (…/scripts/remote_embed/x.sh →
# repo = 2 dirs up) so the launcher runs on any box (kss /home/kss/arc/retrival,
# 4090 /data/arc/retrival, …) without a hardcoded /home/kss path. Venv mirrors
# the repo's sibling layout (…/order/.venv). Env overrides still win.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$(dirname "$_SELF_DIR")")}"
VENV="${VENV:-${WORK_DIR%/retrival}/order/.venv}"
# BGE-M3 weights: KSS keeps them on its NVMe cache; other boxes (4090) derive
# the path next to the repo parent (…/.cache/modelscope/BAAI/bge-m3). Prefer the
# KSS path when it exists so KSS behaviour is unchanged.
if [[ -z "${MODEL_DIR:-}" ]]; then
  if [[ -d /mnt/nvme-disk/.cache/modelscope/BAAI/bge-m3 ]]; then
    MODEL_DIR=/mnt/nvme-disk/.cache/modelscope/BAAI/bge-m3
  else
    MODEL_DIR="${WORK_DIR%/retrival}/.cache/modelscope/BAAI/bge-m3"
  fi
fi
LOG_DIR="${LOG_DIR:-/tmp/retrival_embed}"

mkdir -p "$OUTPUT" "$LOG_DIR"

read -ra GPU_ARR <<<"$GPUS"
WORLD_SIZE=${#GPU_ARR[@]}

echo "input    = $INPUT"
echo "output   = $OUTPUT"
echo "model    = $MODEL_DIR"
echo "venv     = $VENV"
echo "GPUs     = ${GPU_ARR[*]}  (world_size=$WORLD_SIZE)"
echo "logs     = $LOG_DIR"
echo

for i in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$i]}"
    rank="$i"
    sess="embed_g${gpu}"
    log="$LOG_DIR/rank${rank}_gpu${gpu}.log"

    tmux kill-session -t "$sess" 2>/dev/null || true
    cmd="cd $WORK_DIR && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
$VENV/bin/python3 scripts/remote_embed/embed_worker.py \
  --input '$INPUT' \
  --output '$OUTPUT' \
  --model-dir '$MODEL_DIR' \
  --rank $rank \
  --world-size $WORLD_SIZE \
  --batch-size $BATCH \
  2>&1 | tee '$log'"

    tmux new-session -d -s "$sess" "$cmd"
    echo "  launched $sess (rank=$rank gpu=$gpu) -> $log"
done

echo
echo "tmux ls:"
tmux ls | grep '^embed_' || true
echo
echo "monitor: tail -f $LOG_DIR/rank*.log"
