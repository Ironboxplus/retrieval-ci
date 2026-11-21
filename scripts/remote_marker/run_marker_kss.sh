#!/usr/bin/env bash
# Launch N marker workers in tmux (one per GPU) on kss.
#
# Usage on kss:
#   bash scripts/remote_marker/run_marker_kss.sh \
#     /mnt/lsb/retrival/papers/pdf/iclr \
#     /mnt/lsb/retrival/papers/markdown_marker/iclr \
#     [world_size=8]
#
# Each session is named ``marker_g{N}``; tail logs at
# /tmp/retrival_marker/rank{N}_gpu{N}.log. Done checkpoints are written
# to ``{OUTPUT}/done.rank{N}.txt`` so re-launching skips finished PDFs.
#
# Requires:
#   - marker-pdf installed in the venv at ${VENV_ACTIVATE}
#   - tmux installed
set -euo pipefail

INPUT="${1:-}"
OUTPUT="${2:-}"
WORLD="${3:-8}"
# Stack RANKS_PER_GPU > 1 to fit multiple marker procs per GPU (3090 24GB
# typically holds 2 marker pipelines @ ~10-12GB each with full OCR + table).
# Effective total ranks = WORLD; CUDA_VISIBLE_DEVICES = rank % NUM_GPUS.
NUM_GPUS="${NUM_GPUS:-8}"
# Repo root + venv default to THIS script's own location (…/scripts/remote_marker/
# run_marker_kss.sh → repo = 3 dirs up), so the launcher self-locates on ANY box
# (kss /home/kss/arc/retrival, 4090 /data/arc/retrival, …) without the caller
# passing REPO. Env override still wins. Previously hardcoded /home/kss/... which
# doesn't exist on the 4090 → WORKER-not-found → exit 2 (2026-05-31 regression).
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="${REPO:-$(dirname "$(dirname "$_SELF_DIR")")}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO%/retrival}/order/.venv/bin/activate}"
LOG_DIR="${LOG_DIR:-/tmp/retrival_marker}"
# Per-rank CPU thread budget. Surya/torch CPU pre-processing benefits from
# 2-4 threads each; with WORLD=16 and CPU_THREADS=4 we use 64 threads —
# kss has 32-64 cores so slight oversubscription is acceptable for I/O-bound
# stages and helps cover GPU idle gaps.
CPU_THREADS="${CPU_THREADS:-4}"
# Optional: comma-separated physical GPU IDs to use (e.g. "1,2,3,4,5,6,7" to
# skip a dead GPU 0). When set, NUM_GPUS is forced to the array length and
# rank → gpu mapping uses GPU_ARR[rank % len].
GPU_IDS="${GPU_IDS:-}"
# Seconds to sleep between tmux session launches. Prevents 16 concurrent
# rglob("*.pdf") walks from saturating a CIFS/SMB mount (kernel goes into
# wait_for_response D+ across all ranks; SMB server bogged). 0 = parallel
# launch (only safe on local FS).
STAGGER="${STAGGER:-3}"

if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
  echo "usage: $0 <input_pdf_dir> <output_md_dir> [world_size=8]" >&2
  exit 2
fi

if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
  NUM_GPUS=${#GPU_ARR[@]}
  echo "[gpu-map] using ${NUM_GPUS} physical GPUs: ${GPU_IDS}"
fi

mkdir -p "${OUTPUT}" "${LOG_DIR}"

WORKER="${REPO}/scripts/remote_marker/marker_worker.py"
if [[ ! -f "${WORKER}" ]]; then
  echo "worker not found at ${WORKER}" >&2
  exit 2
fi

for ((rank=0; rank<WORLD; rank++)); do
  if [[ -n "${GPU_IDS}" ]]; then
    gpu="${GPU_ARR[rank % NUM_GPUS]}"
  else
    gpu=$((rank % NUM_GPUS))
  fi
  session="marker_r${rank}_g${gpu}"
  log="${LOG_DIR}/rank${rank}_gpu${gpu}.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] tmux session ${session} already exists; kill it with: tmux kill-session -t ${session}"
    continue
  fi

  # CUDA_VISIBLE_DEVICES is BOTH exported here (early so anything started by
  # the shell sees it) AND passed via --gpu to marker_worker.py. The Python
  # layer re-asserts it via os.environ before importing torch/marker, which
  # protects against libraries that don't honor the shell-inline form on
  # shared mixed-GPU boxes (witnessed 2026-05-31: marker landed on physical
  # GPU 0 despite CUDA_VISIBLE_DEVICES=2 in the pipeline preamble).
  # W80b: quote INPUT/OUTPUT/WORKER/log inside the tmux cmd string. Without
  # this, a path with a space (e.g. /mnt/.../usenix security/2025) is re-split
  # by the shell that tmux spawns, and argparse sees --input as 2 tokens
  # → marker_worker silently does the wrong thing or argparse rejects.
  cmd="cd ${REPO} && source ${VENV_ACTIVATE} && \
export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
export CUDA_VISIBLE_DEVICES=${gpu} && \
export TORCH_DEVICE=cuda && \
export CUDNN_BENCHMARK=1 && \
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 && \
export OMP_NUM_THREADS=${CPU_THREADS} && \
export MKL_NUM_THREADS=${CPU_THREADS} && \
export OPENBLAS_NUM_THREADS=${CPU_THREADS} && \
export NUMEXPR_NUM_THREADS=${CPU_THREADS} && \
python \"${WORKER}\" \
  --input \"${INPUT}\" \
  --output \"${OUTPUT}\" \
  --gpu ${gpu} \
  --rank ${rank} --world-size ${WORLD} \
  --recursive --progress-every 10 \
  2>&1 | tee \"${log}\""

  tmux new-session -d -s "${session}" "${cmd}"
  echo "[launched] ${session}  (rank=${rank}, gpu=${gpu}, cpu_threads=${CPU_THREADS}, log=${log})"
  if [[ "${STAGGER}" != "0" && $((rank + 1)) -lt ${WORLD} ]]; then
    sleep "${STAGGER}"
  fi
done

echo
echo "All ${WORLD} workers launched across ${NUM_GPUS} GPUs."
echo "Tail any: tail -f ${LOG_DIR}/rank0_gpu0.log"
echo "List:     tmux ls | grep marker_"
echo "Stop:     for r in \$(seq 0 $((WORLD-1))); do tmux kill-session -t marker_r\${r}_g\$((r % NUM_GPUS)); done"
