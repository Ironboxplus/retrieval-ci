#!/usr/bin/env bash
# Launch 8 MinerU workers in tmux (one per GPU) on kss.
#
# Usage on kss:
#   bash scripts/remote_mineru/run_mineru_kss.sh \
#     /mnt/lsb/retrival/papers/pdf/iclr \
#     /mnt/lsb/retrival/papers/markdown/iclr \
#     [world_size=8]
#
# Each session is named ``mineru_g{N}``; tail logs at
# /tmp/retrival_mineru/rank{N}_gpu{N}.log. Done checkpoints are written to
# ``{OUTPUT}/done.rank{N}.txt`` so re-launching skips finished PDFs.
#
# Requires:
#   - ``mineru`` CLI in PATH inside the venv at /home/kss/arc/mineru/.venv
#   - MinerU model cache pre-warmed (otherwise first call per rank downloads)
#   - tmux installed
set -euo pipefail

INPUT="${1:-}"
OUTPUT="${2:-}"
WORLD="${3:-8}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/kss/arc/order/.venv/bin/activate}"
REPO="${REPO:-/home/kss/arc/retrival}"
LOG_DIR="${LOG_DIR:-/tmp/retrival_mineru}"
# Optional: comma-separated GPU IDs to use (e.g. "1,2,3,4,5,6,7" to skip dead GPU 0).
# If unset, rank N → GPU N (legacy behavior).
GPU_IDS="${GPU_IDS:-}"

if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
  echo "usage: $0 <input_pdf_dir> <output_md_dir> [world_size=8]" >&2
  exit 2
fi

if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
  if [[ ${#GPU_ARR[@]} -ne ${WORLD} ]]; then
    echo "GPU_IDS count (${#GPU_ARR[@]}) must equal WORLD (${WORLD})" >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT}" "${LOG_DIR}"

WORKER="${REPO}/scripts/remote_mineru/mineru_worker.py"
if [[ ! -f "${WORKER}" ]]; then
  echo "worker not found at ${WORKER}" >&2
  exit 2
fi

for ((rank=0; rank<WORLD; rank++)); do
  if [[ -n "${GPU_IDS}" ]]; then
    gpu="${GPU_ARR[rank]}"
  else
    gpu="${rank}"
  fi
  session="mineru_g${gpu}"
  log="${LOG_DIR}/rank${rank}_gpu${gpu}.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] tmux session ${session} already exists; kill it with: tmux kill-session -t ${session}"
    continue
  fi

  cmd="cd ${REPO} && source ${VENV_ACTIVATE} && \
export MINERU_MODEL_SOURCE=modelscope && \
export MODELSCOPE_CACHE=/mnt/nvme-disk/.cache/modelscope && \
export MINERU_DEVICE_MODE=cuda && \
export CUDNN_BENCHMARK=1 && \
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 && \
CUDA_VISIBLE_DEVICES=${gpu} \
MINERU_EXTRA_ARGS=\"-b pipeline -m auto\" \
python ${WORKER} \
  --input ${INPUT} \
  --output ${OUTPUT} \
  --rank ${rank} --world-size ${WORLD} \
  --recursive --progress-every 10 --timeout 600 \
  2>&1 | tee ${log}"

  tmux new-session -d -s "${session}" "${cmd}"
  echo "[launched] ${session}  (rank=${rank}, gpu=${gpu}, log=${log})"
done

echo
echo "All ${WORLD} workers launched."
echo "Tail any: tail -f ${LOG_DIR}/rank0_gpu0.log"
echo "List:     tmux ls | grep mineru_"
echo "Stop:     tmux kill-session -t mineru_g0  (repeat per rank)"
