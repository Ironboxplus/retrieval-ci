#!/bin/bash
# The BGE-M3 vLLM service of an embed host: the resident server embed_worker.py and the backend's
# vllm_client are clients of. Usage: bge_m3_service.sh start|stop|status [extra `vllm serve` args]
#
#   start   runs `vllm serve` with deploy/compute88/vllm_bge_m3.yaml and cuBLAS fp16 accumulation (fp16_accumulation/
#           sitecustomize.py on PYTHONPATH, always; the user's choice 2026-11-14), detached (setsid nohup: it outlives
#           the wsl.exe / ssh session that started it, checked on arc99 2026-11-14), and waits for /health.
#           Refuses when this service already runs or the card has less than BGE_M3_VLLM_MIN_FREE_MB free.
#   stop    TERM to the server's process group (it was started as its own session), KILL after
#           BGE_M3_VLLM_STOP_GRACE_S (5 s; vLLM's graceful exit took 37 s on 2026-11-14), then waits until
#           no process of the group is left and the port is closed: a following start finds the card freed.
#   status  pid, /health and the served model.
#
# Host settings (defaults are arc99's: WSL2 Ubuntu-24.04 on the RTX 3090):
#   BGE_M3_VLLM_VENV          venv with vllm 0.33.0 and the bge_m3_sparse_plugin installed
#   BGE_M3_VLLM_MODEL         model directory; its path must name the profile's revision (the client checks)
#   BGE_M3_VLLM_PORT          18200 (the client's default URL)
#   BGE_M3_VLLM_STATE         directory of the pid file and the log
#   BGE_M3_VLLM_MIN_FREE_MB   4096
#   BGE_M3_VLLM_PROFILE_DIR   set = torch profiler on (POST /start_profile, /stop_profile), traces there
# Start from Windows: wsl.exe -d Ubuntu-24.04 -u root -e bash /mnt/c/Users/Arc/Prd/retrival/scripts/remote_embed/bge_m3_service.sh start
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
VENV=${BGE_M3_VLLM_VENV:-/root/vllm-bench/venv}
MODEL=${BGE_M3_VLLM_MODEL:-/root/vllm-bench/models/bge-m3-5617a9f61b028005a4858fdac845db406aefb181}
PORT=${BGE_M3_VLLM_PORT:-18200}
STATE=${BGE_M3_VLLM_STATE:-/root/bge-m3-service}
MIN_FREE=${BGE_M3_VLLM_MIN_FREE_MB:-4096}
PIDFILE=$STATE/server-$PORT.pid
LOG=$STATE/server-$PORT.log
URL=http://127.0.0.1:$PORT

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
group_alive() { pgrep -g "$1" >/dev/null 2>&1; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
healthy() { curl -s -m 3 --noproxy '*' "$URL/health" >/dev/null 2>&1; }

cmd=${1:-status}
shift || true
case "$cmd" in
start)
  mkdir -p "$STATE"
  if running; then echo "already running: pid $(cat "$PIDFILE") on $URL"; exit 0; fi
  if healthy; then echo "port $PORT already serves something not started here; refusing" >&2; exit 1; fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${free:-0}" -lt "$MIN_FREE" ]; then echo "GPU has ${free} MiB free, need $MIN_FREE; refusing" >&2; exit 1; fi
  extra=()
  # fp16 accumulation in every server process (API server and engine core import it at start-up)
  export PYTHONPATH="$HERE/fp16_accumulation${PYTHONPATH:+:$PYTHONPATH}"
  if [ -n "${BGE_M3_VLLM_PROFILE_DIR:-}" ]; then
    mkdir -p "$BGE_M3_VLLM_PROFILE_DIR"
    extra+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$BGE_M3_VLLM_PROFILE_DIR\",\"ignore_frontend\":true,\"torch_profiler_with_stack\":false}")
  fi
  # loopback only; the host's download proxy must not see it, and the model is local
  export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  {
    echo "=== $(date -Is) start port=$PORT model=$MODEL fp16acc=1 extra: $* ${extra[*]}"
  } >> "$LOG"
  setsid nohup "$VENV/bin/vllm" serve "$MODEL" --config "$REPO/deploy/compute88/vllm_bge_m3.yaml" \
    --port "$PORT" "${extra[@]}" "$@" >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for i in $(seq 1 ${BGE_M3_VLLM_READY_S:-600}); do
    if healthy; then echo "ready after ${i}s: pid $(cat "$PIDFILE") on $URL (log $LOG)"; exit 0; fi
    if ! running; then echo "server exited during start; log tail:" >&2; tail -40 "$LOG" >&2; rm -f "$PIDFILE"; exit 1; fi
    sleep 1
  done
  echo "not ready after ${BGE_M3_VLLM_READY_S:-600}s (still starting? log $LOG)" >&2; exit 1
  ;;
stop)
  if ! running; then echo "not running"; rm -f "$PIDFILE"; exit 0; fi
  pid=$(cat "$PIDFILE")
  before=$(gpu_used)
  started=$(date +%s)
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  # the whole group: the engine core is a child and can outlive the API server
  for i in $(seq 1 $((${BGE_M3_VLLM_STOP_GRACE_S:-5} * 4))); do group_alive "$pid" || break; sleep 0.25; done
  how=TERM
  if group_alive "$pid"; then
    how=KILL
    kill -KILL -- "-$pid" 2>/dev/null
    for i in $(seq 1 60); do group_alive "$pid" || break; sleep 0.25; done
  fi
  if group_alive "$pid"; then echo "process group $pid still alive after KILL: $(pgrep -g "$pid" | tr '\n' ' ')" >&2; exit 1; fi
  for i in $(seq 1 40); do port_open || break; sleep 0.25; done
  if port_open; then echo "port $PORT still open after the group exited; refusing to report stopped" >&2; exit 1; fi
  rm -f "$PIDFILE"
  echo "stopped pid $pid by $how in $(( $(date +%s) - started ))s; port $PORT closed; GPU memory ${before} -> $(gpu_used) MiB"
  ;;
status)
  if running; then echo "pid $(cat "$PIDFILE")"; else echo "not running"; fi
  if healthy; then curl -s -m 5 --noproxy '*' "$URL/v1/models"; echo; else echo "no /health on $URL"; fi
  ;;
*)
  echo "usage: $0 start|stop|status [extra vllm serve args]" >&2; exit 2
  ;;
esac
