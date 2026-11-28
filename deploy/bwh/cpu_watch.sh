#!/usr/bin/env bash
# BWH 1-core edge CPU watchdog.
#
# The public VPS (1 vCPU / 1 GB) was suspended by the provider for sustained CPU
# (W131). The observability stack (Loki on the 4090) was down through the whole
# window, so there was NO record of WHAT pegged the box. This script leaves that
# forensic trail: every INTERVAL it samples overall CPU%, load average, memory,
# established-connection count + top peer IPs, and the top processes by CPU — one
# compact line per sample, plus a DETAIL block whenever CPU crosses a threshold.
#
# Dependency-free (only /proc, ps, ss, awk) and deliberately cheap so the watcher
# itself adds ~0 load to the 1-core box. Self-rotating so it can't fill the disk.
#
# Usage:
#   ./cpu_watch.sh                              # foreground, 5s interval, ./cpu_watch.log
#   INTERVAL=5 ALERT=80 LOG=/var/log/cpu_watch.log nohup ./cpu_watch.sh >/dev/null 2>&1 &
#   snapshot:  ./cpu_watch.sh --once           # print one detailed sample and exit
#
# Env knobs:
#   INTERVAL  seconds between samples           (default 5)
#   ALERT     CPU%% that triggers a DETAIL dump (default 80)
#   LOG       output file                       (default ./cpu_watch.log)
#   MAXKB     rotate LOG when it exceeds this   (default 10240 = 10 MB)
#   DOCKER    1 = include `docker stats` in DETAIL dumps (costs ~1s; default 1)
#   DETAIL_GAP min seconds between DETAIL dumps  (default 60) — throttles the
#             expensive docker-stats hop so the watcher can't pile onto an
#             already-saturated 1-core box that trips the threshold every sample
set -u

INTERVAL="${INTERVAL:-5}"
ALERT="${ALERT:-80}"
LOG="${LOG:-./cpu_watch.log}"
MAXKB="${MAXKB:-10240}"
DOCKER="${DOCKER:-1}"
DETAIL_GAP="${DETAIL_GAP:-60}"

ncpu=$(nproc 2>/dev/null || echo 1)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { printf '%s\n' "$*" >>"$LOG"; }

rotate() {  # keep one previous generation so the disk can't fill
  [ -f "$LOG" ] || return 0
  local kb
  kb=$(du -k "$LOG" 2>/dev/null | awk '{print $1}')
  [ -n "$kb" ] && [ "$kb" -gt "$MAXKB" ] && mv -f "$LOG" "$LOG.1"
}

# idle+iowait and total jiffies from /proc/stat (line "cpu ...")
read_cpu() {
  awk '/^cpu /{
        total=0; for(i=2;i<=NF;i++) total+=$i;
        idle=$5 + $6;                       # idle + iowait
        print idle, total }' /proc/stat
}

mem_line() {  # "used/total MB  avail MB"
  awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2}
       END{ printf "%d/%dMB avail=%dMB", (t-a)/1024, t/1024, a/1024 }' /proc/meminfo
}

conn_count() { ss -tnH state established 2>/dev/null | wc -l; }

top_peers() {  # top remote IPs by established-connection count
  ss -tnH state established 2>/dev/null \
    | awk '{n=split($4,a,":"); ip=$4; sub(/:[0-9]*$/,"",ip); print ip}' \
    | sort | uniq -c | sort -rn | head -8 \
    | awk '{printf "      %5d  %s\n",$1,$2}'
}

top_procs() {  # top processes by %CPU (normalized by ps to a single core)
  ps -eo pid,user,%cpu,%mem,comm --sort=-%cpu 2>/dev/null | head -n "${1:-9}"
}

detail() {  # full forensic dump, called on threshold breach or --once
  log "  ── DETAIL @ $(ts) ──"
  log "  top procs (%CPU):"
  top_procs 9 | sed 's/^/    /' | while IFS= read -r l; do log "$l"; done
  log "  established conns: $(conn_count)  | top peer IPs:"
  top_peers | while IFS= read -r l; do log "$l"; done
  if [ "$DOCKER" = "1" ] && command -v docker >/dev/null 2>&1; then
    log "  docker stats:"
    docker stats --no-stream --format '    {{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' \
      2>/dev/null | while IFS= read -r l; do log "$l"; done
  fi
  log "  ────────────────────"
}

sample_cpu_pct() {  # blocks for INTERVAL, prints integer CPU% busy across all cores
  set -- $(read_cpu); local i1=$1 t1=$2
  sleep "$INTERVAL"
  set -- $(read_cpu); local i2=$1 t2=$2
  local dt=$((t2 - t1)) di=$((i2 - i1))
  [ "$dt" -le 0 ] && { echo 0; return; }
  echo $(( (100 * (dt - di)) / dt ))
}

# ── one-shot snapshot mode ──
if [ "${1:-}" = "--once" ]; then
  cpu=$(sample_cpu_pct)
  read -r l1 l5 l15 _ </proc/loadavg
  log "$(ts) cpu=${cpu}% load=${l1}/${l5}/${l15} ncpu=${ncpu} mem=$(mem_line) conns=$(conn_count)"
  detail
  tail -n 40 "$LOG"
  exit 0
fi

log "=== cpu_watch start $(ts) interval=${INTERVAL}s alert=${ALERT}% gap=${DETAIL_GAP}s ncpu=${ncpu} pid=$$ ==="
trap 'log "=== cpu_watch stop $(ts) ==="; exit 0' INT TERM

last_detail=0
while :; do
  rotate
  cpu=$(sample_cpu_pct)                     # this call sleeps INTERVAL
  read -r l1 l5 l15 _ </proc/loadavg
  conns=$(conn_count)
  log "$(ts) cpu=${cpu}% load=${l1}/${l5}/${l15} mem=$(mem_line) conns=${conns}"
  # On a 1-core box load>1 means a backlog; dump detail on high CPU OR high load,
  # but no more than once per DETAIL_GAP so docker stats can't flood a hot box.
  load_hot=$(awk -v l="$l1" -v n="$ncpu" 'BEGIN{print (l > n*1.5) ? 1 : 0}')
  if [ "$cpu" -ge "$ALERT" ] || [ "$load_hot" = "1" ]; then
    now=$(date +%s)
    if [ $((now - last_detail)) -ge "$DETAIL_GAP" ]; then
      detail
      last_detail=$now
    fi
  fi
done
