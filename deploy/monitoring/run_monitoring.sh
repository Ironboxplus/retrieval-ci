#!/usr/bin/env bash
# Run on the 4090. Starts the observability stack as plain binaries (the box's
# snap-docker is broken). Exposure (W128 decision): ONLY Grafana binds the lab
# overlay IP; Prometheus/Loki/exporters stay on loopback (reach via Grafana or an
# ssh tunnel). All TSDB/log data on /data (nvme), never /home.
#
#   GF_ADMIN_PASSWORD=...  bash run_monitoring.sh           # start/refresh
#   bash stop_monitoring.sh                                  # stop
set -euo pipefail
MON_HOME="${MON_HOME:-/data/arc/monitoring}"
BIND_ADDR="${BIND_ADDR:-10.100.100.111}"      # lab overlay IP for the Grafana UI
DATA="$MON_HOME/data"; LOGS="$MON_HOME/logs"; RUN="$MON_HOME/run"
mkdir -p "$DATA/prometheus" "$DATA/grafana" "$DATA/loki" "$LOGS" "$RUN"

start(){ # name cmd...
  local name="$1"; shift
  local pf="$RUN/$name.pid"
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    echo "• $name already running ($(cat "$pf"))"; return; fi
  echo "▶ $name"
  nohup "$@" >"$LOGS/$name.log" 2>&1 &
  echo $! >"$pf"
}

# ── exporters (loopback; scraped by local Prometheus) ──
start node_exporter "$MON_HOME/bin/node_exporter" --web.listen-address=127.0.0.1:9100
start gpu_exporter  "$MON_HOME/bin/nvidia_gpu_exporter" --web.listen-address=127.0.0.1:9835

# ── Prometheus (loopback UI; datasource is local) ──
start prometheus "$MON_HOME/bin/prometheus" \
  --config.file="$MON_HOME/prometheus.yml" \
  --storage.tsdb.path="$DATA/prometheus" \
  --storage.tsdb.retention.time="${PROM_RETENTION:-30d}" \
  --web.listen-address=127.0.0.1:9090

# ── Loki (loopback; Phase 2 per-user logs). Skipped if binary/config absent. ──
if [ -x "$MON_HOME/bin/loki" ] && [ -f "$MON_HOME/loki.yml" ]; then
  start loki "$MON_HOME/bin/loki" -config.file="$MON_HOME/loki.yml"
fi

# ── Grafana (bound to lab overlay — the only externally-reachable UI) ──
export GF_SERVER_HTTP_ADDR="$BIND_ADDR" GF_SERVER_HTTP_PORT=3000
export GF_PATHS_DATA="$DATA/grafana" GF_PATHS_LOGS="$LOGS"
export GF_PATHS_PROVISIONING="$MON_HOME/grafana/provisioning"
export GF_SECURITY_ADMIN_PASSWORD="${GF_ADMIN_PASSWORD:?set GF_ADMIN_PASSWORD env}"
export GF_USERS_ALLOW_SIGN_UP=false
start grafana "$MON_HOME/grafana-home/bin/grafana" server --homepath "$MON_HOME/grafana-home"

sleep 2
echo "✓ stack up — Grafana: http://$BIND_ADDR:3000  (Prometheus loopback :9090)"
for n in node_exporter gpu_exporter prometheus loki grafana; do
  [ -f "$RUN/$n.pid" ] && printf "  %-14s pid %s\n" "$n" "$(cat "$RUN/$n.pid")"
done
