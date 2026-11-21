#!/usr/bin/env bash
# Run on the 4090. Stops the observability stack (reverse of run_monitoring.sh).
set -euo pipefail
MON_HOME="${MON_HOME:-/data/arc/monitoring}"
RUN="$MON_HOME/run"
for n in grafana loki prometheus gpu_exporter node_exporter; do
  pf="$RUN/$n.pid"
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    kill "$(cat "$pf")" && echo "✓ stopped $n ($(cat "$pf"))"
  else
    echo "• $n not running"
  fi
  rm -f "$pf"
done
