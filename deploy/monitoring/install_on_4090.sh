#!/usr/bin/env bash
# Run on the 4090 AFTER `scp -r dist 4090:/data/arc/monitoring/`. Extracts the
# binaries fetched on Arc into $MON_HOME/bin and $MON_HOME/grafana-home.
set -euo pipefail
MON_HOME="${MON_HOME:-/data/arc/monitoring}"
D="$MON_HOME/dist"
cd "$MON_HOME"; mkdir -p bin
tmp="$(mktemp -d)"

tar xzf "$D/prometheus.tar.gz" -C "$tmp";        cp "$tmp"/prometheus-*/prometheus "$tmp"/prometheus-*/promtool bin/
tar xzf "$D/node_exporter.tar.gz" -C "$tmp";      cp "$tmp"/node_exporter-*/node_exporter bin/
tar xzf "$D/nvidia_gpu_exporter.tar.gz" -C "$tmp"; ( cp "$tmp"/nvidia_gpu_exporter bin/ 2>/dev/null || cp "$tmp"/*/nvidia_gpu_exporter bin/ )
tar xzf "$D/grafana.tar.gz" -C "$tmp";            rm -rf grafana-home && mv "$tmp"/grafana-* grafana-home
[ -f "$D/loki.zip" ]     && { unzip -o "$D/loki.zip" -d "$tmp" >/dev/null;     cp "$tmp"/loki-linux-amd64 bin/loki; }
[ -f "$D/promtail.zip" ] && { unzip -o "$D/promtail.zip" -d "$tmp" >/dev/null; cp "$tmp"/promtail-linux-amd64 bin/promtail; }

chmod +x bin/*
rm -rf "$tmp"
echo "✓ installed:"; bin/prometheus --version 2>&1 | head -1; grafana-home/bin/grafana --version 2>&1 | head -1
