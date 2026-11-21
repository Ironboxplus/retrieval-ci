#!/usr/bin/env bash
# Run on ARC (has the 10808 proxy + internet; the 4090 has no general egress, W111).
# Downloads linux-amd64 binaries for the 4090 monitoring stack into ./dist/, then
# you scp dist/ → 4090 and run install_on_4090.sh there. Versions are overridable
# env vars — bump as needed (verify the release asset names still match on 404).
set -euo pipefail
PROXY="${HTTPS_PROXY:-http://127.0.0.1:10808}"
cd "$(dirname "$0")"
mkdir -p dist && cd dist

PROM_VER="${PROM_VER:-3.1.0}"
NODE_VER="${NODE_VER:-1.8.2}"
GRAFANA_VER="${GRAFANA_VER:-11.4.0}"
LOKI_VER="${LOKI_VER:-3.3.2}"          # Phase 2
GPUX_VER="${GPUX_VER:-1.3.2}"

dl(){ echo "↓ $2"; curl -fL --proxy "$PROXY" --retry 5 --retry-all-errors --max-time 600 -C - -o "$2" "$1"; }

dl "https://github.com/prometheus/prometheus/releases/download/v${PROM_VER}/prometheus-${PROM_VER}.linux-amd64.tar.gz" prometheus.tar.gz
dl "https://github.com/prometheus/node_exporter/releases/download/v${NODE_VER}/node_exporter-${NODE_VER}.linux-amd64.tar.gz" node_exporter.tar.gz
dl "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VER}.linux-amd64.tar.gz" grafana.tar.gz
dl "https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v${GPUX_VER}/nvidia_gpu_exporter_${GPUX_VER}_linux_x86_64.tar.gz" nvidia_gpu_exporter.tar.gz
dl "https://github.com/grafana/loki/releases/download/v${LOKI_VER}/loki-linux-amd64.zip" loki.zip
dl "https://github.com/grafana/loki/releases/download/v${LOKI_VER}/promtail-linux-amd64.zip" promtail.zip

echo "✓ downloaded into $(pwd) — now: scp -r dist 4090:/data/arc/monitoring/dist && ssh 4090 'bash /data/arc/monitoring/install_on_4090.sh'"
