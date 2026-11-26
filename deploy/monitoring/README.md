# Retrieval Observability

Compute-side metrics and logs live on the 4090. BWH remains a small stateless edge apart from its required gateway SQLite volume.

## Signals

| Question | Source |
|---|---|
| request rate, latency and 5xx | FastAPI Prometheus metrics |
| Qdrant health/count/latency | Qdrant `/metrics` and `/readyz` |
| CPU, RAM, disk and network | node_exporter |
| GPU utilization, memory, temperature and power | nvidia_gpu_exporter |
| per-user requests and billing outcomes | gateway structured logs -> Loki |

Do not put user IDs in Prometheus labels. Per-user analysis belongs in logs to avoid unbounded metric cardinality.

## Default Ports

| Component | Bind |
|---|---|
| Grafana | `10.100.100.111:3000` |
| Prometheus | `127.0.0.1:9090` |
| node_exporter | `127.0.0.1:9100` |
| nvidia_gpu_exporter | `127.0.0.1:9835` |
| Loki | `127.0.0.1:3100` |

## Repository Files

```text
prometheus.yml
grafana provisioning/dashboards
loki/promtail configuration
fetch_binaries.sh
install/run/stop helpers
```

External binaries are downloaded on Arc through `http://127.0.0.1:10808` and transferred to 4090 when the compute box has no general egress.

The repository has historical nohup/pidfile helpers, while core production services have moved toward `systemd --user`. Before operating monitoring, inspect the actual machine state and document the current process manager; do not assume either model solely from an old README.

## Minimum Alerts

- backend unavailable or sustained 5xx;
- Qdrant not ready;
- artifact mount missing;
- disk space below ingestion safety floor;
- GPU0 used by Retrieval;
- embedding/reranker GPU OOM;
- code-index queue stalled without progress;
- xray bridge or portal unavailable.
