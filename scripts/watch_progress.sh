#!/usr/bin/env bash
# Polling watcher: Stage C ingest + ICLR 2025 download.
# Writes a single-line JSON snapshot to /tmp/agent_progress.jsonl every 60s
# so the orchestrator can tail without blocking.
set -u

LOG=/tmp/agent_progress.jsonl
INTERVAL=${1:-60}

while :; do
  ts=$(date '+%Y-%m-%dT%H:%M:%S')
  qstat=$(curl -s --max-time 10 -o /tmp/qstat.json -w '%{http_code}' \
    http://localhost:6333/collections/papers || echo "-1")

  if [ "$qstat" = "200" ]; then
    points=$(grep -o '"points_count":[0-9]*' /tmp/qstat.json | head -1 | cut -d: -f2)
    indexed=$(grep -o '"indexed_vectors_count":[0-9]*' /tmp/qstat.json | head -1 | cut -d: -f2)
    status=$(grep -o '"status":"[a-z]*"' /tmp/qstat.json | head -1 | cut -d: -f2 | tr -d '"')
  else
    points=0; indexed=0; status="probe_fail"
  fi

  w1_done=$(wc -l < data/remote_embed/done.w1.txt 2>/dev/null || echo 0)
  w2_done=$(wc -l < data/remote_embed/done.w2.txt 2>/dev/null || echo 0)
  iclr_pdf=$(find data/papers/ccf_a_main_conference_corpus/pdf/iclr/2025/ -name '*.pdf' 2>/dev/null | wc -l)

  printf '{"ts":"%s","status":"%s","points":%s,"indexed":%s,"w1_done":%s,"w2_done":%s,"iclr25_pdf":%s}\n' \
    "$ts" "$status" "${points:-0}" "${indexed:-0}" "${w1_done:-0}" "${w2_done:-0}" "$iclr_pdf" >> "$LOG"
  sleep "$INTERVAL"
done
