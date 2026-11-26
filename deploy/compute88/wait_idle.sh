#!/bin/bash
# Block until the backend has no pending or running tasks, then exit 0.
# Every queue job (journal_index, code_index, public_code_job) and every
# pipeline task is a tasks row, so the tasks API is the only reliable signal;
# quiet logs are not (a code index logs nothing per batch).
# Usage: wait_idle.sh [timeout_seconds]   exit 3 on timeout.
API=${API:-http://127.0.0.1:18080/api/tasks}
deadline=$(( $(date +%s) + ${1:-600} ))
busy() {
  local s n
  for s in running pending; do
    n=$(curl -fsS -m 5 "$API?status=$s&page_size=1" | python3 -c 'import sys,json;print(json.load(sys.stdin)["total"])' 2>/dev/null) || { echo "tasks API unreachable"; return 0; }
    [ "$n" = "0" ] || { echo "$s=$n"; return 0; }
  done
  return 1
}
while [ "$(date +%s)" -lt "$deadline" ]; do
  # Two idle reads 3 s apart, so a job submitted between them is caught.
  if ! busy >/dev/null; then sleep 3; busy >/dev/null || exit 0; fi
  sleep 5
done
echo "still busy: $(busy)" >&2
exit 3
