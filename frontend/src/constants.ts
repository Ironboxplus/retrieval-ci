/**
 * Shared UI constants + small formatters.
 *
 * Consolidates lookups/helpers that were copy-pasted across pages
 * (statusColor in Documents/Crawl/Tasks, the `slice(0,19)` date trim in
 * Documents/Crawl/Tasks). Single source of truth — add new statuses here.
 */

/**
 * Public deployment flag. The CI build for the public edge (BWH) sets
 * `VITE_PUBLIC_MODE=1`; the internal 4090 build leaves it unset. In public mode
 * the SPA hides admin surfaces (Dashboard/Documents/Tasks/Pipeline/Crawl) whose
 * backend endpoints are blocked at the Traefik edge, and lands on Search.
 */
export const PUBLIC_MODE = import.meta.env.VITE_PUBLIC_MODE === '1';

// Status → Ant tag color. Superset of every page's status vocabulary
// (document index states, crawl record states, task states). No key collides:
// the same status string always maps to the same color.
export const statusColor: Record<string, string> = {
  // document index / generic
  indexed: 'success',
  processing: 'processing',
  // shared
  pending: 'default',
  failed: 'error',
  completed: 'success',
  running: 'processing',
  // crawl
  success: 'success',
  skipped: 'default',
  // task
  cancelled: 'warning',
};

// Chinese label for the same status vocabulary as `statusColor` above — a
// state dot (antd <Badge status={statusColor[s]} text={statusLabel[s]} />)
// instead of a raw English status word in a Tag.
export const statusLabel: Record<string, string> = {
  indexed: '已索引',
  processing: '处理中',
  pending: '待处理',
  failed: '失败',
  completed: '已完成',
  running: '运行中',
  success: '成功',
  skipped: '已跳过',
  cancelled: '已取消',
};

// Trim an ISO timestamp to `YYYY-MM-DD HH:MM:SS` (drops millis/zone).
// Tolerates null/undefined → '-'.
export function formatDateTime(v?: string | null): string {
  return v ? v.slice(0, 19) : '-';
}
