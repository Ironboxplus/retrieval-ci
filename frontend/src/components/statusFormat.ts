/** Number/byte/time formatting for the header status bar and its panel. */

export function fmtCount(n: number | null | undefined): string {
  return n == null ? '—' : n.toLocaleString();
}

export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u += 1; }
  return `${v.toFixed(v >= 100 || u === 0 ? 0 : 1)} ${units[u]}`;
}

/** `15000` -> `"4 小时 10 分"`, `200000` -> `"2 天 7 小时"`. */
export function fmtUptime(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return '—';
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d} 天 ${h} 小时`;
  if (h > 0) return `${h} 小时 ${m} 分`;
  return `${m} 分`;
}

export function fmtSigned(n: number): string {
  return `${n > 0 ? '+' : ''}${n.toLocaleString()}`;
}

export function fmtPercent(ratio: number | null | undefined, digits = 1): string {
  return ratio == null ? '—' : `${(ratio * 100).toFixed(digits)}%`;
}
