/**
 * Task timestamp parsing (frontend refactor F1-f).
 *
 * Production fact confirmed on .88: `tasks.created_at` is written as UTC,
 * `tasks.started_at` is written as local wall-clock time — the two differ by
 * a fixed 8h offset in production. Neither string carries a timezone suffix,
 * so a plain `new Date(str)` would treat both as local and misrender
 * `created_at` by the local UTC offset. This file is the one place that
 * distinction is applied; nothing on the backend changes.
 */

const HAS_ZONE = /Z$|[+-]\d{2}:?\d{2}$/;

/** `created_at`: naive strings are UTC. */
export function parseCreatedAt(raw: string): Date {
  if (HAS_ZONE.test(raw)) return new Date(raw);
  const iso = raw.includes(' ') ? raw.replace(' ', 'T') : raw;
  return new Date(`${iso}Z`);
}

/** `started_at`: naive strings are already local wall-clock time. */
export function parseStartedAt(raw: string): Date {
  if (HAS_ZONE.test(raw)) return new Date(raw);
  const iso = raw.includes(' ') ? raw.replace(' ', 'T') : raw;
  return new Date(iso);
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** Coarse "N 分钟前" style label. Clamps future timestamps (clock skew
 * between client and server) to "刚刚" instead of a negative duration. */
export function relativeTimeFrom(date: Date, now: Date = new Date()): string {
  const deltaMs = now.getTime() - date.getTime();
  if (deltaMs < MINUTE) return '刚刚';
  if (deltaMs < HOUR) return `${Math.floor(deltaMs / MINUTE)} 分钟前`;
  if (deltaMs < DAY) return `${Math.floor(deltaMs / HOUR)} 小时前`;
  return `${Math.floor(deltaMs / DAY)} 天前`;
}

/** `51` -> `"51 秒"`, `3725` -> `"1 小时 2 分"`. null/negative -> null. */
export function formatDuration(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 60) return `${Math.round(seconds)} 秒`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return s > 0 ? `${m} 分 ${s} 秒` : `${m} 分`;
  }
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return m > 0 ? `${h} 小时 ${m} 分` : `${h} 小时`;
}

/** Local wall-clock `YYYY-MM-DD HH:mm`. */
export function formatDateTime(date: Date): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())} ${p(date.getHours())}:${p(date.getMinutes())}`;
}
