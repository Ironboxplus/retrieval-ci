/**
 * Timestamp parsing.
 *
 * Task, event, resource and schedule timestamps are RFC 3339 with an explicit
 * offset (`...Z` / `+08:00`), so `parseTimestamp` is a plain instant parse and
 * `instantMs` is what to compare or sort by (never the strings: offsets differ).
 *
 * `parseCreatedAt` remains for endpoints still sending naive strings (document
 * `created_at`, written as UTC with no zone suffix): a naive string is read as
 * UTC instead of the browser's local time.
 */

const HAS_ZONE = /Z$|[+-]\d{2}:?\d{2}$/;

/** An RFC 3339 timestamp (explicit offset) as a Date. */
export function parseTimestamp(raw: string): Date {
  return new Date(raw);
}

/** Epoch ms of an RFC 3339 timestamp; missing or unparseable is 0, so it sorts first and never yields NaN. */
export function instantMs(raw: string | null | undefined): number {
  const ms = raw ? Date.parse(raw) : NaN;
  return Number.isNaN(ms) ? 0 : ms;
}

/** Naive strings are UTC. */
export function parseCreatedAt(raw: string): Date {
  if (HAS_ZONE.test(raw)) return new Date(raw);
  const iso = raw.includes(' ') ? raw.replace(' ', 'T') : raw;
  return new Date(`${iso}Z`);
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

/** Local wall-clock `HH:mm:ss`. */
export function formatClock(date: Date): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(date.getHours())}:${p(date.getMinutes())}:${p(date.getSeconds())}`;
}
