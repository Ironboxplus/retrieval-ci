/**
 * Pure logic behind /updates: trigger wording and editing, tick -> history row,
 * run-result counts, relative/absolute times. Shapes follow the backend
 * (app/services/scheduler/triggers.py, app/api/routes/schedules.py).
 */

export const WEEKDAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'] as const;
export type Weekday = (typeof WEEKDAYS)[number];

const WEEKDAY_LABEL: Record<Weekday, string> = {
  mon: '一', tue: '二', wed: '三', thu: '四', fri: '五', sat: '六', sun: '日',
};

export interface DailyTrigger { type: 'daily'; time: string; timezone: string }
export interface WeeklyTrigger { type: 'weekly'; days: Weekday[]; time: string; timezone: string }
export interface IntervalTrigger { type: 'interval'; seconds: number; timezone: string }
export type Trigger = DailyTrigger | WeeklyTrigger | IntervalTrigger;

export const TRIGGER_TYPE_LABEL: Record<Trigger['type'], string> = {
  daily: '每日', weekly: '每周', interval: '间隔',
};

/** `90` -> `"90 秒"`, `1800` -> `"30 分钟"`, `7200` -> `"2 小时"`. */
export function formatIntervalSeconds(seconds: number): string {
  if (seconds % 3600 === 0) return `${seconds / 3600} 小时`;
  if (seconds % 60 === 0) return `${seconds / 60} 分钟`;
  return `${seconds} 秒`;
}

function sortedDays(days: Weekday[]): Weekday[] {
  return WEEKDAYS.filter((d) => days.includes(d));
}

/** "每日 03:00（Asia/Shanghai）" / "每周一、三 09:30（…）" / "每 30 秒". */
export function formatTrigger(trigger: Trigger): string {
  if (trigger.type === 'interval') return `每 ${formatIntervalSeconds(trigger.seconds)}`;
  if (trigger.type === 'weekly') {
    const days = sortedDays(trigger.days).map((d) => WEEKDAY_LABEL[d]).join('、');
    return `每周${days} ${trigger.time}（${trigger.timezone}）`;
  }
  return `每日 ${trigger.time}（${trigger.timezone}）`;
}

// ─── Trigger editing ───────────────────────────────────────────────────────

export type IntervalUnit = 'seconds' | 'minutes' | 'hours';
const UNIT_SECONDS: Record<IntervalUnit, number> = { seconds: 1, minutes: 60, hours: 3600 };
export const INTERVAL_UNIT_LABEL: Record<IntervalUnit, string> = {
  seconds: '秒', minutes: '分钟', hours: '小时',
};

/** Every field of every shape, so switching the type keeps what was typed. */
export interface TriggerDraft {
  type: Trigger['type'];
  time: string;
  days: Weekday[];
  amount: number;
  unit: IntervalUnit;
  timezone: string;
}

export function draftFromTrigger(trigger: Trigger): TriggerDraft {
  const draft: TriggerDraft = {
    type: trigger.type, time: '03:00', days: ['mon'], amount: 30, unit: 'minutes', timezone: trigger.timezone,
  };
  if (trigger.type === 'interval') {
    const unit: IntervalUnit = trigger.seconds % 3600 === 0 ? 'hours' : trigger.seconds % 60 === 0 ? 'minutes' : 'seconds';
    return { ...draft, amount: trigger.seconds / UNIT_SECONDS[unit], unit };
  }
  if (trigger.type === 'weekly') return { ...draft, time: trigger.time, days: sortedDays(trigger.days) };
  return { ...draft, time: trigger.time };
}

const CLOCK = /^([01]\d|2[0-3]):[0-5]\d$/;

/** The trigger a draft describes, or null while it is incomplete or invalid. */
export function triggerFromDraft(draft: TriggerDraft): Trigger | null {
  const timezone = draft.timezone.trim();
  if (!timezone) return null;
  if (draft.type === 'interval') {
    const seconds = Math.round(draft.amount * UNIT_SECONDS[draft.unit]);
    return Number.isFinite(seconds) && seconds >= 1 ? { type: 'interval', seconds, timezone } : null;
  }
  if (!CLOCK.test(draft.time)) return null;
  if (draft.type === 'weekly') {
    return draft.days.length > 0 ? { type: 'weekly', days: sortedDays(draft.days), time: draft.time, timezone } : null;
  }
  return { type: 'daily', time: draft.time, timezone };
}

// ─── Times ─────────────────────────────────────────────────────────────────

/** `YYYY-MM-DD HH:mm` of an instant, on the wall clock of `timeZone`. */
export function formatInZone(value: string | Date, timeZone: string): string {
  const date = typeof value === 'string' ? new Date(value) : value;
  if (Number.isNaN(date.getTime())) return '—';
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(date);
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? '';
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}`;
}

/** "30 秒后" / "3 小时前" — future and past, coarse like taskTime.relativeTimeFrom. */
export function relativeFromNow(value: string | Date, now: Date = new Date()): string {
  const date = typeof value === 'string' ? new Date(value) : value;
  const deltaS = Math.round((date.getTime() - now.getTime()) / 1000);
  const abs = Math.abs(deltaS);
  if (abs < 5) return '现在';
  let amount: string;
  if (abs < 60) amount = `${abs} 秒`;
  else if (abs < 3600) amount = `${Math.floor(abs / 60)} 分钟`;
  else if (abs < 86400) amount = `${Math.floor(abs / 3600)} 小时`;
  else amount = `${Math.floor(abs / 86400)} 天`;
  return deltaS > 0 ? `${amount}后` : `${amount}前`;
}

/** Whole seconds between two ISO instants; null while either is missing. */
export function secondsBetween(start?: string | null, end?: string | null): number | null {
  if (!start || !end) return null;
  const s = new Date(start).getTime();
  const e = new Date(end).getTime();
  return Number.isNaN(s) || Number.isNaN(e) ? null : Math.max(0, (e - s) / 1000);
}

/** How long a task has run: the server's figure once it is finished, the wall
 * clock since `startedAtMs` (epoch ms) while it still runs. */
export function elapsedSeconds(
  task: { status: string; duration_seconds?: number | null } | null | undefined,
  startedAtMs: number | null | undefined,
  nowMs: number,
): number | null {
  const running = task?.status === 'running' || task?.status === 'pending';
  if (running && startedAtMs != null) return Math.max(0, (nowMs - startedAtMs) / 1000);
  return task?.duration_seconds ?? null;
}

export function formatCount(value: number | null | undefined): string {
  return value == null ? '—' : value.toLocaleString('en-US');
}

// ─── Metrics ───────────────────────────────────────────────────────────────

export interface ScheduleMetric { name: string; value: number | string | null; unit: string }

/** The value with its unit; a gauge without a reading shows a dash, no unit. */
export function formatMetric(metric: ScheduleMetric): string {
  if (metric.value == null || metric.value === '') return '—';
  const value = typeof metric.value === 'number' ? formatCount(metric.value) : metric.value;
  return metric.unit ? `${value} ${metric.unit}` : value;
}

// ─── Run results and history ───────────────────────────────────────────────

/** Papers touched by one arXiv delta run, in the order the tabs show them. */
export const PAPER_KINDS = ['new', 'replaced', 'withdrawn', 'revived', 'html_missing', 'convert_rejected'] as const;
export type PaperKind = (typeof PAPER_KINDS)[number];

export const PAPER_KIND_LABEL: Record<PaperKind, string> = {
  new: '新论文', replaced: '新版本', withdrawn: '撤稿', revived: '恢复', html_missing: '缺 HTML', convert_rejected: '转换拒绝',
};

/** Only the arXiv delta kind reports the papers a run touched. */
export const hasRunPapers = (kind: string): boolean => kind === 'arxiv_delta';

/** Every count `result_json` of a delta run can carry, with its name (unit: 篇). */
const RESULT_COUNT_LABEL: [key: string, label: string][] = [
  ['selected', '本轮入选'],
  ['new', '新论文'],
  ['replaced', '新版本'],
  ['withdrawn', '撤稿'],
  ['revived', '恢复'],
  ['unchanged', '无变化'],
  ['html_missing', '缺 HTML'],
  ['convert_rejected', '转换拒绝'],
  ['convert_errors', '转换出错'],
  ['download_errors', '下载失败'],
  ['preindex_failed', '预索引失败'],
  ['superseded_deleted', '旧版已删'],
  ['superseded_kept', '旧版保留'],
  ['withdrawn_only', '仅更新撤稿标志'],
];

function num(result: Record<string, unknown> | null | undefined, key: string): number | null {
  const v = result?.[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export interface ResultCount { key: string; label: string; value: number }

/** The counts a result carries (absent keys are omitted, zeros kept). */
export function resultCounts(result: Record<string, unknown> | null | undefined): ResultCount[] {
  return RESULT_COUNT_LABEL.flatMap(([key, label]) => {
    const value = num(result, key);
    return value == null ? [] : [{ key, label, value }];
  });
}

const sumOrNull = (...values: (number | null)[]): number | null => {
  const present = values.filter((v): v is number => v != null);
  return present.length > 0 ? present.reduce((a, b) => a + b, 0) : null;
};

export interface TickTask {
  id: number;
  status: string;
  error?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  result_json?: Record<string, unknown> | null;
}

export interface Tick {
  id: number;
  schedule_id: number;
  scheduled_for: string | null;
  fired_at: string | null;
  trigger: string;
  status: string;
  reason?: string | null;
  task_id: number | null;
  attempt: number;
  task?: TickTask | null;
}

const ORIGIN_LABEL: Record<string, string> = { cron: '定时', manual: '手动', retry: '重试' };

/** A tick that never started a task says why; otherwise the task's own state shows. */
const TICK_ONLY_STATUS_LABEL: Record<string, string> = {
  skipped: '已跳过', missed: '已错过', error: '出错', enqueued: '已入队',
};

/** A count column of the history table, read off the run's `result_json`. */
export interface CountColumn {
  key: string;
  label: string;
  pick: (result: Record<string, unknown> | null) => number | null;
}

const field = (key: string, label: string): CountColumn => ({ key, label, pick: (r) => num(r, key) });

/** Errors that kept papers out of the run, summed (unit: 篇). */
const failedColumn: CountColumn = {
  key: 'failed',
  label: '失败',
  pick: (r) => sumOrNull(num(r, 'download_errors'), num(r, 'preindex_failed'), num(r, 'convert_errors')),
};

const COUNT_COLUMNS: Record<string, CountColumn[]> = {
  arxiv_delta: [
    field('new', '新论文'), field('replaced', '新版本'), field('withdrawn', '撤稿'),
    field('html_missing', '缺 HTML'), field('convert_rejected', '转换拒绝'), failedColumn,
  ],
};

/** The count columns a kind's history shows; a kind with no result counts has none. */
export function countColumns(kind: string): CountColumn[] {
  return COUNT_COLUMNS[kind] ?? [];
}

export interface TickRow {
  id: number;
  taskId: number | null;
  /** The instant the slot was due (falls back to when it fired). */
  time: string | null;
  origin: string;
  /** Task status, or null when the tick ended without a task. */
  taskStatus: string | null;
  /** Label for a tick that has no task to show a status for. */
  tickLabel: string | null;
  reason: string | null;
  durationSeconds: number | null;
  /** By column key, for the kind's `countColumns`. */
  counts: Record<string, number | null>;
}

export function tickRow(tick: Tick, kind: string = ''): TickRow {
  const result = tick.task?.result_json ?? null;
  const origin = ORIGIN_LABEL[tick.trigger] ?? tick.trigger;
  return {
    id: tick.id,
    taskId: tick.task_id,
    time: tick.scheduled_for ?? tick.fired_at,
    origin: tick.trigger === 'retry' && tick.attempt > 0 ? `${origin} ${tick.attempt}` : origin,
    taskStatus: tick.task?.status ?? null,
    tickLabel: tick.task ? null : (TICK_ONLY_STATUS_LABEL[tick.status] ?? tick.status),
    reason: tick.reason ?? tick.task?.error ?? null,
    durationSeconds: secondsBetween(tick.task?.started_at, tick.task?.completed_at),
    counts: Object.fromEntries(countColumns(kind).map((c) => [c.key, c.pick(result)])),
  };
}

/** Where a started task came from, read off `params_json.origin`. */
export function originLabel(origin: unknown): string | null {
  const trigger = (origin as { trigger?: unknown } | null | undefined)?.trigger;
  return typeof trigger === 'string' ? (ORIGIN_LABEL[trigger] ?? trigger) : null;
}
