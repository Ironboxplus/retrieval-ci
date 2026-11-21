import { taskGroupKey } from './taskBody';

/**
 * Pure lifecycle-column + stacking logic for the Tasks page (design addendum
 * §1.1: "lifecycle columns 排队 → 运行中 → 完成 / 失败 with directional
 * connectors" + "group repeated attempts of the same subject into one
 * stacked card, visual stack depth = attempt count, latest attempt on top").
 * Kept JSX-free so the grouping rules are unit-testable without MorphGrid.
 */

export type LifecycleColumn = 'pending' | 'running' | 'completed' | 'failed';
export const LIFECYCLE_COLUMNS: LifecycleColumn[] = ['pending', 'running', 'completed', 'failed'];

export interface TaskLike {
  id: number;
  task_type: string;
  status: string;
  created_at: string;
  started_at?: string | null;
  last_heartbeat?: string | null;
  params?: Record<string, unknown> | null;
}

/** Every task lands in exactly one of the four lifecycle columns — 'cancelled'
 * folds into 'failed' (closest terminal-non-success fit) rather than being
 * silently dropped from the board. */
export function lifecycleColumnOf(status: string): LifecycleColumn {
  if (status === 'pending') return 'pending';
  if (status === 'running') return 'running';
  if (status === 'completed') return 'completed';
  return 'failed'; // failed, cancelled, timeout, or any future terminal status
}

export function bucketTasksByLifecycle<T extends TaskLike>(tasks: T[]): Record<LifecycleColumn, T[]> {
  const buckets: Record<LifecycleColumn, T[]> = { pending: [], running: [], completed: [], failed: [] };
  for (const task of tasks) buckets[lifecycleColumnOf(task.status)].push(task);
  return buckets;
}

export interface TaskStack<T> {
  key: string;
  latest: T;
  earlier: T[];
}

/** Groups tasks sharing a `taskGroupKey` into one stack, latest `created_at`
 * first — the compact card shows `latest`; `earlier` are attempt history. */
export function groupTaskStacks<T extends TaskLike>(tasks: T[]): TaskStack<T>[] {
  const byKey = new Map<string, T[]>();
  for (const task of tasks) {
    const key = taskGroupKey(task);
    const bucket = byKey.get(key);
    if (bucket) bucket.push(task);
    else byKey.set(key, [task]);
  }
  const stacks: TaskStack<T>[] = [];
  for (const [key, group] of byKey) {
    const sorted = [...group].sort((a, b) => b.created_at.localeCompare(a.created_at));
    stacks.push({ key, latest: sorted[0], earlier: sorted.slice(1) });
  }
  return stacks;
}

/** Board order within a column. 运行中 puts the task whose state moved most
 * recently on top: the pipeline stamps `last_heartbeat` on every stage change
 * and progress tick, so the one actually working rises while the dozens
 * parked behind a stage lock sink. `created_at` is UTC and the other two are
 * server-local, so they are never compared with each other. Other columns
 * stay newest-created first. */
export function sortStacksForColumn<T extends TaskLike>(column: LifecycleColumn, stacks: TaskStack<T>[]): TaskStack<T>[] {
  const byCreated = (a: TaskStack<T>, b: TaskStack<T>) => b.latest.created_at.localeCompare(a.latest.created_at);
  if (column !== 'running') return [...stacks].sort(byCreated);
  const touched = (s: TaskStack<T>) => s.latest.last_heartbeat ?? s.latest.started_at ?? '';
  return [...stacks].sort((a, b) => touched(b).localeCompare(touched(a)) || byCreated(a, b));
}

// running/failed are the columns worth a bigger share of board width when
// they have anything in them (something needs attention, or is happening
// right now) — pending/completed still get room, just proportionally less.
const PRIORITY_COLUMNS = new Set<LifecycleColumn>(['running', 'failed']);

/**
 * CSS `grid-template-columns` for the lifecycle board, one explicit track
 * per column plus a connector track between each pair — NOT flex. Flex
 * items default to `min-width: auto`, so a column with a long card subtitle
 * could refuse to shrink below that content's width and visually spill
 * into the next column even though the flex boxes themselves never
 * overlapped on paper; CSS grid tracks are hard boundaries content can't
 * cross (paired with `min-width: 0` on each column's own root, so it
 * actually honors the track it's given).
 *
 * An empty column collapses to `auto` (just enough for its "0" rail); a
 * non-empty one is `minmax(0, Nfr)`.
 */
export function boardGridTemplateColumns(stackCountByColumn: Record<LifecycleColumn, number>): string {
  const track = (column: LifecycleColumn): string => {
    if (stackCountByColumn[column] === 0) return 'auto';
    return PRIORITY_COLUMNS.has(column) ? 'minmax(0, 2fr)' : 'minmax(0, 1fr)';
  };
  return LIFECYCLE_COLUMNS
    .flatMap((column, i) => (i === 0 ? [track(column)] : ['auto', track(column)]))
    .join(' ');
}

// antd's own `lg` breakpoint — Layout.tsx's sider-vs-mobile-drawer split
// already uses the same 992px threshold, so the board collapsing to a
// single column lines up with where the rest of the chrome already went
// mobile, instead of introducing a second, different breakpoint.
const NARROW_BOARD_BREAKPOINT_PX = 992;

/** Below this width, 4 side-by-side grid columns don't fit — a 390px phone
 * squeezed 完成 to ~70px with invisible card titles and wrapped "N 天前"
 * vertically. */
export function isNarrowBoard(viewportWidth: number): boolean {
  return viewportWidth < NARROW_BOARD_BREAKPOINT_PX;
}

// On the narrow (single-column) board, which column shows by default before
// the user taps another one — 运行中 first (most likely to need attention
// right now), 完成 last (usually the biggest bucket; showing it by default
// just because it's large would bury the columns actually worth a look).
const MOBILE_DEFAULT_COLUMN_ORDER: LifecycleColumn[] = ['running', 'pending', 'failed', 'completed'];

export function defaultMobileColumn(stackCountByColumn: Record<LifecycleColumn, number>): LifecycleColumn {
  return MOBILE_DEFAULT_COLUMN_ORDER.find((c) => stackCountByColumn[c] > 0) ?? 'completed';
}
