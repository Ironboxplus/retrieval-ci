import { describe, expect, it } from 'vitest';

import {
  lifecycleColumnOf, bucketTasksByLifecycle, groupTaskStacks, boardGridTemplateColumns,
  isNarrowBoard, defaultMobileColumn, sortStacksForColumn, type TaskLike,
} from './taskColumns';

describe('lifecycleColumnOf', () => {
  it('maps the four primary statuses straight across', () => {
    expect(lifecycleColumnOf('pending')).toBe('pending');
    expect(lifecycleColumnOf('running')).toBe('running');
    expect(lifecycleColumnOf('completed')).toBe('completed');
    expect(lifecycleColumnOf('failed')).toBe('failed');
  });

  it('folds cancelled and any unknown terminal status into failed rather than dropping it', () => {
    expect(lifecycleColumnOf('cancelled')).toBe('failed');
    expect(lifecycleColumnOf('timeout')).toBe('failed');
  });
});

describe('bucketTasksByLifecycle', () => {
  const task = (id: number, status: string): TaskLike => ({ id, task_type: 'ingest_pipeline', status, created_at: '2026-01-01T00:00:00' });

  it('every task lands in exactly one bucket', () => {
    const tasks = [task(1, 'pending'), task(2, 'running'), task(3, 'completed'), task(4, 'failed'), task(5, 'cancelled')];
    const buckets = bucketTasksByLifecycle(tasks);
    expect(buckets.pending.map((t) => t.id)).toEqual([1]);
    expect(buckets.running.map((t) => t.id)).toEqual([2]);
    expect(buckets.completed.map((t) => t.id)).toEqual([3]);
    expect(buckets.failed.map((t) => t.id)).toEqual([4, 5]);
    const total = Object.values(buckets).reduce((n, b) => n + b.length, 0);
    expect(total).toBe(tasks.length);
  });

  it('is all-empty buckets for no tasks', () => {
    const buckets = bucketTasksByLifecycle([]);
    expect(buckets).toEqual({ pending: [], running: [], completed: [], failed: [] });
  });
});

describe('groupTaskStacks', () => {
  const matchTask = (id: number, paperId: string, createdAt: string): TaskLike => ({
    id, task_type: 'match_paper', status: 'completed', created_at: createdAt, params: { paper_id: paperId },
  });

  it('stacks repeated attempts against the same subject, latest first', () => {
    const tasks = [
      matchTask(1, 'iclr-2024-x', '2026-01-01T00:00:00'),
      matchTask(2, 'iclr-2024-x', '2026-01-03T00:00:00'),
      matchTask(3, 'iclr-2024-x', '2026-01-02T00:00:00'),
    ];
    const stacks = groupTaskStacks(tasks);
    expect(stacks).toHaveLength(1);
    expect(stacks[0].latest.id).toBe(2);
    expect(stacks[0].earlier.map((t) => t.id)).toEqual([3, 1]);
  });

  it('keeps different subjects as separate single-task stacks', () => {
    const tasks = [
      matchTask(1, 'a', '2026-01-01T00:00:00'),
      matchTask(2, 'b', '2026-01-01T00:00:00'),
    ];
    const stacks = groupTaskStacks(tasks);
    expect(stacks).toHaveLength(2);
    expect(stacks.every((s) => s.earlier.length === 0)).toBe(true);
  });

  it('is empty for no tasks', () => {
    expect(groupTaskStacks([])).toEqual([]);
  });
});

describe('boardGridTemplateColumns', () => {
  // An explicit CSS grid track per column (+ a connector track between each
  // pair) — not flex. Flex items default to min-width: auto, so a long card
  // subtitle could refuse to shrink below its content width and visually
  // overlap the next column even though the flex boxes themselves never
  // did; grid tracks are hard boundaries content can't spill past.
  it('gives every empty column an auto (rail) track', () => {
    const counts = { pending: 0, running: 0, completed: 0, failed: 0 };
    expect(boardGridTemplateColumns(counts)).toBe('auto auto auto auto auto auto auto');
  });

  it('gives non-empty pending/completed a 1fr track and non-empty running/failed a 2fr track', () => {
    const counts = { pending: 1, running: 1, completed: 1, failed: 1 };
    expect(boardGridTemplateColumns(counts)).toBe(
      'minmax(0, 1fr) auto minmax(0, 2fr) auto minmax(0, 1fr) auto minmax(0, 2fr)',
    );
  });

  it('interleaves one auto connector track between every pair of column tracks', () => {
    const counts = { pending: 0, running: 3, completed: 0, failed: 5 };
    // Split on top-level tokens only — a naive split(' ') would over-split
    // "minmax(0, 2fr)" on the space after its comma.
    const tracks = boardGridTemplateColumns(counts).match(/auto|minmax\([^)]*\)/g);
    // 4 columns + 3 connectors = 7 tracks total.
    expect(tracks).toHaveLength(7);
  });

  it('matches the real board shape from the reported overlap bug: only completed and failed non-empty', () => {
    const counts = { pending: 0, running: 0, completed: 20, failed: 5 };
    expect(boardGridTemplateColumns(counts)).toBe(
      'auto auto auto auto minmax(0, 1fr) auto minmax(0, 2fr)',
    );
  });
});

describe('isNarrowBoard', () => {
  // Below the lg breakpoint (<992px, antd's own breakpoint and the one
  // Layout.tsx's sider/mobile-drawer split already uses) the lifecycle
  // board collapses from 4 side-by-side columns to a single selected one —
  // a 390px phone squeezed 完成 to ~70px with invisible card titles.
  it('is true below 992px', () => {
    expect(isNarrowBoard(390)).toBe(true);
    expect(isNarrowBoard(768)).toBe(true);
    expect(isNarrowBoard(991)).toBe(true);
  });

  it('is false at 992px and above', () => {
    expect(isNarrowBoard(992)).toBe(false);
    expect(isNarrowBoard(1440)).toBe(false);
  });
});

describe('defaultMobileColumn', () => {
  // First non-empty of 运行中 → 排队 → 失败 → 完成 — running is most likely to
  // need attention right now; completed (usually the biggest bucket) is the
  // last resort so it doesn't dominate by default just because it's large.
  it('picks running when it has anything', () => {
    const counts = { pending: 3, running: 2, completed: 41, failed: 5 };
    expect(defaultMobileColumn(counts)).toBe('running');
  });

  it('falls through to pending, then failed, then completed', () => {
    expect(defaultMobileColumn({ pending: 3, running: 0, completed: 41, failed: 5 })).toBe('pending');
    expect(defaultMobileColumn({ pending: 0, running: 0, completed: 41, failed: 5 })).toBe('failed');
    expect(defaultMobileColumn({ pending: 0, running: 0, completed: 41, failed: 0 })).toBe('completed');
  });

  it('defaults to completed when every column is empty (nothing better to show)', () => {
    expect(defaultMobileColumn({ pending: 0, running: 0, completed: 0, failed: 0 })).toBe('completed');
  });
});

describe('sortStacksForColumn', () => {
  const stack = (id: number, created_at: string, last_heartbeat?: string, started_at?: string) => {
    const latest: TaskLike = { id, task_type: 'ingest_pipeline', status: 'running', created_at, last_heartbeat, started_at, params: { batch: id } };
    return { key: String(id), latest, earlier: [] };
  };

  it('puts the most recently updated running task first, not the newest created', () => {
    // Live .88 shape: 365 is embedding (heartbeat now), 401 was created last but waits on the GPU lock.
    const stacks = [
      stack(401, '2026-09-25T06:14:59', '2026-09-25T15:55:42', '2026-09-25T14:14:59'),
      stack(365, '2026-09-25T06:14:43', '2026-09-25T17:26:46', '2026-09-25T14:14:43'),
      stack(388, '2026-09-25T06:14:51', '2026-09-25T16:41:44', '2026-09-25T14:14:51'),
    ];
    expect(sortStacksForColumn('running', stacks).map((s) => s.latest.id)).toEqual([365, 388, 401]);
  });

  it('falls back to started_at, then creation order', () => {
    const stacks = [
      stack(1, '2026-09-25T01:00:00'),
      stack(2, '2026-09-25T02:00:00'),
      stack(3, '2026-09-25T00:00:00', undefined, '2026-09-25T09:00:00'),
    ];
    expect(sortStacksForColumn('running', stacks).map((s) => s.latest.id)).toEqual([3, 2, 1]);
  });

  it('keeps the other columns newest-created first', () => {
    const stacks = [stack(1, '2026-09-25T01:00:00', '2026-09-25T23:00:00'), stack(2, '2026-09-25T02:00:00')];
    expect(sortStacksForColumn('completed', stacks).map((s) => s.latest.id)).toEqual([2, 1]);
  });
});
