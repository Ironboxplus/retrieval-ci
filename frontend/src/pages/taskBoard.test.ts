import { describe, expect, it } from 'vitest';
import { loadTaskBoard } from './taskBoard';
import type { TaskListResponse } from '../types';

describe('task board history and active work', () => {
  it('includes old active tasks beyond the first page and deduplicates recent tasks', async () => {
    const all = Array.from({ length: 260 }, (_, i) => ({
      id: i + 1, status: i < 150 ? 'running' : 'completed',
    })).reverse();
    const data = await loadTaskBoard(async (page, size, status) => {
      const rows = status ? all.filter(t => t.status === status) : all;
      return { tasks: rows.slice((page - 1) * size, page * size), total: rows.length,
        page, page_size: size } as TaskListResponse;
    });
    expect(data.tasks.filter(t => t.status === 'running')).toHaveLength(150);
    expect(data.tasks.filter(t => t.status === 'completed')).toHaveLength(100);
    expect(new Set(data.tasks.map(t => t.id)).size).toBe(data.tasks.length);
    expect(data.tasks[0].id).toBe(260);
    expect(data.tasks.at(-1)?.id).toBe(1);
  });
});
