import type { TaskListResponse } from '../types';

type FetchPage = (page: number, size: number, status?: string) => Promise<TaskListResponse>;

/** Keep recent history while including active work regardless of its submission date. */
export async function loadTaskBoard(fetchPage: FetchPage): Promise<TaskListResponse> {
  const recent = await fetchPage(1, 100);
  const tasks = new Map(recent.tasks.map(task => [task.id, task]));
  for (const status of ['pending', 'running']) {
    let page = 1;
    let seen = 0;
    let total: number;
    do {
      const data = await fetchPage(page++, 100, status);
      data.tasks.forEach(task => tasks.set(task.id, task));
      seen += data.tasks.length;
      total = data.total;
      if (!data.tasks.length) break;
    } while (seen < total);
  }
  return { ...recent, tasks: [...tasks.values()].sort((a, b) => b.id - a.id) };
}
