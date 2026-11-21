import { describe, expect, it } from 'vitest';

import type { TaskDetail, TaskEvent, WorkflowNode } from '../../types';
import { initialTaskStream, mergeEvents, reduceTaskStream, type TaskStreamAction, type TaskStreamState } from './taskStreamModel';

const ev = (id: number, message = `m${id}`): TaskEvent => ({
  id, task_id: 1, event_type: 'stage.log', level: 'info', message, created_at: '2026-11-03T00:00:00',
});
const task = (status: string, recent: TaskEvent[] = []): TaskDetail => ({ id: 1, status, recent_events: recent } as TaskDetail);
const node = (id: number, status: string): WorkflowNode => ({ id, name: `n${id}`, status, progress: 0, total: 0 });

const run = (actions: TaskStreamAction[], from: TaskStreamState = initialTaskStream) =>
  actions.reduce(reduceTaskStream, from);

describe('mergeEvents', () => {
  it('appends in order', () => {
    expect(mergeEvents([ev(1), ev(2)], [ev(3), ev(4)]).events.map((e) => e.id)).toEqual([1, 2, 3, 4]);
  });

  it('ignores ids it already has and keeps ascending order', () => {
    const merged = mergeEvents([ev(5), ev(6)], [ev(3), ev(6), ev(7)]);
    expect(merged.events.map((e) => e.id)).toEqual([3, 5, 6, 7]);
  });

  it('drops the oldest past the cap and counts them', () => {
    const merged = mergeEvents([ev(1), ev(2), ev(3)], [ev(4), ev(5)], 4);
    expect(merged.events.map((e) => e.id)).toEqual([2, 3, 4, 5]);
    expect(merged.dropped).toBe(1);
  });

  it('returns the same array when nothing arrives', () => {
    const existing = [ev(1)];
    expect(mergeEvents(existing, []).events).toBe(existing);
  });
});

describe('reduceTaskStream', () => {
  it('snapshot seeds the task, stages and its recent events (newest-first on the wire)', () => {
    const s = run([{ type: 'snapshot', task: task('running', [ev(9), ev(8)]), nodes: [node(1, 'running')] }]);
    expect(s.connection).toBe('live');
    expect(s.events.map((e) => e.id)).toEqual([8, 9]);
    expect(s.nodes).toHaveLength(1);
  });

  it('replayed older events slot in before the snapshot ones', () => {
    const s = run([
      { type: 'snapshot', task: task('running', [ev(9), ev(8)]), nodes: [] },
      { type: 'events', events: [ev(1), ev(2), ev(8)] },
    ]);
    expect(s.events.map((e) => e.id)).toEqual([1, 2, 8, 9]);
  });

  it('a second snapshot after a reconnect does not duplicate events', () => {
    const s = run([
      { type: 'snapshot', task: task('running', [ev(2), ev(1)]), nodes: [] },
      { type: 'reconnecting' },
      { type: 'snapshot', task: task('running', [ev(3), ev(2), ev(1)]), nodes: [] },
    ]);
    expect(s.events.map((e) => e.id)).toEqual([1, 2, 3]);
    expect(s.connection).toBe('live');
  });

  it('workflow replaces the stage chain', () => {
    const s = run([
      { type: 'snapshot', task: task('running'), nodes: [node(1, 'running'), node(2, 'pending')] },
      { type: 'workflow', nodes: [node(1, 'completed'), node(2, 'running')] },
    ]);
    expect(s.nodes.map((n) => n.status)).toEqual(['completed', 'running']);
  });

  it('final ends the stream and a late reconnect signal cannot reopen it', () => {
    const s = run([
      { type: 'snapshot', task: task('running'), nodes: [] },
      { type: 'final', task: task('completed'), nodes: [node(1, 'completed')] },
      { type: 'reconnecting' },
      { type: 'open' },
    ]);
    expect(s.connection).toBe('done');
    expect(s.task?.status).toBe('completed');
  });

  it('a batch applies its actions in order', () => {
    const s = run([{ type: 'batch', actions: [{ type: 'events', events: [ev(1)] }, { type: 'events', events: [ev(2)] }] }]);
    expect(s.events.map((e) => e.id)).toEqual([1, 2]);
  });

  it('keeps the last resource sample', () => {
    const s = run([{ type: 'resource', sample: { id: 1, cpu_percent: 10 } }, { type: 'resource', sample: { id: 2, cpu_percent: 20 } }]);
    expect(s.resource?.cpu_percent).toBe(20);
  });

  it('failed records the message', () => {
    const s = run([{ type: 'failed', message: 'Task not found' }]);
    expect(s).toMatchObject({ connection: 'error', error: 'Task not found' });
  });
});
