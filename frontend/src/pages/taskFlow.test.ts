import { describe, expect, it } from 'vitest';

import {
  scaleNodeRadius, groupEventsBySubtask, buildSyntheticStageNodes, compressStageChainNodes,
  reachedStageLabel,
} from './taskFlow';
import type { TaskEvent } from '../types';

describe('scaleNodeRadius', () => {
  it('returns the minimum radius for a zero count', () => {
    expect(scaleNodeRadius(0, 100)).toBe(14);
  });

  it('returns the maximum radius when count equals the max', () => {
    expect(scaleNodeRadius(100, 100)).toBe(32);
  });

  it('scales linearly between min and max', () => {
    expect(scaleNodeRadius(50, 100)).toBe(23);
  });

  it('is the minimum radius when max is zero (avoids divide by zero)', () => {
    expect(scaleNodeRadius(0, 0)).toBe(14);
  });

  it('clamps to the maximum radius even if count exceeds max', () => {
    expect(scaleNodeRadius(150, 100)).toBe(32);
  });
});

describe('groupEventsBySubtask', () => {
  const event = (overrides: Partial<TaskEvent>): TaskEvent => ({
    id: 1, task_id: 1, event_type: 'progress', level: 'info', message: 'm', created_at: 'now',
    ...overrides,
  });

  it('buckets events by subtask_id', () => {
    const events = [
      event({ id: 1, subtask_id: 10 }),
      event({ id: 2, subtask_id: 20 }),
      event({ id: 3, subtask_id: 10 }),
    ];
    const grouped = groupEventsBySubtask(events);
    expect(grouped.get(10)?.map((e) => e.id)).toEqual([1, 3]);
    expect(grouped.get(20)?.map((e) => e.id)).toEqual([2]);
  });

  it('buckets task-level events (no subtask_id) under the null key', () => {
    const events = [event({ id: 1, subtask_id: undefined }), event({ id: 2, subtask_id: 10 })];
    const grouped = groupEventsBySubtask(events);
    expect(grouped.get(null)?.map((e) => e.id)).toEqual([1]);
    expect(grouped.get(10)?.map((e) => e.id)).toEqual([2]);
  });

  it('returns an empty map for no events', () => {
    expect(groupEventsBySubtask([]).size).toBe(0);
  });
});

describe('buildSyntheticStageNodes', () => {
  const event = (stage: string, overrides: Partial<TaskEvent> = {}): TaskEvent => ({
    id: 1, task_id: 1, event_type: 'progress', level: 'info', message: 'm', created_at: 'now', stage,
    ...overrides,
  });

  it('a still-running task colors only its last reached stage with the live status', () => {
    const events = [event('match_paper.queued'), event('match_paper.prepared'), event('match_paper.embedding')];
    const nodes = buildSyntheticStageNodes(events, 'running', 'match_paper');
    expect(nodes.map((n) => n.key)).toEqual(['match_paper.queued', 'match_paper.prepared', 'match_paper.embedding']);
    expect(nodes.map((n) => n.status)).toEqual(['completed', 'completed', 'running']);
    expect(nodes.every((n) => !n.hollow)).toBe(true);
  });

  it('a completed task colors its last reached stage completed, no terminal node added', () => {
    const events = [event('match_paper.queued'), event('match_paper.aggregate')];
    const nodes = buildSyntheticStageNodes(events, 'completed', 'match_paper');
    expect(nodes.map((n) => n.status)).toEqual(['completed', 'completed']);
    expect(nodes.some((n) => n.key === '__terminal__')).toBe(false);
  });

  it('a cancelled task appends an explicit 已取消 terminal node instead of recoloring the last reached stage', () => {
    const events = [event('match_paper.queued'), event('match_paper.prepared'), event('match_paper.embedding')];
    const nodes = buildSyntheticStageNodes(events, 'cancelled', 'match_paper');
    // The reached stages stay 'completed' (they genuinely happened); the
    // terminal marker is its own node with the cancelled status/label.
    expect(nodes[0].status).toBe('completed');
    expect(nodes[1].status).toBe('completed');
    expect(nodes[2].status).toBe('completed');
    const terminal = nodes.find((n) => n.key === '__terminal__');
    expect(terminal?.status).toBe('cancelled');
    expect(terminal?.label).toBe('已取消');
    expect(terminal?.hollow).toBeFalsy();
  });

  it('appends the remaining match_paper phases after the terminal node as hollow/pending', () => {
    const events = [event('match_paper.queued'), event('match_paper.prepared'), event('match_paper.embedding')];
    const nodes = buildSyntheticStageNodes(events, 'cancelled', 'match_paper');
    const afterTerminal = nodes.slice(nodes.findIndex((n) => n.key === '__terminal__') + 1);
    // embedding is reached; qdrant/rerank/aggregate never happened, completed
    // is excluded (a terminal run never reaches it).
    expect(afterTerminal.map((n) => n.key)).toEqual(['__future__qdrant', '__future__rerank', '__future__aggregate']);
    expect(afterTerminal.every((n) => n.hollow && n.status === 'pending')).toBe(true);
  });

  it('recolors an already-reached explicit terminal-status stage instead of appending a duplicate (real match_paper.cancelled event)', () => {
    // app/services/match_doc/jobs.py emits an explicit last-stage event whose
    // OWN stage suffix already names the terminal status (e.g.
    // stage="match_paper.cancelled") rather than just stopping mid-phase —
    // confirmed against a real cancelled task's event log.
    const events = [
      event('match_paper.queued'), event('match_paper.running'), event('match_paper.prepared'),
      event('match_paper.embedding'), event('match_paper.qdrant'), event('match_paper.rerank'),
      event('match_paper.cancelled'),
    ];
    const nodes = buildSyntheticStageNodes(events, 'cancelled', 'match_paper');
    // No duplicate/redundant terminal node appended.
    expect(nodes.filter((n) => n.status === 'cancelled')).toHaveLength(1);
    const terminal = nodes.find((n) => n.status === 'cancelled');
    expect(terminal?.key).toBe('match_paper.cancelled');
    expect(terminal?.label).toBe('已取消');
    // "How far did it get" is computed from the last REAL progress phase
    // (rerank), not the terminal marker itself — only 'aggregate' remains.
    const hollow = nodes.filter((n) => n.hollow);
    expect(hollow.map((n) => n.key)).toEqual(['__future__aggregate']);
  });

  it('does not append future hollow phases for a task_type with no known canonical order', () => {
    const events = [event('crawl_urls.fetching')];
    const nodes = buildSyntheticStageNodes(events, 'failed', 'crawl_urls');
    expect(nodes.some((n) => n.key === '__terminal__')).toBe(true);
    expect(nodes.some((n) => n.hollow)).toBe(false);
  });

  it('returns an empty array for no events', () => {
    expect(buildSyntheticStageNodes([], 'running', 'match_paper')).toEqual([]);
  });
});

describe('compressStageChainNodes', () => {
  const node = (key: string, label: string, status: string, hollow = false) => ({
    key, label, status, hollow, events: [{ id: key } as unknown as TaskEvent],
  });

  it('keeps every node full when the chain has 5 or fewer', () => {
    const nodes = [node('a', 'A', 'completed'), node('b', 'B', 'completed'), node('c', 'C', 'running')];
    expect(compressStageChainNodes(nodes)).toEqual(nodes.map((n) => ({ kind: 'full', node: n })));
  });

  // The real task-333 shape: 8 nodes, all distinct labels — 排队中/运行中/
  // 准备完成/向量化/向量检索/重排序 (all completed) -> 已取消 (terminal) ->
  // 汇总结果 (hollow/pending). At 8 nodes every label truncated inline
  // ("运行…", two indistinguishable "向量…"); the fix collapses the middle
  // completed run into dots, keeping full labels only for: first node, the
  // last completed node before the terminal, the terminal node itself, and
  // any pending/hollow node.
  it('collapses the middle completed run into dots, keeping first/last-completed/terminal/pending full', () => {
    const nodes = [
      node('queued', '排队中', 'completed'),
      node('running', '运行中', 'completed'),
      node('prepared', '准备完成', 'completed'),
      node('embedding', '向量化', 'completed'),
      node('qdrant', '向量检索', 'completed'),
      node('rerank', '重排序', 'completed'),
      node('cancelled', '已取消', 'cancelled'),
      node('aggregate', '汇总结果', 'pending', true),
    ];
    const result = compressStageChainNodes(nodes);
    expect(result.map((r) => r.kind)).toEqual([
      'full', 'dot', 'dot', 'dot', 'dot', 'full', 'full', 'full',
    ]);
    expect(result.map((r) => r.node.key)).toEqual(nodes.map((n) => n.key));
  });

  it('dedupes consecutive nodes sharing the same label, summing their events', () => {
    const nodes = [
      node('a1', 'X', 'completed'),
      node('a2', 'X', 'completed'),
      node('b', 'Y', 'completed'),
    ];
    const result = compressStageChainNodes(nodes);
    expect(result).toHaveLength(2);
    expect(result[0].node.events).toHaveLength(2);
    expect(result[0].node.key).toBe('a1');
  });

  it('a chain with no terminal node (still running) treats the last node as the protected tail', () => {
    const nodes = [
      node('a', 'A', 'completed'), node('b', 'B', 'completed'), node('c', 'C', 'completed'),
      node('d', 'D', 'completed'), node('e', 'E', 'completed'), node('f', 'F', 'running'),
    ];
    const result = compressStageChainNodes(nodes);
    // First (a), last-completed-before-tail (e), and the running tail (f)
    // stay full; b/c/d collapse.
    expect(result.map((r) => r.kind)).toEqual(['full', 'dot', 'dot', 'dot', 'full', 'full']);
  });
});

describe('reachedStageLabel', () => {
  // "更早的尝试" rows (design addendum: a status dot + the stage it reached +
  // relative time, not a bare timestamp) — this is the "stage it reached"
  // part, sourced from whichever of the two chain shapes the attempt has.
  it('prefers the last non-pending WorkflowNode (real workflow chain)', () => {
    const nodes = [
      { name: 'Download SIGIR 2026 PDFs', subtask_key: 'download', status: 'completed' },
      { name: 'Extract SIGIR 2026 (local marker)', subtask_key: 'extract', status: 'completed' },
      { name: 'Preindex SIGIR 2026 → SQL', subtask_key: 'preindex', status: 'failed' },
      { name: 'Embed SIGIR 2026 (local BGE-M3)', subtask_key: 'embed', status: 'pending' },
    ];
    expect(reachedStageLabel(nodes, [], 'ingest_pipeline')).toBe('预索引');
  });

  it('falls back to the last event carrying a stage when there are no WorkflowNode rows (synthetic chain)', () => {
    const events = [{ stage: 'match_paper.queued' }, { stage: 'match_paper.embedding' }, { stage: null }];
    expect(reachedStageLabel([], events, 'match_paper')).toBe('向量化');
  });

  it('is empty when neither source has anything to report', () => {
    expect(reachedStageLabel([], [], 'match_paper')).toBe('');
    expect(reachedStageLabel([{ name: 'x', subtask_key: null, status: 'pending' }], [], 'ingest_pipeline')).toBe('');
  });
});
