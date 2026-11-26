import type { TaskEvent } from '../types';
import { stageLabel, workflowNodeLabel, MATCH_PAPER_STAGE_ORDER } from './taskStageLabels';

/**
 * Pure layout helpers for the Tasks page structural redesign (frontend
 * refactor F1 design addendum, §1.1): the status-bucket row is a flow
 * diagram (排队 → 运行中 → 完成/失败) with node size = magnitude and
 * connectors = order, and the expanded task card groups its event log under
 * the workflow stage node each event belongs to, instead of one flat list.
 */

const MIN_RADIUS = 14;
const MAX_RADIUS = 32;

/** Node circle radius for the bucket-count flow diagram — linear between
 * MIN_RADIUS (count 0) and MAX_RADIUS (count == max), clamped so no bucket
 * ever shrinks the diagram below readable or grows a single node past its
 * neighbors' layout space. */
export function scaleNodeRadius(count: number, max: number): number {
  if (max <= 0) return MIN_RADIUS;
  const ratio = Math.max(0, Math.min(1, count / max));
  return Math.round(MIN_RADIUS + ratio * (MAX_RADIUS - MIN_RADIUS));
}

/**
 * Groups a task's event log by the workflow subtask it belongs to
 * (`event.subtask_id`, which is the same id as `WorkflowNode.id` — see
 * app/api/routes/tasks.py's `_events_response`), so the expanded task card
 * can render each stage node with only the events that happened during it.
 * Task-level events (no subtask_id, e.g. "task.started") land under the
 * `null` key.
 */
export function groupEventsBySubtask(events: TaskEvent[]): Map<number | null, TaskEvent[]> {
  const grouped = new Map<number | null, TaskEvent[]>();
  for (const event of events) {
    const key = event.subtask_id ?? null;
    const bucket = grouped.get(key);
    if (bucket) bucket.push(event);
    else grouped.set(key, [event]);
  }
  return grouped;
}

export interface ChainNode {
  key: string;
  label: string;
  status: string;
  hollow?: boolean;
  events: TaskEvent[];
}

export type CompressedStageEntry<T> = { kind: 'full'; node: T } | { kind: 'dot'; node: T };

const CHAIN_TERMINAL_STATUSES = new Set(['failed', 'cancelled', 'timeout']);

/**
 * Compacts a long stage chain so it never has to shrink or truncate labels
 * to fit (design addendum: "no label may be truncated" — an 8-node chain at
 * a fixed per-node width produced ambiguous truncations like two different
 * "向量…" labels). First dedupes consecutive nodes sharing the same Chinese
 * label, concatenating their event lists (real data can produce this from a
 * merged/retried stage). A chain of 5 nodes or fewer renders every node in
 * full. Longer chains keep full labels only for: the first node, the last
 * 'completed' node before the protected tail, the protected tail itself
 * (the terminal failed/cancelled/timeout node if the chain has one, else
 * the chain's actual last node), and any pending/hollow node — every other
 * 'completed' node in between collapses to a small dot (its label moves to
 * a hover tooltip at the call site).
 */
export function compressStageChainNodes<T extends ChainNode>(nodes: T[]): CompressedStageEntry<T>[] {
  const deduped: T[] = [];
  for (const n of nodes) {
    const prev = deduped[deduped.length - 1];
    if (prev && prev.label === n.label) {
      deduped[deduped.length - 1] = { ...prev, events: [...prev.events, ...n.events], status: n.status, hollow: n.hollow };
    } else {
      deduped.push(n);
    }
  }

  if (deduped.length <= 5) {
    return deduped.map((node) => ({ kind: 'full', node }));
  }

  const terminalIdx = deduped.findIndex((n) => CHAIN_TERMINAL_STATUSES.has(n.status));
  const tailIdx = terminalIdx >= 0 ? terminalIdx : deduped.length - 1;
  let lastCompletedIdx = -1;
  for (let i = tailIdx - 1; i >= 0; i -= 1) {
    if (deduped[i].status === 'completed') { lastCompletedIdx = i; break; }
  }

  return deduped.map((node, i) => {
    const keepFull = i === 0 || i === lastCompletedIdx || i === tailIdx || node.hollow || node.status === 'pending';
    return { kind: keepFull ? 'full' : 'dot', node };
  });
}

export interface SyntheticStageNode {
  key: string;
  label: string;
  events: TaskEvent[];
  status: string;
  /** A stage the run never reached — render as an outline/hollow node. */
  hollow?: boolean;
}

const SYNTHETIC_TERMINAL_STATUSES = new Set(['failed', 'cancelled', 'timeout']);

/**
 * Builds the node-chain for a task type with no formal workflow (match_paper
 * — see app/services/match_doc/jobs.py, which only ever calls `emit_event`,
 * never `upsert_subtask`) from its raw `event.stage` sequence
 * ("match_paper.queued" -> "match_paper.embedding" -> ...).
 *
 * A run that stopped abnormally (failed/cancelled/timeout) does NOT just
 * recolor whichever stage it happened to be in when it stopped — that reads
 * as "this stage succeeded". Two shapes are handled:
 *  - the backend's own except-block sometimes emits an explicit last-stage
 *    event whose stage SUFFIX already names the terminal status (e.g.
 *    "match_paper.cancelled" — app/services/match_doc/jobs.py) — that
 *    reached stage IS the terminal marker, so it's recolored in place
 *    rather than followed by a redundant second "已取消" node;
 *  - otherwise (the run was killed with no such event, e.g. a worker
 *    crash) an explicit terminal node is appended after the last reached
 *    stage.
 * Either way, when `taskType` has a known canonical phase order
 * (MATCH_PAPER_STAGE_ORDER; unknown task types get just the terminal
 * marker, no guessed-at future stages), the phases never reached — counted
 * from the last REAL progress phase, not the terminal marker itself —
 * render after that as hollow/pending nodes.
 */
export function buildSyntheticStageNodes(
  events: TaskEvent[],
  taskStatus: string,
  taskType?: string,
): SyntheticStageNode[] {
  const order: string[] = [];
  const byStage = new Map<string, TaskEvent[]>();
  for (const e of events) {
    const key = e.stage ?? '__none__';
    if (!byStage.has(key)) { byStage.set(key, []); order.push(key); }
    byStage.get(key)!.push(e);
  }
  if (order.length === 0) return [];

  const reached = order.map((key) => ({
    key,
    suffix: key === '__none__' ? null : (key.split('.').pop() ?? key),
    label: key === '__none__' ? '事件' : stageLabel(key),
    events: byStage.get(key)!,
  }));

  const isTerminal = SYNTHETIC_TERMINAL_STATUSES.has(taskStatus);
  const lastReached = reached[reached.length - 1];
  const lastIsExplicitTerminal = isTerminal && lastReached?.suffix === taskStatus;

  const nodes: SyntheticStageNode[] = reached.map((s, i) => {
    const isLast = i === reached.length - 1;
    const status = isLast && (!isTerminal || lastIsExplicitTerminal) ? taskStatus : 'completed';
    return { key: s.key, label: s.label, events: s.events, status };
  });

  if (isTerminal && !lastIsExplicitTerminal) {
    nodes.push({ key: '__terminal__', label: stageLabel(taskStatus), events: [], status: taskStatus });
  }

  if (isTerminal && taskType === 'match_paper') {
    // "How far did it get" is measured from the last REAL progress phase —
    // when the last reached stage IS the terminal marker itself, it isn't a
    // phase in MATCH_PAPER_STAGE_ORDER at all, so it's excluded here.
    const progressReached = lastIsExplicitTerminal ? reached.slice(0, -1) : reached;
    const lastProgressSuffix = progressReached[progressReached.length - 1]?.suffix;
    const idx = lastProgressSuffix ? MATCH_PAPER_STAGE_ORDER.indexOf(lastProgressSuffix) : -1;
    const remaining = idx >= 0 ? MATCH_PAPER_STAGE_ORDER.slice(idx + 1) : [];
    for (const suffix of remaining) {
      if (suffix === 'completed') continue; // a terminal run never reaches this one
      nodes.push({
        key: `__future__${suffix}`, label: stageLabel(suffix), events: [], status: 'pending', hollow: true,
      });
    }
  }
  return nodes;
}

/**
 * "Which stage did this earlier attempt reach" — for a "更早的尝试" row
 * (design addendum: each earlier attempt shows a status dot + the stage it
 * reached + relative time, not a bare timestamp). Prefers the last
 * non-pending WorkflowNode (real workflow, ingest/upload pipelines); falls
 * back to the last event carrying a `stage` (synthetic chains, e.g.
 * match_paper, which have no WorkflowNode rows at all); empty string when
 * neither source has anything to report yet.
 */
export function reachedStageLabel(
  nodes: { name: string; subtask_key?: string | null; key?: string | null; status: string }[],
  events: { stage?: string | null }[],
  taskType: string,
): string {
  const lastReachedNode = [...nodes].reverse().find((n) => n.status !== 'pending');
  if (lastReachedNode) return workflowNodeLabel(lastReachedNode.subtask_key ?? lastReachedNode.key, taskType, lastReachedNode.name);
  const lastStagedEvent = [...events].reverse().find((e) => e.stage);
  if (lastStagedEvent?.stage) return stageLabel(lastStagedEvent.stage);
  return '';
}
