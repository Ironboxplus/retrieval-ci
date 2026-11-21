/**
 * State of one task as seen through `GET /api/tasks/{id}/workflow/stream`
 * (app/api/routes/tasks.py). Events: `snapshot` / `final` carry `{task, workflow}`,
 * `workflow` carries the stage chain when it changed, `task-event` one TaskEvent
 * (its SSE `id:` is TaskEvent.id), `resource` one resource sample.
 */
import type { TaskDetail, TaskEvent, WorkflowNode } from '../../types';

/** Newest events kept in memory; older ones are counted in `dropped`. */
export const MAX_STREAM_EVENTS = 5000;

export type StreamConnection = 'connecting' | 'live' | 'reconnecting' | 'done' | 'error';

export interface ResourceSample {
  id: number;
  phase?: string | null;
  cpu_percent?: number | null;
  rss_bytes?: number | null;
}

export interface TaskStreamState {
  connection: StreamConnection;
  task: TaskDetail | null;
  nodes: WorkflowNode[];
  /** Ascending by id, unique. */
  events: TaskEvent[];
  /** Events pushed out of the buffer because it was full. */
  dropped: number;
  resource: ResourceSample | null;
  error: string | null;
}

export type TaskStreamAction =
  | { type: 'reset' }
  | { type: 'open' }
  | { type: 'reconnecting' }
  | { type: 'failed'; message: string }
  | { type: 'snapshot'; task: TaskDetail; nodes: WorkflowNode[] }
  | { type: 'workflow'; nodes: WorkflowNode[] }
  | { type: 'events'; events: TaskEvent[] }
  | { type: 'resource'; sample: ResourceSample }
  | { type: 'final'; task: TaskDetail; nodes: WorkflowNode[] }
  | { type: 'batch'; actions: TaskStreamAction[] };

export const initialTaskStream: TaskStreamState = {
  connection: 'connecting', task: null, nodes: [], events: [], dropped: 0, resource: null, error: null,
};

/** Merge `incoming` into the ascending buffer: ids already present are ignored
 * (a reconnect replays the snapshot's recent events), order is by id, and the
 * oldest fall off past `cap`. */
export function mergeEvents(
  existing: TaskEvent[], incoming: TaskEvent[], cap: number = MAX_STREAM_EVENTS,
): { events: TaskEvent[]; dropped: number } {
  if (incoming.length === 0) return { events: existing, dropped: 0 };
  const lastId = existing.length > 0 ? existing[existing.length - 1].id : -Infinity;
  let merged: TaskEvent[];
  if (incoming.every((e, i) => e.id > lastId && (i === 0 || e.id > incoming[i - 1].id))) {
    merged = existing.concat(incoming);
  } else {
    const byId = new Map<number, TaskEvent>();
    for (const e of existing) byId.set(e.id, e);
    for (const e of incoming) if (!byId.has(e.id)) byId.set(e.id, e);
    merged = [...byId.values()].sort((a, b) => a.id - b.id);
  }
  const over = Math.max(0, merged.length - cap);
  return { events: over > 0 ? merged.slice(over) : merged, dropped: over };
}

export function reduceTaskStream(state: TaskStreamState, action: TaskStreamAction): TaskStreamState {
  switch (action.type) {
    case 'reset':
      return initialTaskStream;
    case 'open':
      return state.connection === 'done' ? state : { ...state, connection: 'live', error: null };
    case 'reconnecting':
      return state.connection === 'done' ? state : { ...state, connection: 'reconnecting' };
    case 'failed':
      return { ...state, connection: 'error', error: action.message };
    case 'snapshot': {
      const merged = mergeEvents(state.events, [...(action.task.recent_events ?? [])].sort((a, b) => a.id - b.id));
      return {
        ...state, connection: 'live', error: null, task: action.task, nodes: action.nodes,
        events: merged.events, dropped: state.dropped + merged.dropped,
      };
    }
    case 'workflow':
      return { ...state, nodes: action.nodes };
    case 'events': {
      const merged = mergeEvents(state.events, action.events);
      return { ...state, events: merged.events, dropped: state.dropped + merged.dropped };
    }
    case 'resource':
      return { ...state, resource: action.sample };
    case 'final':
      return { ...state, connection: 'done', error: null, task: action.task, nodes: action.nodes };
    case 'batch':
      return action.actions.reduce(reduceTaskStream, state);
  }
}
