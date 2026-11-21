import { useEffect, useReducer } from 'react';

import type { TaskDetail, WorkflowNode } from '../../types';
import {
  initialTaskStream, reduceTaskStream,
  type ResourceSample, type TaskStreamAction, type TaskStreamState,
} from './taskStreamModel';

// Events and samples are applied in batches: a backlog replay can be thousands
// of messages, and one render per message would freeze the page.
const FLUSH_MS = 200;

interface TaskAndWorkflow { task: TaskDetail; workflow: { nodes: WorkflowNode[] } }

/**
 * Live view of one task over native EventSource (WHATWG HTML 9.2). The browser
 * owns reconnection: it waits the server's `retry:` and resends the last
 * `task-event` id as `Last-Event-ID`, so no event is lost or repeated. The
 * server ends the stream after `final`; the client must `close()` then, or the
 * browser reconnects and replays a snapshot every few seconds.
 */
export function useTaskStream(taskId: number | null): TaskStreamState {
  const [state, dispatch] = useReducer(reduceTaskStream, initialTaskStream);

  useEffect(() => {
    if (taskId == null) return;
    dispatch({ type: 'reset' });

    const source = new EventSource(`/api/tasks/${taskId}/workflow/stream`);
    let queue: TaskStreamAction[] = [];
    let timer: number | undefined;

    const flush = () => {
      window.clearTimeout(timer);
      timer = undefined;
      if (queue.length === 0) return;
      const actions = queue;
      queue = [];
      dispatch({ type: 'batch', actions });
    };
    const enqueue = (action: TaskStreamAction) => {
      queue.push(action);
      timer ??= window.setTimeout(flush, FLUSH_MS);
    };
    const parse = <T,>(e: Event): T => JSON.parse((e as MessageEvent<string>).data) as T;

    source.onopen = () => dispatch({ type: 'open' });
    source.addEventListener('snapshot', (e) => {
      flush();
      const { task, workflow } = parse<TaskAndWorkflow>(e);
      dispatch({ type: 'snapshot', task, nodes: workflow.nodes });
    });
    source.addEventListener('workflow', (e) => enqueue({ type: 'workflow', nodes: parse<{ nodes: WorkflowNode[] }>(e).nodes }));
    source.addEventListener('task-event', (e) => enqueue({ type: 'events', events: [parse(e)] }));
    source.addEventListener('resource', (e) => enqueue({ type: 'resource', sample: parse<ResourceSample>(e) }));
    source.addEventListener('final', (e) => {
      flush();
      const { task, workflow } = parse<TaskAndWorkflow>(e);
      dispatch({ type: 'final', task, nodes: workflow.nodes });
      source.close();
    });
    // The server's own `event: error` (task not found) arrives as a MessageEvent
    // with data; a dropped connection is a plain Event, and the browser retries it.
    source.addEventListener('error', (e) => {
      if (e instanceof MessageEvent && typeof e.data === 'string') {
        flush();
        dispatch({ type: 'failed', message: (JSON.parse(e.data) as { message?: string }).message ?? 'stream error' });
        source.close();
      } else {
        dispatch({ type: 'reconnecting' });
      }
    });

    return () => {
      window.clearTimeout(timer);
      source.close();
    };
  }, [taskId]);

  return state;
}
