import type { TaskBrief, TaskEvent, WorkflowNode } from '../types';
import { groupEventsBySubtask, buildSyntheticStageNodes } from './taskFlow';
import { workflowNodeLabel } from './taskStageLabels';

export interface StageItem {
  key: string;
  label: string;
  status: string;
  hollow?: boolean;
  events: TaskEvent[];
  progress?: number;
  total?: number;
}

export type StageNode = Pick<WorkflowNode, 'id' | 'name' | 'status' | 'progress' | 'total'> & {
  subtask_key?: string | null;
  /** What `/workflow` actually sends for a node's stable stage key. */
  key?: string | null;
};

/** The stage chain of one task: real workflow nodes with their events, or —
 * for a task without nodes — synthetic stages derived from its events. */
export function buildStageItems(
  task: Pick<TaskBrief, 'task_type' | 'status'>,
  nodes: StageNode[],
  events: TaskEvent[],
): StageItem[] {
  if (nodes.length > 0) {
    const bySubtask = groupEventsBySubtask(events);
    return nodes.map((n) => ({
      key: String(n.id),
      label: workflowNodeLabel(n.subtask_key ?? n.key, task.task_type, n.name),
      status: n.status,
      events: bySubtask.get(n.id) ?? [],
      progress: n.progress,
      total: n.total,
    }));
  }
  return buildSyntheticStageNodes(events, task.status, task.task_type);
}
