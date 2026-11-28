import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import api from '../client';
import { loadTaskBoard } from '../../pages/taskBoard';
import type {
  TaskListResponse,
  TaskDetail,
  TaskSubtask,
  TaskEvent,
  WorkflowResponse,
  ActionResponse,
} from '../../types';

export function useTasks(
  page = 1,
  pageSize = 20,
  status?: string,
  taskType?: string,
) {
  return useQuery({
    queryKey: ['tasks', page, pageSize, status, taskType],
    queryFn: () =>
      api
        .get<TaskListResponse>('/tasks', {
          params: { page, page_size: pageSize, status, task_type: taskType },
        })
        .then((r) => r.data),
    placeholderData: (prev) => prev,
    refetchInterval: 5000,
  });
}

export function useTaskBoard() {
  return useQuery({
    queryKey: ['tasks', 'board'],
    queryFn: () => loadTaskBoard((page, pageSize, status) => api.get<TaskListResponse>('/tasks', {
      params: { page, page_size: pageSize, status },
    }).then(r => r.data)),
    refetchInterval: 5000,
  });
}

export function useTaskDetail(taskId: number | undefined) {
  return useQuery({
    queryKey: ['tasks', taskId],
    queryFn: () =>
      api.get<TaskDetail>(`/tasks/${taskId}`).then((r) => r.data),
    enabled: !!taskId,
    refetchInterval: 3000,
  });
}

export function useTaskSubtasks(taskId: number | undefined, isActive = false) {
  return useQuery({
    queryKey: ['tasks', taskId, 'subtasks'],
    queryFn: () =>
      api
        .get<{ total: number; subtasks: TaskSubtask[] }>(
          `/tasks/${taskId}/subtasks`,
        )
        .then((r) => r.data),
    enabled: !!taskId,
    refetchInterval: isActive ? 2000 : false,
  });
}

/** The newest events of a task (ascending by id): a long task has thousands, and the
 * page shows where it is now, not how it started. */
export const TASK_EVENTS_TAIL = 500;

export function useTaskEvents(taskId: number | undefined, isActive = false) {
  return useQuery({
    queryKey: ['tasks', taskId, 'events'],
    queryFn: () =>
      api
        .get<{ total: number; events: TaskEvent[] }>(
          `/tasks/${taskId}/events`,
          { params: { tail: true, limit: TASK_EVENTS_TAIL } },
        )
        .then((r) => r.data),
    enabled: !!taskId,
    refetchInterval: isActive ? 2000 : false,
  });
}

export function useTaskWorkflow(taskId: number | undefined) {
  return useQuery({
    queryKey: ['tasks', taskId, 'workflow'],
    queryFn: () =>
      api.get<WorkflowResponse>(`/tasks/${taskId}/workflow`).then((r) => r.data),
    enabled: !!taskId,
  });
}

/** Text for a failed cancel/retry toast: the server's string `detail` (e.g. a
 * 409 reason), else the error message. */
export function taskActionErrorDetail(error: unknown): string {
  const e = error as { message?: unknown; response?: { data?: { detail?: unknown } } } | null | undefined;
  const detail = e?.response?.data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  return typeof e?.message === 'string' && e.message ? e.message : '未知错误';
}

export function useCancelTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (taskId: number) =>
      api.post<ActionResponse>(`/tasks/${taskId}/cancel`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
    onError: (err) => toast.error(`取消失败：${taskActionErrorDetail(err)}`),
  });
}

export function useRetryTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (taskId: number) =>
      api.post<ActionResponse>(`/tasks/${taskId}/retry`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
    onError: (err) => toast.error(`重试失败：${taskActionErrorDetail(err)}`),
  });
}

export function useDeleteTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (taskId: number) =>
      api.delete<ActionResponse>(`/tasks/${taskId}`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
  });
}
