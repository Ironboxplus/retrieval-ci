import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../client';
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

export function useTaskEvents(taskId: number | undefined, isActive = false) {
  return useQuery({
    queryKey: ['tasks', taskId, 'events'],
    queryFn: () =>
      api
        .get<{ total: number; events: TaskEvent[] }>(
          `/tasks/${taskId}/events`,
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

export function useCancelTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (taskId: number) =>
      api.post<ActionResponse>(`/tasks/${taskId}/cancel`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
  });
}

export function useRetryTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (taskId: number) =>
      api.post<ActionResponse>(`/tasks/${taskId}/retry`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
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
