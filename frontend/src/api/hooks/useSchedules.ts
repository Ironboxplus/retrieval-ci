import { keepPreviousData, useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import type { RJSFSchema } from '@rjsf/utils';

import api from '../client';
import type { TaskDetail, TaskEvent, WorkflowNode } from '../../types';
import { mergeEvents } from './taskStreamModel';
import type { PaperKind, ScheduleMetric, Tick, Trigger } from '../../pages/updatesModel';

// Scheduler API: app/api/routes/schedules.py. Internal / admin only.

export interface ScheduleRunningTask {
  id: number;
  task_type: string;
  status: string;
  progress: number;
  total: number;
  /** RFC 3339 with offset. */
  started_at: string | null;
}

export interface Schedule {
  id: number;
  key: string;
  kind: string;
  title: string;
  enabled: boolean;
  system: boolean;
  trigger_json: Trigger;
  params_json: Record<string, unknown>;
  catchup: string;
  misfire_grace_s: number | null;
  next_run_time: string | null;
  next_fire_times: string[];
  last_tick: Tick | null;
  running_task: ScheduleRunningTask | null;
  metrics: ScheduleMetric[];
}

export interface ScheduleKind {
  name: string;
  title: string;
  /** Pydantic JSON Schema of the kind's parameters. */
  params_schema: RJSFSchema;
}

export interface SchedulePatch {
  title?: string;
  enabled?: boolean;
  trigger?: Trigger;
  params?: Record<string, unknown>;
}

export interface FireResponse { status: string; task_id: number | null; reason: string | null }

export const SCHEDULES_REFETCH_MS = 10_000;
export const TICK_PAGE = 20;

export function useSchedules() {
  return useQuery({
    queryKey: ['schedules'],
    queryFn: () => api.get<Schedule[]>('/schedules').then((r) => r.data),
    refetchInterval: SCHEDULES_REFETCH_MS,
  });
}

export function useScheduleKinds() {
  return useQuery({
    queryKey: ['schedules', 'kinds'],
    queryFn: () => api.get<ScheduleKind[]>('/schedules/kinds').then((r) => r.data),
    staleTime: Infinity,
  });
}

/** Newest-first run history, `TICK_PAGE` ticks per page (`before_id` pagination). */
export function useScheduleTicks(scheduleId: number) {
  return useInfiniteQuery({
    queryKey: ['schedules', scheduleId, 'ticks'],
    initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam }) =>
      api
        .get<Tick[]>(`/schedules/${scheduleId}/ticks`, { params: { limit: TICK_PAGE, before_id: pageParam } })
        .then((r) => r.data),
    getNextPageParam: (last) => (last.length === TICK_PAGE ? last[last.length - 1].id : undefined),
    refetchInterval: SCHEDULES_REFETCH_MS,
  });
}

/** The server's next fire times for a trigger; the error text is the API's own. */
export function useTriggerPreview(trigger: Trigger | null) {
  return useQuery({
    queryKey: ['schedules', 'preview', trigger],
    enabled: trigger != null,
    queryFn: () =>
      api.post<{ next_fire_times: string[] }>('/schedules/preview', { trigger, count: 5 }).then((r) => r.data.next_fire_times),
    placeholderData: keepPreviousData,
    retry: false,
  });
}

/** Human text of a failed request: the API's `detail` (string or pydantic error list). */
export function apiErrorText(error: unknown): string {
  if (!axios.isAxiosError(error)) return error instanceof Error ? error.message : String(error);
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).join('.')}: ${d.msg ?? ''}`).join('；');
  }
  return error.message;
}

export function usePatchSchedule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: number; patch: SchedulePatch }) =>
      api.patch<Schedule>(`/schedules/${id}`, patch).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['schedules'] }),
  });
}

export function useRunSchedule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.post<FireResponse>(`/schedules/${id}/run`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['schedules'] }),
  });
}

// ─── One arXiv delta run ───────────────────────────────────────────────────

export interface RunPaper {
  paper_id: string;
  arxiv_id: string | null;
  title: string | null;
  source_version: string | null;
  withdrawn: boolean | null;
}

export interface RunPapersPage {
  task_id: number;
  kind: PaperKind;
  offset: number;
  limit: number;
  total: number;
  counts: Record<string, number>;
  papers: RunPaper[];
}

export function useRunPapers(taskId: number, kind: PaperKind, offset: number, limit: number) {
  return useQuery({
    queryKey: ['arxiv-run-papers', taskId, kind, offset, limit],
    queryFn: () =>
      api
        .get<RunPapersPage>(`/pipeline/arxiv/runs/${taskId}/papers`, { params: { kind, offset, limit } })
        .then((r) => r.data),
    placeholderData: keepPreviousData,
  });
}

// ─── Replay of a finished task ─────────────────────────────────────────────

const EVENT_PAGE = 500; // the API's cap (app/api/routes/tasks.py get_task_events)

export interface TaskReplay { task: TaskDetail; nodes: WorkflowNode[]; events: TaskEvent[] }

/** A finished task as stored: detail, stage chain and events (oldest `maxEvents`,
 * paged by `after_id`, since `/events` returns at most 500 per call). */
export function useTaskReplay(taskId: number, maxEvents: number) {
  return useQuery({
    queryKey: ['task-replay', taskId, maxEvents],
    staleTime: Infinity,
    queryFn: async (): Promise<TaskReplay> => {
      const [task, workflow] = await Promise.all([
        api.get<TaskDetail>(`/tasks/${taskId}`).then((r) => r.data),
        api.get<{ nodes: WorkflowNode[] }>(`/tasks/${taskId}/workflow`).then((r) => r.data),
      ]);
      const events: TaskEvent[] = [];
      for (;;) {
        const page = await api
          .get<{ total: number; events: TaskEvent[] }>(`/tasks/${taskId}/events`, {
            params: { limit: Math.min(EVENT_PAGE, maxEvents - events.length), after_id: events.at(-1)?.id },
          })
          .then((r) => r.data.events);
        events.push(...page);
        if (page.length < EVENT_PAGE || events.length >= maxEvents) break;
      }
      // Past the cap the paging only saw the oldest events; the detail's recent ones carry the end.
      const merged = mergeEvents(events, [...(task.recent_events ?? [])].sort((a, b) => a.id - b.id), maxEvents);
      return { task, nodes: workflow.nodes, events: merged.events };
    },
  });
}
