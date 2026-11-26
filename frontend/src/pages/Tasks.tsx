import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { AlertCircle, RefreshCw } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import ConfirmButton from '../components/ConfirmButton';
import { Bar, EventRow, MetaItem, SectionTitle, StageRow, StatusBadge } from '../components/TaskParts';
import {
  useTaskBoard, useTaskEvents, useTaskWorkflow,
  useCancelTask, useRetryTask,
} from '../api/hooks/useTasks';
import { useCrawlRecords } from '../api/hooks/useCrawl';
import { useDocumentDetail } from '../api/hooks/useDocuments';
import { usePaperDetail } from '../api/hooks/usePapers';
import type { TaskBrief, TaskEvent } from '../types';
import { selectedTaskIdFromSearch } from './taskSelection';
import { taskDisplayLabel, taskCanRetry, taskCanCancel, taskSubject, taskDocId } from './taskBody';
import {
  parseTimestamp, relativeTimeFrom, formatDuration, formatDateTime,
} from './taskTime';
import { groupEventsBySubtask, reachedStageLabel } from './taskFlow';
import { buildStageItems } from './taskStageItems';
import { styleOf } from './taskStatusStyle';
import {
  stageLabel, taskLevelEventLabel, taskStatusLabel, taskErrorLabel,
  pipelineProgressView,
} from './taskStageLabels';
import {
  LIFECYCLE_COLUMNS, bucketTasksByLifecycle, groupTaskStacks, lifecycleColumnOf,
  isNarrowBoard, defaultMobileColumn, sortStacksForColumn,
  type LifecycleColumn, type TaskStack,
} from './taskColumns';
import MorphGrid from '../components/morph/MorphGrid';
import MorphCard from '../components/morph/MorphCard';

interface TaskSubjectDisplay {
  text: string;
  /** A fallback for an unreachable reference, rendered muted. */
  muted: boolean;
}

// `taskSubject` never returns an id; this upgrades its fallback to the real
// paper/document title once that lookup resolves.
function useTaskSubjectDisplay(task: TaskBrief): TaskSubjectDisplay {
  const paperId = task.task_type === 'match_paper' ? (task.params ?? {}).paper_id : undefined;
  const paper = usePaperDetail(typeof paperId === 'string' ? paperId : undefined);
  const docId = taskDocId(task);
  const doc = useDocumentDetail(docId ?? undefined);
  if (task.task_type === 'match_paper' && paper.data?.title) return { text: paper.data.title, muted: false };
  if (task.task_type === 'upload_pipeline') {
    if (doc.data?.title) return { text: doc.data.title, muted: false };
    const status = axios.isAxiosError(doc.error) ? doc.error.response?.status : undefined;
    if (doc.isError && status === 404) return { text: '已删除的文档', muted: true };
    const fallback = taskSubject(task);
    return fallback ? { text: fallback, muted: false } : { text: '上传的文档', muted: true };
  }
  return { text: taskSubject(task), muted: false };
}

const COLUMN_LABEL: Record<LifecycleColumn, string> = {
  pending: '排队中', running: '运行中', completed: '已完成', failed: '失败 / 已取消',
};
const COLUMN_STATUS: Record<LifecycleColumn, string> = {
  pending: 'pending', running: 'running', completed: 'completed', failed: 'failed',
};

function taskPercent(task: TaskBrief): number | null {
  return task.total > 0 ? Math.round((task.progress / task.total) * 100) : null;
}

function taskDurationSeconds(task: TaskBrief): number | null {
  if (task.duration_seconds != null) return task.duration_seconds;
  if (task.started_at && task.completed_at) {
    return (parseTimestamp(task.completed_at).getTime() - parseTimestamp(task.started_at).getTime()) / 1000;
  }
  return null;
}

// ─── Compact card: what, on which subject, which state, how far, when, how long ───
function TaskCompactCard({ task, attempts }: { task: TaskBrief; attempts: number }) {
  const subject = useTaskSubjectDisplay(task);
  const pct = taskPercent(task);
  const pipe = task.status === 'running' ? pipelineProgressView(task) : null;
  const eta = formatDuration(pipe?.eta);
  const duration = formatDuration(taskDurationSeconds(task));
  const style = styleOf(task.status);

  return (
    <MorphCard>
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-semibold">{taskDisplayLabel(task)}</span>
        <StatusBadge status={task.status} />
      </div>
      <div title={subject.text} className={cn('mt-1 truncate text-xs text-muted-foreground', subject.muted && 'italic')}>
        {subject.text || ' '}
      </div>
      {pipe ? (
        <div className="mt-2.5 grid gap-1">
          <div className="flex items-center gap-2">
            <Bar pct={pipe.pct} className={style.bar} />
            <span className="text-[11px] tabular-nums text-muted-foreground">{pipe.pct}%</span>
          </div>
          <div className="flex justify-between gap-2 text-[11px] tabular-nums text-muted-foreground">
            <span className="truncate">{pipe.stage}</span>
            {eta && <span className="shrink-0">剩余约 {eta}</span>}
          </div>
        </div>
      ) : pct != null && task.status !== 'completed' && (
        <div className="mt-2.5 flex items-center gap-2">
          <Bar pct={pct} className={style.bar} />
          <span className="text-[11px] tabular-nums text-muted-foreground">{task.progress}/{task.total}</span>
        </div>
      )}
      {task.error_message && task.status !== 'completed' && (
        <div className="mt-1.5 truncate text-xs text-destructive" title={task.error_message}>
          {taskErrorLabel(task.error_message, task.task_type)}
        </div>
      )}
      <div className="mt-2.5 flex justify-between gap-2 border-t pt-2 text-[11px] text-muted-foreground">
        <span>创建于 {relativeTimeFrom(parseTimestamp(task.created_at))}{duration && <> · 耗时 {duration}</>}</span>
        {attempts > 1 && <span className="shrink-0">共 {attempts} 次尝试</span>}
      </div>
    </MorphCard>
  );
}

// Repeated attempts against the same subject stack into one card; the front
// card is the latest attempt and names the attempt count.
function StackedTaskCard({ stack }: { stack: TaskStack<TaskBrief> }) {
  const depth = Math.min(stack.earlier.length, 2);
  return (
    <div className="relative min-w-0" style={{ paddingBottom: depth * 4 }}>
      {Array.from({ length: depth }).map((_, i) => (
        <div
          key={i}
          className="absolute inset-y-0 rounded-xl border bg-card"
          style={{ left: (i + 1) * 6, right: (i + 1) * 6, transform: `translateY(${(i + 1) * 4}px)`, zIndex: -1 - i }}
        />
      ))}
      <TaskCompactCard task={stack.latest} attempts={stack.earlier.length + 1} />
    </div>
  );
}

function EarlierAttemptRow({ task, active, onSelect }: { task: TaskBrief; active: boolean; onSelect: () => void }) {
  const workflow = useTaskWorkflow(task.id);
  const events = useTaskEvents(task.id, false);
  const reached = reachedStageLabel(
    (workflow.data?.nodes ?? []).map((n) => ({ name: n.name, subtask_key: n.subtask_key, key: n.key, status: n.status })),
    events.data?.events ?? [],
    task.task_type,
  );
  const statusText = taskStatusLabel(task.status);
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'grid w-full grid-cols-[72px_1fr_auto] items-center gap-3 rounded-md px-2 py-1.5 text-left text-xs hover:bg-accent',
        active && 'bg-accent',
      )}
    >
      <StatusBadge status={task.status} />
      <span className="truncate text-muted-foreground">{reached && reached !== statusText ? `停在「${reached}」` : ''}</span>
      <span className="tabular-nums text-muted-foreground">{formatDateTime(parseTimestamp(task.created_at))}</span>
    </button>
  );
}

function TaskExpandedCard({ task, close, earlier = [] }: { task: TaskBrief; close: () => void; earlier?: TaskBrief[] }) {
  const [viewedId, setViewedId] = useState<number | null>(null);
  const activeTask = viewedId == null ? task : (earlier.find((t) => t.id === viewedId) ?? task);

  const workflow = useTaskWorkflow(activeTask.id);
  const events = useTaskEvents(activeTask.id, activeTask.status === 'running');
  const crawlRecords = useCrawlRecords(1, 50, task.task_type === 'crawl_urls' ? task.id : undefined);
  const cancelTask = useCancelTask();
  const retryTask = useRetryTask();

  const nodes = useMemo(() => workflow.data?.nodes ?? [], [workflow.data]);
  const eventList = useMemo(() => events.data?.events ?? [], [events.data]);
  const stages = useMemo(() => buildStageItems(activeTask, nodes, eventList), [activeTask, nodes, eventList]);
  const taskLevelEvents = useMemo(
    () => (nodes.length > 0 ? (groupEventsBySubtask(eventList).get(null) ?? []) : [])
      .map((e) => ({ event: e, label: taskLevelEventLabel(e) }))
      .filter((x): x is { event: TaskEvent; label: string } => x.label !== null),
    [nodes.length, eventList],
  );
  const pct = taskPercent(activeTask);
  const subject = useTaskSubjectDisplay(activeTask);
  const duration = formatDuration(taskDurationSeconds(activeTask));

  return (
    <div className="p-6">
      <div className="flex items-center gap-2.5 pr-8">
        <h3 className="text-lg font-semibold">{taskDisplayLabel(activeTask)}</h3>
        <StatusBadge status={activeTask.status} />
        {viewedId != null && (
          <button type="button" className="ml-auto text-xs text-primary hover:underline" onClick={() => setViewedId(null)}>
            返回最新一次
          </button>
        )}
      </div>
      <p className={cn('mt-0.5 text-sm text-muted-foreground', subject.muted && 'italic')}>{subject.text}</p>

      <div className="mt-4 grid grid-cols-2 gap-3 rounded-lg bg-muted/50 p-3.5 sm:grid-cols-3">
        <MetaItem label="创建时间">{formatDateTime(parseTimestamp(activeTask.created_at))}</MetaItem>
        <MetaItem label="开始时间">{activeTask.started_at ? formatDateTime(parseTimestamp(activeTask.started_at)) : '—'}</MetaItem>
        <MetaItem label="结束时间">{activeTask.completed_at ? formatDateTime(parseTimestamp(activeTask.completed_at)) : '—'}</MetaItem>
        <MetaItem label="耗时">{duration ?? '—'}</MetaItem>
        <MetaItem label="进度">{pct != null ? `${activeTask.progress}/${activeTask.total}（${pct}%）` : '—'}</MetaItem>
        <MetaItem label="已重试 / 上限">{activeTask.retry_count} / {activeTask.max_retries}</MetaItem>
      </div>

      {activeTask.error_message && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle />
          <AlertTitle>{taskErrorLabel(activeTask.error_message, activeTask.task_type)}</AlertTitle>
          {taskErrorLabel(activeTask.error_message, activeTask.task_type) !== activeTask.error_message && (
            <AlertDescription className="font-mono text-xs">{activeTask.error_message}</AlertDescription>
          )}
        </Alert>
      )}

      <div className="mt-5">
        <SectionTitle>执行阶段</SectionTitle>
        {stages.length > 0 ? (
          <div>
            {stages.map((s, i) => (
              <StageRow key={s.key} stage={s} last={i === stages.length - 1} errorMessage={activeTask.error_message} taskType={activeTask.task_type} />
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">{events.isLoading || workflow.isLoading ? '加载中…' : '该任务没有记录阶段'}</p>
        )}
      </div>

      {taskLevelEvents.length > 0 && (
        <div className="mt-4 border-t pt-3">
          <SectionTitle>任务事件</SectionTitle>
          <div className="max-h-40 overflow-auto">
            {taskLevelEvents.map(({ event, label }) => <EventRow key={event.id} event={{ ...event, message: label }} />)}
          </div>
        </div>
      )}

      {task.task_type === 'crawl_urls' && (crawlRecords.data?.records.length ?? 0) > 0 && (
        <div className="mt-4 border-t pt-3">
          <SectionTitle>采集记录（{crawlRecords.data!.records.length}）</SectionTitle>
          <Table>
            <TableHeader>
              <TableRow><TableHead>URL</TableHead><TableHead className="w-24">状态</TableHead><TableHead className="w-16">HTTP</TableHead></TableRow>
            </TableHeader>
            <TableBody>
              {crawlRecords.data!.records.map((r) => (
                <TableRow key={r.id}>
                  <TableCell className="max-w-0 truncate" title={r.url}>{r.url}</TableCell>
                  <TableCell>{stageLabel(r.status)}</TableCell>
                  <TableCell className="tabular-nums">{r.status_code ?? '—'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {earlier.length > 0 && (
        <div className="mt-4 border-t pt-3">
          <SectionTitle>更早的尝试（{earlier.length}）</SectionTitle>
          <div className="grid gap-0.5">
            {earlier.map((t) => (
              <EarlierAttemptRow key={t.id} task={t} active={viewedId === t.id} onSelect={() => setViewedId(t.id)} />
            ))}
          </div>
        </div>
      )}

      {(((task.status === 'running' || task.status === 'pending') && taskCanCancel(task)) || (task.status === 'failed' && taskCanRetry(task))) && (
        <div className="mt-5 flex gap-2">
          {(task.status === 'running' || task.status === 'pending') && taskCanCancel(task) && (
            <ConfirmButton variant="destructive" title="取消该任务？" confirmLabel="取消任务" loading={cancelTask.isPending}
              onConfirm={() => cancelTask.mutate(task.id, { onSuccess: close })}>
              取消任务
            </ConfirmButton>
          )}
          {task.status === 'failed' && taskCanRetry(task) && (
            <ConfirmButton title="重试该任务？" confirmLabel="重试" loading={retryTask.isPending} onConfirm={() => retryTask.mutate(task.id)}>
              重试
            </ConfirmButton>
          )}
        </div>
      )}
    </div>
  );
}

// 已完成 accumulates far more cards than are worth showing at once.
const COMPLETED_VISIBLE = 8;

function ColumnHeader({ column, count, etaSeconds }: { column: LifecycleColumn; count: number; etaSeconds?: number | null }) {
  const style = styleOf(COLUMN_STATUS[column]);
  const eta = column === 'running' ? formatDuration(etaSeconds) : null;
  return (
    <div className={cn('mb-3 flex items-center gap-2 border-b-2 px-0.5 pb-2', count > 0 ? style.line : 'border-border')}>
      <span className={cn('size-2 rounded-full', style.dot)} />
      <span className="text-sm font-semibold">{COLUMN_LABEL[column]}</span>
      {eta && <span className="truncate text-xs text-muted-foreground">入库全部完成约 {eta}</span>}
      <span className="ml-auto text-sm tabular-nums text-muted-foreground">{count}</span>
    </div>
  );
}

function LifecycleColumnView({ column, stacks, deepLinkKey, hideHeader = false, etaSeconds }: {
  column: LifecycleColumn;
  stacks: TaskStack<TaskBrief>[];
  deepLinkKey: string | null;
  hideHeader?: boolean;
  etaSeconds?: number | null;
}) {
  // A deep link into a completed stack past the truncation starts expanded.
  const [showAllCompleted, setShowAllCompleted] = useState(() => {
    if (column !== 'completed' || deepLinkKey == null) return false;
    const idx = sortStacksForColumn(column, stacks).findIndex((s) => s.key === deepLinkKey);
    return idx >= COMPLETED_VISIBLE;
  });
  const sorted = useMemo(
    () => sortStacksForColumn(column, stacks),
    [column, stacks],
  );
  const truncate = column === 'completed' && !showAllCompleted && sorted.length > COMPLETED_VISIBLE;
  const visible = truncate ? sorted.slice(0, COMPLETED_VISIBLE) : sorted;
  const hiddenCount = sorted.length - visible.length;

  return (
    <div className="min-w-0">
      {!hideHeader && <ColumnHeader column={column} count={stacks.length} etaSeconds={etaSeconds} />}
      {stacks.length === 0 ? (
        <div className="rounded-xl border border-dashed py-5 text-center text-xs text-muted-foreground">
          没有{COLUMN_LABEL[column]}的任务
        </div>
      ) : (
        <>
          <MorphGrid
            items={visible}
            getKey={(s) => s.key}
            initialExpandedKey={deepLinkKey}
            columns="minmax(0, 1fr)"
            renderCompact={(s) => <StackedTaskCard stack={s} />}
            renderExpanded={(s, close) => <TaskExpandedCard task={s.latest} close={close} earlier={s.earlier} />}
          />
          {hiddenCount > 0 && (
            <Button variant="ghost" size="sm" className="mt-2 w-full text-muted-foreground" onClick={() => setShowAllCompleted(true)}>
              显示更早的 {hiddenCount} 个
            </Button>
          )}
          {column === 'completed' && showAllCompleted && sorted.length > COMPLETED_VISIBLE && (
            <Button variant="ghost" size="sm" className="mt-2 w-full text-muted-foreground" onClick={() => setShowAllCompleted(false)}>
              收起
            </Button>
          )}
        </>
      )}
    </div>
  );
}

function MobileLifecycleTabs({ counts, selected, onSelect }: {
  counts: Record<LifecycleColumn, number>;
  selected: LifecycleColumn;
  onSelect: (column: LifecycleColumn) => void;
}) {
  return (
    <div className="grid grid-cols-4 gap-1.5">
      {LIFECYCLE_COLUMNS.map((column) => {
        const active = column === selected;
        const style = styleOf(COLUMN_STATUS[column]);
        return (
          <button
            key={column}
            type="button"
            onClick={() => onSelect(column)}
            className={cn('rounded-lg border bg-card px-1 py-1.5 text-center', active && 'border-primary bg-primary/10')}
          >
            <div className={cn('text-base font-bold tabular-nums', style.text)}>{counts[column]}</div>
            <div className="truncate text-[11px] text-muted-foreground">{COLUMN_LABEL[column]}</div>
          </button>
        );
      })}
    </div>
  );
}

export default function Tasks() {
  const [searchParams] = useSearchParams();
  const { data, isFetching } = useTaskBoard();
  const qc = useQueryClient();

  const deepLinkId = useMemo(
    () => selectedTaskIdFromSearch(searchParams.toString()),
    // Read once on mount — each column's MorphGrid owns expand/collapse after.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const tasks = useMemo(() => data?.tasks ?? [], [data]);
  const buckets = useMemo(() => bucketTasksByLifecycle(tasks), [tasks]);
  const stacksByColumn = useMemo(
    () => Object.fromEntries(LIFECYCLE_COLUMNS.map((c) => [c, groupTaskStacks(buckets[c])])) as Record<LifecycleColumn, TaskStack<TaskBrief>[]>,
    [buckets],
  );
  const deepLink = useMemo(() => {
    if (deepLinkId == null) return null;
    const task = tasks.find((t) => t.id === deepLinkId);
    if (!task) return null;
    const column = lifecycleColumnOf(task.status);
    const stackKey = stacksByColumn[column].find(
      (s) => s.latest.id === deepLinkId || s.earlier.some((e) => e.id === deepLinkId),
    )?.key ?? null;
    return stackKey ? { column, stackKey } : null;
  }, [deepLinkId, tasks, stacksByColumn]);

  const columnCounts = useMemo(
    () => Object.fromEntries(LIFECYCLE_COLUMNS.map((c) => [c, stacksByColumn[c].length])) as Record<LifecycleColumn, number>,
    [stacksByColumn],
  );

  const [viewportWidth, setViewportWidth] = useState(() => (typeof window !== 'undefined' ? window.innerWidth : 1440));
  useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  const narrow = isNarrowBoard(viewportWidth);
  const [selectedColumnOverride, setSelectedColumnOverride] = useState<LifecycleColumn | null>(() => deepLink?.column ?? null);
  const selectedColumn = selectedColumnOverride ?? defaultMobileColumn(columnCounts);

  return (
    <div className="grid gap-4">
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground">
          最近 {tasks.length} 个任务{data && data.total > tasks.length ? `，共 ${data.total} 个` : ''}
        </span>
        <Button variant="outline" size="sm" className="ml-auto" onClick={() => qc.invalidateQueries({ queryKey: ['tasks'] })}>
          <RefreshCw className={cn('size-3.5', isFetching && 'animate-spin')} />
          刷新
        </Button>
      </div>

      {tasks.length === 0 ? (
        <div className="py-16 text-center text-sm text-muted-foreground">{data ? '暂无任务' : '加载中…'}</div>
      ) : narrow ? (
        <div className="grid gap-3">
          <MobileLifecycleTabs counts={columnCounts} selected={selectedColumn} onSelect={setSelectedColumnOverride} />
          <LifecycleColumnView
            key={selectedColumn}
            column={selectedColumn}
            stacks={stacksByColumn[selectedColumn]}
            deepLinkKey={deepLink?.column === selectedColumn ? deepLink.stackKey : null}
            hideHeader
          />
        </div>
      ) : (
        <div className="grid items-start gap-4" style={{ gridTemplateColumns: LIFECYCLE_COLUMNS.map((c) => (columnCounts[c] > 0 ? 'minmax(0, 1fr)' : 'minmax(0, 0.55fr)')).join(' ') }}>
          {LIFECYCLE_COLUMNS.map((column) => (
            <LifecycleColumnView
              key={column}
              column={column}
              stacks={stacksByColumn[column]}
              deepLinkKey={deepLink?.column === column ? deepLink.stackKey : null}
              etaSeconds={data?.pipeline_eta_seconds}
            />
          ))}
        </div>
      )}
    </div>
  );
}
