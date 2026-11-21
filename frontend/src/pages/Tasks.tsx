import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'framer-motion';
import axios from 'axios';
import { AlertCircle, Check, Circle, Loader2, RefreshCw, X } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import ConfirmButton from '../components/ConfirmButton';
import {
  useTasks, useTaskEvents, useTaskWorkflow,
  useCancelTask, useRetryTask,
} from '../api/hooks/useTasks';
import { useCrawlRecords } from '../api/hooks/useCrawl';
import { useDocumentDetail } from '../api/hooks/useDocuments';
import { usePaperDetail } from '../api/hooks/usePapers';
import type { TaskBrief, TaskEvent } from '../types';
import { selectedTaskIdFromSearch } from './taskSelection';
import { taskTypeLabel, taskSubject, taskDocId } from './taskBody';
import {
  parseCreatedAt, parseStartedAt, relativeTimeFrom, formatDuration, formatDateTime,
} from './taskTime';
import { groupEventsBySubtask, buildSyntheticStageNodes, reachedStageLabel } from './taskFlow';
import {
  stageLabel, workflowNodeLabel, taskLevelEventLabel, taskStatusLabel, taskErrorLabel,
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

// status -> [text colour, bar colour, tinted badge]
const STATUS_STYLE: Record<string, { text: string; bar: string; badge: string; dot: string; line: string }> = {
  running: { text: 'text-primary', bar: 'bg-primary', badge: 'border-primary/40 bg-primary/10 text-primary', dot: 'bg-primary', line: 'border-primary' },
  pending: { text: 'text-muted-foreground', bar: 'bg-muted-foreground', badge: 'border-border bg-muted text-muted-foreground', dot: 'bg-muted-foreground', line: 'border-muted-foreground' },
  completed: { text: 'text-success', bar: 'bg-success', badge: 'border-success/40 bg-success/10 text-success', dot: 'bg-success', line: 'border-success' },
  failed: { text: 'text-destructive', bar: 'bg-destructive', badge: 'border-destructive/40 bg-destructive/10 text-destructive', dot: 'bg-destructive', line: 'border-destructive' },
  timeout: { text: 'text-destructive', bar: 'bg-destructive', badge: 'border-destructive/40 bg-destructive/10 text-destructive', dot: 'bg-destructive', line: 'border-destructive' },
  cancelled: { text: 'text-muted-foreground', bar: 'bg-muted-foreground/60', badge: 'border-border bg-muted text-muted-foreground', dot: 'bg-muted-foreground/60', line: 'border-muted-foreground' },
};
const styleOf = (status: string) => STATUS_STYLE[status] ?? STATUS_STYLE.pending;

const COLUMN_LABEL: Record<LifecycleColumn, string> = {
  pending: '排队中', running: '运行中', completed: '已完成', failed: '失败 / 已取消',
};
const COLUMN_STATUS: Record<LifecycleColumn, string> = {
  pending: 'pending', running: 'running', completed: 'completed', failed: 'failed',
};

function StatusBadge({ status }: { status: string }) {
  return (
    <Badge variant="outline" className={cn('shrink-0 font-semibold', styleOf(status).badge)}>
      {taskStatusLabel(status)}
    </Badge>
  );
}

function Bar({ pct, className }: { pct: number; className?: string }) {
  return (
    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
      <div className={cn('h-full rounded-full', className)} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
    </div>
  );
}

function taskPercent(task: TaskBrief): number | null {
  return task.total > 0 ? Math.round((task.progress / task.total) * 100) : null;
}

function taskDurationSeconds(task: TaskBrief): number | null {
  if (task.duration_seconds != null) return task.duration_seconds;
  if (task.started_at && task.completed_at) {
    return (parseStartedAt(task.completed_at).getTime() - parseStartedAt(task.started_at).getTime()) / 1000;
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
        <span className="min-w-0 flex-1 truncate text-sm font-semibold">{taskTypeLabel(task.task_type)}</span>
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
        <span>创建于 {relativeTimeFrom(parseCreatedAt(task.created_at))}{duration && <> · 耗时 {duration}</>}</span>
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

function EventRow({ event }: { event: TaskEvent }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
      className="grid grid-cols-[64px_1fr] gap-2 py-0.5 text-xs"
    >
      <span className="tabular-nums text-muted-foreground">{event.created_at?.slice(11, 19)}</span>
      <span className={cn('break-words', event.level === 'error' ? 'text-destructive' : event.level === 'warning' ? 'text-warning' : '')}>
        {event.message}
      </span>
    </motion.div>
  );
}

interface StageItem {
  key: string;
  label: string;
  status: string;
  hollow?: boolean;
  events: TaskEvent[];
  progress?: number;
  total?: number;
}

function StageIcon({ status, hollow }: { status: string; hollow?: boolean }) {
  const base = 'grid size-6 shrink-0 place-items-center rounded-full border-2';
  if (hollow || status === 'pending') return <span className={cn(base, 'border-border text-muted-foreground')}><Circle className="size-2" /></span>;
  if (status === 'running') return <span className={cn(base, 'border-primary text-primary')}><Loader2 className="size-3.5 animate-spin" /></span>;
  if (status === 'failed' || status === 'timeout') return <span className={cn(base, 'border-destructive bg-destructive text-white')}><AlertCircle className="size-3.5" /></span>;
  if (status === 'cancelled') return <span className={cn(base, 'border-muted-foreground/60 text-muted-foreground')}><X className="size-3.5" /></span>;
  return <span className={cn(base, 'border-success bg-success text-white')}><Check className="size-3.5" /></span>;
}

function StageRow({ stage, last, errorMessage, taskType }: { stage: StageItem; last: boolean; errorMessage?: string; taskType: string }) {
  const [open, setOpen] = useState(false);
  const pct = stage.total && stage.total > 0 ? Math.round(((stage.progress ?? 0) / stage.total) * 100) : null;
  const doneLine = !stage.hollow && (stage.status === 'completed' || stage.status === 'skipped');
  return (
    <div className="grid grid-cols-[24px_1fr] gap-x-3">
      <div className="flex flex-col items-center">
        <StageIcon status={stage.status} hollow={stage.hollow} />
        {!last && <div className={cn('my-1 w-0.5 flex-1 rounded-full', doneLine ? 'bg-success/60' : 'bg-border')} />}
      </div>
      <div className={cn('min-w-0', last ? 'pb-1' : 'pb-4')}>
        <div className="flex items-baseline gap-2">
          <span className={cn('text-sm font-medium', stage.hollow && 'text-muted-foreground')}>{stage.label}</span>
          <span className={cn('text-xs', stage.hollow ? 'text-muted-foreground' : styleOf(stage.status).text)}>
            {stage.hollow ? '未执行' : taskStatusLabel(stage.status)}
          </span>
        </div>
        {pct != null && (
          <div className="mt-1.5 flex max-w-sm items-center gap-2">
            <Bar pct={pct} className={styleOf(stage.status).bar} />
            <span className="text-[11px] tabular-nums text-muted-foreground">{stage.progress}/{stage.total}</span>
          </div>
        )}
        {(stage.status === 'failed' || stage.status === 'timeout') && errorMessage && (
          <div className="mt-1 text-xs text-destructive">{taskErrorLabel(errorMessage, taskType)}</div>
        )}
        {stage.events.length > 0 && (
          <>
            <button type="button" className="mt-1 text-xs text-primary hover:underline" onClick={() => setOpen((v) => !v)}>
              {open ? '收起事件' : `查看 ${stage.events.length} 条事件`}
            </button>
            {open && (
              <div className="mt-1 max-h-52 overflow-auto rounded-md bg-muted/50 px-2 py-1">
                <AnimatePresence initial={false}>
                  {stage.events.map((e) => <EventRow key={e.id} event={e} />)}
                </AnimatePresence>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function useStageItems(
  task: TaskBrief,
  nodes: { id: number; name: string; subtask_key: string | null; status: string; progress: number; total: number }[],
  events: TaskEvent[],
): StageItem[] {
  return useMemo<StageItem[]>(() => {
    if (nodes.length > 0) {
      const bySubtask = groupEventsBySubtask(events);
      return nodes.map((n) => ({
        key: String(n.id),
        label: workflowNodeLabel(n.subtask_key, task.task_type, n.name),
        status: n.status,
        events: bySubtask.get(n.id) ?? [],
        progress: n.progress,
        total: n.total,
      }));
    }
    return buildSyntheticStageNodes(events, task.status, task.task_type);
  }, [nodes, events, task.task_type, task.status]);
}

function EarlierAttemptRow({ task, active, onSelect }: { task: TaskBrief; active: boolean; onSelect: () => void }) {
  const workflow = useTaskWorkflow(task.id);
  const events = useTaskEvents(task.id, false);
  const reached = reachedStageLabel(
    (workflow.data?.nodes ?? []).map((n) => ({ name: n.name, subtask_key: n.subtask_key, status: n.status })),
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
      <span className="tabular-nums text-muted-foreground">{formatDateTime(parseCreatedAt(task.created_at))}</span>
    </button>
  );
}

function MetaItem({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="text-sm tabular-nums">{children}</div>
    </div>
  );
}

function SectionTitle({ children }: { children: ReactNode }) {
  return <h4 className="mb-2 text-sm font-semibold">{children}</h4>;
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
  const stages = useStageItems(activeTask, nodes, eventList);
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
        <h3 className="text-lg font-semibold">{taskTypeLabel(activeTask.task_type)}</h3>
        <StatusBadge status={activeTask.status} />
        {viewedId != null && (
          <button type="button" className="ml-auto text-xs text-primary hover:underline" onClick={() => setViewedId(null)}>
            返回最新一次
          </button>
        )}
      </div>
      <p className={cn('mt-0.5 text-sm text-muted-foreground', subject.muted && 'italic')}>{subject.text}</p>

      <div className="mt-4 grid grid-cols-2 gap-3 rounded-lg bg-muted/50 p-3.5 sm:grid-cols-3">
        <MetaItem label="创建时间">{formatDateTime(parseCreatedAt(activeTask.created_at))}</MetaItem>
        <MetaItem label="开始时间">{activeTask.started_at ? formatDateTime(parseStartedAt(activeTask.started_at)) : '—'}</MetaItem>
        <MetaItem label="结束时间">{activeTask.completed_at ? formatDateTime(parseStartedAt(activeTask.completed_at)) : '—'}</MetaItem>
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

      {(task.status === 'running' || task.status === 'pending' || task.status === 'failed') && (
        <div className="mt-5 flex gap-2">
          {(task.status === 'running' || task.status === 'pending') && (
            <ConfirmButton variant="destructive" title="取消该任务？" confirmLabel="取消任务" loading={cancelTask.isPending}
              onConfirm={() => cancelTask.mutate(task.id, { onSuccess: close })}>
              取消任务
            </ConfirmButton>
          )}
          {task.status === 'failed' && (
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
  // The backend caps page_size at 100 (app/api/routes/tasks.py); more 422s.
  const { data, isFetching } = useTasks(1, 100);
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
