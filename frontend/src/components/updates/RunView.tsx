import { useMemo, useState, type ReactNode } from 'react';
import { AlertCircle } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import Segmented from '../common/Segmented';
import { MetaItem, SectionTitle, StageRow, StatusBadge } from '../TaskParts';
import EventList from './EventList';
import type { TaskDetail, TaskEvent, WorkflowNode } from '../../types';
import type { ResourceSample, StreamConnection } from '../../api/hooks/taskStreamModel';
import { buildStageItems } from '../../pages/taskStageItems';
import { workflowNodeLabel, taskErrorLabel } from '../../pages/taskStageLabels';
import { instantMs, formatDateTime, formatDuration } from '../../pages/taskTime';
import { elapsedSeconds, formatCount, originLabel, resultCounts } from '../../pages/updatesModel';
import { useNow } from '../../hooks/useNow';

const CONNECTION_LABEL: Record<StreamConnection, { text: string; dot: string } | null> = {
  connecting: { text: '连接中', dot: 'bg-muted-foreground' },
  live: { text: '实时', dot: 'bg-success' },
  reconnecting: { text: '连接中断，重连中', dot: 'bg-warning' },
  error: { text: '连接失败', dot: 'bg-destructive' },
  done: null,
};

const GIB = 1024 ** 3;

/** One task's run: header figures, result counts, stage chain, events. Takes
 * plain data so a live stream and a stored replay render identically. */
export default function RunView({ task, nodes, events, dropped = 0, startedAtMs, connection, error, resource, actions }: {
  task: TaskDetail | null;
  nodes: WorkflowNode[];
  events: TaskEvent[];
  /** Older events pushed out of a full buffer. */
  dropped?: number;
  /** Epoch ms of the start, when the caller knows it more exactly than the task row. */
  startedAtMs?: number | null;
  connection?: StreamConnection;
  error?: string | null;
  resource?: ResourceSample | null;
  actions?: ReactNode;
}) {
  const running = task == null || task.status === 'running' || task.status === 'pending';
  const now = useNow(1000, running);
  const [stageFilter, setStageFilter] = useState<string>('all');

  const taskType = task?.task_type ?? 'ingest_pipeline';
  const stages = useMemo(
    // Stage events are listed in the event list below (filterable by stage); only a
    // task without a stage chain needs them to derive its stages.
    () => buildStageItems({ task_type: taskType, status: task?.status ?? 'running' }, nodes, nodes.length > 0 ? [] : events),
    [taskType, task?.status, nodes, events],
  );
  const stageOptions = useMemo(
    () => [
      { value: 'all', label: '全部' },
      ...nodes.map((n) => ({ value: String(n.id), label: workflowNodeLabel(n.subtask_key ?? n.key, taskType, n.name) })),
    ],
    [nodes, taskType],
  );
  const filtered = useMemo(
    () => (stageFilter === 'all' ? events : events.filter((e) => String(e.subtask_id) === stageFilter)),
    [events, stageFilter],
  );

  const done = nodes.filter((n) => n.status === 'completed' || n.status === 'skipped').length;
  const startMs = startedAtMs ?? (task?.started_at ? instantMs(task.started_at) : null);
  const duration = formatDuration(elapsedSeconds(task, startMs, now));
  const counts = resultCounts(task?.result);
  const origin = originLabel(task?.params?.origin);
  const conn = connection ? CONNECTION_LABEL[connection] : null;

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-2 gap-3 rounded-lg bg-muted/50 p-3.5 sm:grid-cols-4">
        <MetaItem label="状态">
          {task ? <StatusBadge status={task.status} /> : '—'}
        </MetaItem>
        <MetaItem label="开始时间">{startMs != null ? formatDateTime(new Date(startMs)) : '—'}</MetaItem>
        <MetaItem label="用时">{duration ?? '—'}</MetaItem>
        <MetaItem label="阶段">{nodes.length > 0 ? `${done}/${nodes.length}` : '—'}</MetaItem>
        {origin && <MetaItem label="触发">{origin}</MetaItem>}
        {task && <MetaItem label="任务">#{task.id}</MetaItem>}
        {resource?.cpu_percent != null && <MetaItem label="CPU">{resource.cpu_percent.toFixed(0)} %</MetaItem>}
        {resource?.rss_bytes != null && <MetaItem label="内存">{(resource.rss_bytes / GIB).toFixed(1)} GiB</MetaItem>}
        {conn && (
          <MetaItem label="连接">
            <span className="inline-flex items-center gap-1.5"><span className={cn('size-2 rounded-full', conn.dot)} />{conn.text}</span>
          </MetaItem>
        )}
      </div>

      {error && (
        <Alert variant="destructive"><AlertCircle /><AlertTitle>{error}</AlertTitle></Alert>
      )}
      {task?.error_message && (
        <Alert variant="destructive">
          <AlertCircle />
          <AlertTitle>{taskErrorLabel(task.error_message, taskType)}</AlertTitle>
          {taskErrorLabel(task.error_message, taskType) !== task.error_message && (
            <AlertDescription className="font-mono text-xs">{task.error_message}</AlertDescription>
          )}
        </Alert>
      )}

      {counts.length > 0 && (
        <div>
          <SectionTitle>本轮结果</SectionTitle>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {counts.map((c) => <MetaItem key={c.key} label={c.label}>{formatCount(c.value)} 篇</MetaItem>)}
          </div>
        </div>
      )}

      <div>
        <SectionTitle>执行阶段</SectionTitle>
        {stages.length > 0 ? (
          stages.map((s, i) => (
            <StageRow key={s.key} stage={s} last={i === stages.length - 1} errorMessage={task?.error_message} taskType={taskType} />
          ))
        ) : (
          <p className="text-xs text-muted-foreground">{task ? '该任务没有记录阶段' : '加载中…'}</p>
        )}
      </div>

      <div>
        <div className="mb-2 flex flex-wrap items-center gap-3">
          <h4 className="text-sm font-semibold">事件</h4>
          <span className="text-xs tabular-nums text-muted-foreground">
            {formatCount(filtered.length)} 条
          </span>
          {dropped > 0 && <span className="text-xs tabular-nums text-muted-foreground">更早 {formatCount(dropped)} 条已丢弃</span>}
          {nodes.length > 0 && (
            <Segmented className="ml-auto" ariaLabel="按阶段筛选事件" value={stageFilter} options={stageOptions} onChange={setStageFilter} />
          )}
        </div>
        <EventList events={filtered} />
      </div>

      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}
