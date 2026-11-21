import { useState } from 'react';
import { Pencil, Play } from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import ConfirmButton from '../ConfirmButton';
import { MetaItem, SectionTitle, StatusBadge } from '../TaskParts';
import { LiveRun } from './RunPanels';
import ScheduleEditDialog from './ScheduleEditDialog';
import TickHistory from './TickHistory';
import {
  apiErrorText, usePatchSchedule, useRunSchedule,
  type Schedule, type ScheduleKind,
} from '../../api/hooks/useSchedules';
import {
  formatInZone, formatMetric, formatTrigger, relativeFromNow, tickRow,
} from '../../pages/updatesModel';
import { paramRows } from '../../pages/updatesParams';

function When({ iso, timezone }: { iso: string | null | undefined; timezone: string }) {
  if (!iso) return <>—</>;
  return (
    <>
      {relativeFromNow(iso)}
      <span className="ml-2 text-xs text-muted-foreground">{formatInZone(iso, timezone)}</span>
    </>
  );
}

export default function ScheduleCard({ schedule, kind }: { schedule: Schedule; kind: ScheduleKind | undefined }) {
  const patch = usePatchSchedule();
  const run = useRunSchedule();
  const [editing, setEditing] = useState(false);

  const timezone = schedule.trigger_json.timezone;
  const last = schedule.last_tick ? tickRow(schedule.last_tick) : null;
  const upcoming = schedule.enabled ? (schedule.next_run_time ?? schedule.next_fire_times[0]) : null;
  const params = paramRows(schedule.kind, schedule.params_json);
  const running = schedule.running_task;

  const toggle = (enabled: boolean) =>
    patch.mutate({ id: schedule.id, patch: { enabled } }, { onError: (e) => toast.error(apiErrorText(e)) });

  const runNow = () =>
    run.mutate(schedule.id, {
      onSuccess: (r) => (r.status === 'enqueued' ? toast.success(`已启动任务 #${r.task_id}`) : toast.info(r.reason ?? r.status)),
      onError: (e) => toast.error(apiErrorText(e)),
    });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          {schedule.title}
          {schedule.system && <Badge variant="outline" className="font-normal text-muted-foreground">系统</Badge>}
        </CardTitle>
        <CardAction className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">{schedule.enabled ? '已启用' : '已停用'}</span>
          <Switch checked={schedule.enabled} disabled={patch.isPending} onCheckedChange={toggle} aria-label="启用" />
        </CardAction>
      </CardHeader>

      <CardContent className="grid gap-5">
        <div className="grid grid-cols-2 gap-3 rounded-lg bg-muted/50 p-3.5 sm:grid-cols-4">
          <MetaItem label="触发">{formatTrigger(schedule.trigger_json)}</MetaItem>
          <MetaItem label="下次运行"><When iso={upcoming} timezone={timezone} /></MetaItem>
          <MetaItem label="上次运行">
            {last?.time ? (
              <span className="inline-flex flex-wrap items-center gap-x-2">
                <When iso={last.time} timezone={timezone} />
                {last.taskStatus ? <StatusBadge status={last.taskStatus} /> : <span className="text-xs text-muted-foreground">{last.tickLabel}</span>}
              </span>
            ) : '—'}
          </MetaItem>
          {params.map((p) => <MetaItem key={p.name} label={p.name}>{p.value}</MetaItem>)}
          {schedule.metrics.map((m) => <MetaItem key={m.name} label={m.name}>{formatMetric(m)}</MetaItem>)}
        </div>

        <div className="flex gap-2">
          <ConfirmButton title={`立即运行「${schedule.title}」？`} confirmLabel="立即运行" variant="outline" loading={run.isPending} onConfirm={runNow}>
            <Play className="size-3.5" />立即运行
          </ConfirmButton>
          <Button variant="outline" onClick={() => setEditing(true)}><Pencil className="size-3.5" />编辑</Button>
        </div>

        {running && (
          <div>
            <SectionTitle>当前运行</SectionTitle>
            <LiveRun key={running.id} taskId={running.id} startedAtMs={running.started_at != null ? Date.parse(running.started_at) : null} />
          </div>
        )}

        <TickHistory scheduleId={schedule.id} kind={schedule.kind} timezone={timezone} />
      </CardContent>

      {editing && <ScheduleEditDialog schedule={schedule} kind={kind} onClose={() => setEditing(false)} />}
    </Card>
  );
}
