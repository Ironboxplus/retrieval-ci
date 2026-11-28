/**
 * CodeJobBoard — code mode's index-job queue: a MorphGrid board over the
 * caller's own /code/jobs, the same shape as Tasks.tsx. Only queued/running
 * jobs show (finished ones are already reflected in CodeScopePanel's tree);
 * the board is absent when nothing is active. Narrow screens collapse to a
 * single list of rows that open the same detail in a bottom sheet.
 *
 * Cancel's 409 code_index_committing is an expected outcome (the write
 * already started between click and response) — it flips the button to a
 * disabled "正在提交，无法取消" state. Any other cancel failure (404 stale
 * job, network) surfaces as a toast.
 */
import { useState } from 'react';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet';
import { useCodeIndexJobs, useCancelCodeIndexJob, type CodeIndexJob } from '../../api/hooks/useCodeSearch';
import { useIsWide } from '../../hooks/useIsWide';
import MorphGrid from '../morph/MorphGrid';
import MorphCard from '../morph/MorphCard';

const STATUS_STYLE: Record<CodeIndexJob['status'], { label: string; dot: string; bar: string }> = {
  queued: { label: '排队中', dot: 'bg-slate-400', bar: 'bg-slate-400' },
  running: { label: '索引中', dot: 'bg-primary', bar: 'bg-primary' },
  completed: { label: '已完成', dot: 'bg-success', bar: 'bg-success' },
  failed: { label: '失败', dot: 'bg-destructive', bar: 'bg-destructive' },
  cancelled: { label: '已取消', dot: 'bg-zinc-400', bar: 'bg-zinc-400' },
};

function jobProgressLabel(job: CodeIndexJob): string {
  if (job.chunks_total > 0) {
    const chunks = `分块 ${job.chunks_done.toLocaleString()}/${job.chunks_total.toLocaleString()}`;
    return job.files_total > 0 ? `${chunks} · 文件 ${job.files_total.toLocaleString()}` : chunks;
  }
  if (job.removed_total > 0) return `删除 ${job.removed_total} 个路径`;
  return job.queue_position > 0 ? `队列第 ${job.queue_position} 位` : '等待开始';
}

function JobProgress({ job, className }: { job: CodeIndexJob; className?: string }) {
  const s = STATUS_STYLE[job.status];
  const pct = Math.round(job.progress * 100);
  return (
    <div className={className}>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="truncate text-muted-foreground">{jobProgressLabel(job)}</span>
        <span className="font-semibold tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div className={cn('h-full rounded-full transition-[width]', s.bar)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function StatusLabel({ job }: { job: CodeIndexJob }) {
  const s = STATUS_STYLE[job.status];
  return (
    <span className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
      <span className={cn('size-2 rounded-full', s.dot, job.status === 'running' && 'animate-pulse')} />
      {s.label}
    </span>
  );
}

function CompactJobCard({ job }: { job: CodeIndexJob }) {
  return (
    <MorphCard>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold" title={job.project}>{job.project}</div>
          <div className="truncate text-xs text-muted-foreground">{job.host}</div>
        </div>
        <StatusLabel job={job} />
      </div>
      <JobProgress job={job} className="mt-3" />
    </MorphCard>
  );
}

function StripRow({ job, onOpen }: { job: CodeIndexJob; onOpen: () => void }) {
  return (
    <button type="button" onClick={onOpen} className="block w-full py-2.5 text-left">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="truncate text-sm font-semibold">{job.project}</span>
        <StatusLabel job={job} />
      </div>
      <JobProgress job={job} />
    </button>
  );
}

function JobDetail({
  job, committing, onCommitting,
}: {
  job: CodeIndexJob;
  committing: boolean;
  onCommitting: () => void;
}) {
  const cancel = useCancelCodeIndexJob();
  const cancellable = (job.status === 'queued' || job.status === 'running') && !committing;

  return (
    <div>
      <div className="text-base font-semibold">{job.project}</div>
      <div className="text-xs text-muted-foreground">主机 {job.host}</div>

      <div className="mt-4 flex items-center gap-2">
        <StatusLabel job={job} />
        {committing && <Badge variant="outline" className="border-warning/40 text-warning">正在提交，无法取消</Badge>}
      </div>

      <JobProgress job={job} className="mt-3" />

      <div className="mt-5">
        <Button
          variant="destructive"
          disabled={!cancellable || cancel.isPending}
          onClick={() => cancel.mutate(job.job_id, {
            onError: (err) => {
              if (err.code === 'code_index_committing') { onCommitting(); return; }
              toast.error(err.status === 404 ? '任务不存在或已结束' : '取消失败');
            },
          })}
        >
          {committing ? '正在提交，无法取消' : cancel.isPending ? '取消中…' : '取消索引'}
        </Button>
      </div>
    </div>
  );
}

export default function CodeJobBoard() {
  const { data: jobs = [] } = useCodeIndexJobs();
  const active = jobs.filter((j) => j.status === 'queued' || j.status === 'running');
  const isWide = useIsWide();
  // Sticky per-job "saw a 409 committing" flag — once a cancel lands in the
  // race window the job WILL finish; the button stays disabled rather than
  // inviting a retry that will just 409 again.
  const [committingIds, setCommittingIds] = useState<Set<string>>(new Set());
  const [stripOpenId, setStripOpenId] = useState<string | null>(null);

  if (active.length === 0) return null;

  const markCommitting = (id: string) => setCommittingIds((prev) => new Set(prev).add(id));

  if (!isWide) {
    const openJob = active.find((j) => j.job_id === stripOpenId) ?? null;
    return (
      <div className="divide-y rounded-xl border bg-card px-3">
        {active.map((j) => (
          <StripRow key={j.job_id} job={j} onOpen={() => setStripOpenId(j.job_id)} />
        ))}
        <Sheet open={!!openJob} onOpenChange={(o) => !o && setStripOpenId(null)}>
          <SheetContent side="bottom" className="p-5">
            <SheetTitle className="sr-only">索引任务</SheetTitle>
            {openJob && (
              <JobDetail job={openJob} committing={committingIds.has(openJob.job_id)} onCommitting={() => markCommitting(openJob.job_id)} />
            )}
          </SheetContent>
        </Sheet>
      </div>
    );
  }

  return (
    <MorphGrid
      items={active}
      getKey={(j) => j.job_id}
      columns="repeat(auto-fill, minmax(240px, 1fr))"
      renderCompact={(j) => <CompactJobCard job={j} />}
      renderExpanded={(j) => (
        <div className="p-5">
          <JobDetail job={j} committing={committingIds.has(j.job_id)} onCommitting={() => markCommitting(j.job_id)} />
        </div>
      )}
    />
  );
}
