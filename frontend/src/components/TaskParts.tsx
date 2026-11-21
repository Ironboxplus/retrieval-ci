import { useState, type ReactNode } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, Check, Circle, Loader2, X } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import type { TaskEvent } from '../types';
import { taskErrorLabel, taskStatusLabel } from '../pages/taskStageLabels';
import { styleOf } from '../pages/taskStatusStyle';
import { formatClock, parseTimestamp } from '../pages/taskTime';
import type { StageItem } from '../pages/taskStageItems';

// Building blocks shared by the task board (pages/Tasks.tsx) and the
// scheduled-update run panels (pages/Updates.tsx).

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge variant="outline" className={cn('shrink-0 font-semibold', styleOf(status).badge)}>
      {taskStatusLabel(status)}
    </Badge>
  );
}

export function Bar({ pct, className }: { pct: number; className?: string }) {
  return (
    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
      <div className={cn('h-full rounded-full', className)} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
    </div>
  );
}

/** `animated={false}` for rows that mount and unmount while scrolling (virtual lists). */
export function EventRow({ event, animated = true }: { event: TaskEvent; animated?: boolean }) {
  return (
    <motion.div
      initial={animated ? { opacity: 0, x: -6 } : false} animate={{ opacity: 1, x: 0 }}
      className="grid grid-cols-[64px_1fr] gap-2 py-0.5 text-xs"
    >
      <span className="tabular-nums text-muted-foreground">{event.created_at ? formatClock(parseTimestamp(event.created_at)) : ''}</span>
      <span className={cn('break-words', event.level === 'error' ? 'text-destructive' : event.level === 'warning' ? 'text-warning' : '')}>
        {event.message}
      </span>
    </motion.div>
  );
}

function StageIcon({ status, hollow }: { status: string; hollow?: boolean }) {
  const base = 'grid size-6 shrink-0 place-items-center rounded-full border-2';
  if (hollow || status === 'pending') return <span className={cn(base, 'border-border text-muted-foreground')}><Circle className="size-2" /></span>;
  if (status === 'running') return <span className={cn(base, 'border-primary text-primary')}><Loader2 className="size-3.5 animate-spin" /></span>;
  if (status === 'failed' || status === 'timeout') return <span className={cn(base, 'border-destructive bg-destructive text-white')}><AlertCircle className="size-3.5" /></span>;
  if (status === 'cancelled') return <span className={cn(base, 'border-muted-foreground/60 text-muted-foreground')}><X className="size-3.5" /></span>;
  return <span className={cn(base, 'border-success bg-success text-white')}><Check className="size-3.5" /></span>;
}

export function StageRow({ stage, last, errorMessage, taskType }: { stage: StageItem; last: boolean; errorMessage?: string; taskType: string }) {
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
                {/* newest first: opening the list shows where the stage is now */}
                <AnimatePresence initial={false}>
                  {[...stage.events].reverse().map((e) => <EventRow key={e.id} event={e} />)}
                </AnimatePresence>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function MetaItem({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="text-sm tabular-nums">{children}</div>
    </div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return <h4 className="mb-2 text-sm font-semibold">{children}</h4>;
}
