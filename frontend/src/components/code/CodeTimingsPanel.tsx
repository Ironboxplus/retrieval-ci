/**
 * Code mode's per-stage timing bar. One proportional bar (the private and
 * public legs can overlap in wall-clock time, so a step chain would lie);
 * every segment is named with its milliseconds, and wall time outside the
 * measured stages is its own named segment instead of an unexplained gap.
 */
import { cn } from '@/lib/utils';
import { timingsSegments, type CodeSearchTimingsMsLike } from '../../pages/codeSearchModel';

const SEGMENT_COLOR: Record<string, string> = {
  embed: '#4F46E5', retrieve: '#0891b2', fuse: '#7c3aed', rerank: '#f59e0b',
};

export default function CodeTimingsPanel({ timings, searching }: { timings?: CodeSearchTimingsMsLike; searching: boolean }) {
  if (!timings && !searching) return null;
  const segments = timings ? timingsSegments(timings) : [];
  const total = timings?.total ?? null;
  const measured = segments.reduce((s, x) => s + x.ms, 0);
  const other = total != null ? Math.max(0, total - measured) : 0;
  const otherPct = total ? (other / total) * 100 : 0;

  return (
    <section className="rounded-xl border bg-card p-4">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">检索耗时</h3>
        {total != null && (
          <span className="text-xs text-muted-foreground">
            总计 <span className="text-sm font-semibold tabular-nums text-foreground">{Math.round(total)} ms</span>
          </span>
        )}
      </div>
      {searching && !timings ? (
        <div className="h-2.5 overflow-hidden rounded-full bg-muted">
          <div className="h-full w-2/5 animate-pulse rounded-full bg-primary/50" />
        </div>
      ) : (
        <>
          <div className="flex h-2.5 overflow-hidden rounded-full bg-muted">
            {segments.map((s) => (
              <div key={s.key} style={{ width: `${s.widthPct}%`, minWidth: 2, background: SEGMENT_COLOR[s.key] }} />
            ))}
            {otherPct > 0.5 && <div className="bg-muted-foreground/30" style={{ width: `${otherPct}%` }} />}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
            {segments.map((s) => (
              <span key={s.key} className="flex items-center gap-1.5">
                <span className="size-2 rounded-sm" style={{ background: SEGMENT_COLOR[s.key] }} />
                {s.label}
                <span className="tabular-nums text-muted-foreground">{Math.round(s.ms)} ms</span>
              </span>
            ))}
            {otherPct > 0.5 && (
              <span className="flex items-center gap-1.5">
                <span className={cn('size-2 rounded-sm bg-muted-foreground/30')} />
                其他（范围判定、网络）
                <span className="tabular-nums text-muted-foreground">{Math.round(other)} ms</span>
              </span>
            )}
            {segments.length === 0 && <span className="text-muted-foreground">没有返回阶段耗时</span>}
          </div>
        </>
      )}
    </section>
  );
}
