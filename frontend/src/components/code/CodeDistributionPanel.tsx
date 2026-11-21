/**
 * CodeDistributionPanel — code mode's hit distribution, parity with paper
 * mode's venue/year bars: segmented bars from codeSearchModel.ts's
 * originDistribution / scopeDistribution, each with a colour-keyed legend
 * carrying label and count. Clicking a scope segment or its legend chip
 * toggles a filter routed by the segment's origin: private → projects[],
 * public → source_ids[] (different request fields, so the bar keeps them
 * apart). The origin bar only renders when both origins are present.
 */
import { cn } from '@/lib/utils';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { originDistribution, scopeDistribution, type FederatedHitLike } from '../../pages/codeSearchModel';

const ORIGIN_COLOR: Record<string, string> = { private: '#4F46E5', public: '#0891b2' };
const SCOPE_PALETTE = ['#4F46E5', '#0891b2', '#7c3aed', '#f59e0b', '#10b981', '#ef4444', '#a855f7', '#f43f5e'];
const LEGEND_MAX = 8;

interface Seg { key: string; label: string; count: number; pct: number; color: string; selected?: boolean; onClick?: () => void }

function SegmentBar({ title, total, segs }: { title: string; total: number; segs: Seg[] }) {
  const hidden = segs.slice(LEGEND_MAX);
  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex items-baseline justify-between text-xs">
        <span className="font-medium text-muted-foreground">{title}</span>
        <span className="tabular-nums text-muted-foreground">共 {total} 条</span>
      </div>
      <div className="flex h-5 overflow-hidden rounded-md bg-muted">
        {segs.map((s) => (
          <Tooltip key={s.key}>
            <TooltipTrigger asChild>
              <div
                onClick={s.onClick}
                className={cn(s.onClick && 'click-filter-segment cursor-pointer', s.selected && 'ring-2 ring-inset ring-foreground')}
                style={{ width: `${s.pct}%`, minWidth: s.count > 0 ? 3 : 0, background: s.color }}
              />
            </TooltipTrigger>
            <TooltipContent>{s.label}：{s.count} 条（{Math.round(s.pct)}%）</TooltipContent>
          </Tooltip>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {segs.slice(0, LEGEND_MAX).map((s) => {
          const chip = cn(
            'flex max-w-[220px] items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs',
            s.selected ? 'border-foreground/40 bg-accent font-medium' : 'border-transparent bg-muted/60',
          );
          const body = (
            <>
              <span className="size-2 shrink-0 rounded-sm" style={{ background: s.color }} />
              <span className="truncate">{s.label}</span>
              <span className="tabular-nums text-muted-foreground">{s.count}</span>
            </>
          );
          return s.onClick ? (
            <button key={s.key} type="button" title={s.label} onClick={s.onClick} className={cn(chip, 'hover:bg-accent')}>{body}</button>
          ) : (
            <span key={s.key} title={s.label} className={chip}>{body}</span>
          );
        })}
        {hidden.length > 0 && (
          <span className="px-1 py-0.5 text-xs text-muted-foreground">
            其他 {hidden.length} 项 · {hidden.reduce((n, s) => n + s.count, 0)} 条
          </span>
        )}
      </div>
    </div>
  );
}

export default function CodeDistributionPanel({
  hits, selectedProjects, selectedSourceIds, onToggleProject, onToggleSource,
}: {
  hits: FederatedHitLike[];
  selectedProjects: string[];
  selectedSourceIds: string[];
  onToggleProject: (project: string) => void;
  onToggleSource: (sourceId: string) => void;
}) {
  const origin = originDistribution(hits);
  const scope = scopeDistribution(hits);
  if (origin.length === 0 && scope.length === 0) return null;
  const showOrigin = origin.length > 1;

  const originSegs: Seg[] = origin.map((d) => ({ key: d.key, label: d.label, count: d.count, pct: d.pct, color: ORIGIN_COLOR[d.key] }));
  const scopeSegs: Seg[] = scope.map((d, i) => {
    const pub = d.origin === 'public';
    return {
      key: `${d.origin}:${d.key}`,
      label: d.label,
      count: d.count,
      pct: d.pct,
      color: SCOPE_PALETTE[i % SCOPE_PALETTE.length],
      selected: pub ? selectedSourceIds.includes(d.key) : selectedProjects.includes(d.key),
      onClick: () => (pub ? onToggleSource(d.key) : onToggleProject(d.key)),
    };
  });

  return (
    <section className={cn('grid gap-6 rounded-xl border bg-card px-4 py-3', showOrigin && 'md:grid-cols-[2fr_3fr]')}>
      {showOrigin && <SegmentBar title="来源分布" total={hits.length} segs={originSegs} />}
      <SegmentBar title="按项目 / 公共源" total={hits.length} segs={scopeSegs} />
    </section>
  );
}
