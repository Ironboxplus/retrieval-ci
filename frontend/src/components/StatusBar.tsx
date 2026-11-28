import { useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useSystemStatus } from '../api/hooks/useSystemStatus';
import { useDistribution, useSqliteDetail, useSystem, useActivity } from '../api/hooks/useDashboard';
import { HEALTH_LABEL, LATENCY_NORMAL_MS, type HealthLevel } from '../api/hooks/systemStatusModel';
import { fmtBytes, fmtCount, fmtPercent, fmtSigned, fmtUptime } from './statusFormat';

const HEALTH_DOT: Record<HealthLevel, string> = {
  ok: 'bg-success', warn: 'bg-warning', down: 'bg-destructive',
};
const HEALTH_TEXT: Record<HealthLevel, string> = {
  ok: 'text-success', warn: 'text-warning', down: 'text-destructive',
};

const VENUE_COLORS = ['#2563eb', '#0891b2', '#7c3aed', '#d97706', '#059669', '#db2777'];

// Appends a sample only when the polled value changes ("adjust state during
// render", not an effect).
function useProbeHistory(value: number | null | undefined, max = 30): number[] {
  const [state, setState] = useState<{ last: number | null | undefined; hist: number[] }>({ last: undefined, hist: [] });
  if (value !== state.last) {
    setState((s) => ({ last: value, hist: value == null ? s.hist : [...s.hist, value].slice(-max) }));
  }
  return state.hist;
}

/** One labelled figure in the header strip. */
function StatItem({ label, value, tone, className }: { label: string; value: ReactNode; tone?: string; className?: string }) {
  return (
    <span className={cn('flex items-baseline gap-1.5 whitespace-nowrap', className)}>
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={cn('text-sm font-semibold tabular-nums', tone)}>{value}</span>
    </span>
  );
}

function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-lg border bg-card p-4">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

function Row({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className={cn('font-medium tabular-nums', tone)}>{value}</span>
    </div>
  );
}

/** A horizontal bar with its name on the left and value on the right, on a shared scale. */
function LabelledBar({ label, value, max, color, display }: { label: string; value: number; max: number; color: string; display?: string }) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="grid grid-cols-[72px_1fr_auto] items-center gap-2 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="min-w-[84px] text-right font-medium tabular-nums">{display ?? value.toLocaleString()}</span>
    </div>
  );
}

function CorpusSection() {
  const { data } = useSystemStatus();
  const dist = useDistribution();
  const venues = [...(dist.data?.venues ?? [])].sort((a, b) => b.papers - a.papers);
  const top = venues.slice(0, 6);
  const rest = venues.slice(6).reduce((s, v) => s + v.papers, 0);
  const segments = [
    ...top.map((v, i) => ({ label: v.venue, value: v.papers, color: VENUE_COLORS[i] })),
    ...(rest > 0 ? [{ label: `其他 ${venues.length - 6} 个会议`, value: rest, color: 'var(--muted-foreground)' }] : []),
  ];
  const total = segments.reduce((s, x) => s + x.value, 0) || 1;
  const years = [...(dist.data?.years ?? [])].sort((a, b) => a.year - b.year).slice(-6);
  const yearMax = Math.max(1, ...years.map((y) => y.papers));

  return (
    <Section title="语料" aside={<span className="text-xs text-muted-foreground">{venues.length} 个会议</span>}>
      <div className="mb-3 grid grid-cols-2 gap-2">
        <div>
          <div className="text-xs text-muted-foreground">论文</div>
          <div className="text-2xl font-bold tabular-nums">{fmtCount(data?.totalPapers)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">文档（含上传与网页）</div>
          <div className="text-2xl font-bold tabular-nums">{fmtCount(data?.totalDocuments)}</div>
        </div>
      </div>
      <div className="text-xs text-muted-foreground">按会议</div>
      <div className="mt-1 flex h-2.5 overflow-hidden rounded-full bg-muted">
        {segments.map((s) => (
          <div key={s.label} style={{ width: `${(s.value / total) * 100}%`, background: s.color }} />
        ))}
      </div>
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {segments.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5">
            <span className="size-2 shrink-0 rounded-sm" style={{ background: s.color }} />
            <span className="truncate">{s.label}</span>
            <span className="ml-auto tabular-nums text-muted-foreground">{s.value.toLocaleString()}</span>
          </div>
        ))}
      </div>
      {years.length > 0 && (
        <>
          <div className="mt-4 text-xs text-muted-foreground">按年份（篇）</div>
          <div className="mt-1 grid items-end gap-1.5" style={{ gridTemplateColumns: `repeat(${years.length}, minmax(0, 1fr))` }}>
            {years.map((y) => (
              <div key={y.year} className="flex flex-col items-center gap-1">
                <span className="text-[10px] tabular-nums text-muted-foreground">{y.papers.toLocaleString()}</span>
                <div className="w-full rounded-sm bg-primary/80" style={{ height: Math.max(3, (y.papers / yearMax) * 44) }} />
                <span className="text-[11px] tabular-nums">{y.year}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </Section>
  );
}

function IndexSyncSection() {
  const { data } = useSystemStatus();
  const chunks = data?.totalChunks ?? 0;
  const points = data?.vectorPoints ?? 0;
  const max = Math.max(chunks, points, 1);
  const delta = data?.vectorDelta ?? null;
  const optimizerOk = data?.optimizerStatus === 'ok';

  return (
    <Section
      title="索引同步"
      aside={delta == null ? null : (
        <span className={cn('text-xs font-medium', delta === 0 ? 'text-success' : 'text-warning')}>
          {delta === 0 ? '一致' : delta > 0 ? '向量多于分块' : '向量少于分块'}
        </span>
      )}
    >
      <div className="grid gap-2">
        <LabelledBar label="SQL 分块" value={chunks} max={max} color="var(--muted-foreground)" />
        <LabelledBar label="Qdrant 向量" value={points} max={max} color="var(--primary)" />
      </div>
      <div className="mt-3 border-t pt-2">
        <Row
          label="差值（向量 − 分块）"
          value={delta == null ? '未知' : fmtSigned(delta)}
          tone={delta == null ? undefined : delta === 0 ? 'text-success' : 'text-warning'}
        />
        <Row label="已建 HNSW 索引（稠密 + 稀疏）" value={fmtCount(data?.indexedVectors)} />
        <Row label="Qdrant 段数" value={fmtCount(data?.segments)} />
        <Row
          label="优化器"
          value={data?.optimizerStatus == null ? '未知' : optimizerOk ? '正常' : '异常'}
          tone={data?.optimizerStatus == null ? undefined : optimizerOk ? 'text-success' : 'text-warning'}
        />
      </div>
    </Section>
  );
}

/** Latency line against a labelled threshold, with a labelled y-axis. */
function LatencyChart({ values }: { values: number[] }) {
  const W = 300;
  const H = 72;
  const PAD_L = 48;
  const top = Math.max(LATENCY_NORMAL_MS * 1.5, ...values) * 1.1;
  const x = (i: number) => PAD_L + (values.length <= 1 ? 0 : (i / (values.length - 1)) * (W - PAD_L - 4));
  const y = (v: number) => H - 4 - (v / top) * (H - 12);
  const thresholdY = y(LATENCY_NORMAL_MS);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="检索探测延迟">
      <text x={PAD_L - 4} y={10} textAnchor="end" className="fill-muted-foreground text-[9px]">{Math.round(top)} ms</text>
      <text x={PAD_L - 4} y={H - 4} textAnchor="end" className="fill-muted-foreground text-[9px]">0</text>
      <line x1={PAD_L} x2={W - 4} y1={thresholdY} y2={thresholdY} stroke="var(--warning)" strokeDasharray="4 3" strokeWidth={1} />
      <text x={W - 4} y={thresholdY - 3} textAnchor="end" className="fill-warning text-[9px]">阈值 {LATENCY_NORMAL_MS} ms</text>
      <line x1={PAD_L} x2={PAD_L} y1={4} y2={H - 4} stroke="var(--border)" />
      {values.length > 1 && (
        <polyline
          points={values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}
          fill="none" stroke="var(--primary)" strokeWidth={1.5}
        />
      )}
      {values.map((v, i) => (
        <circle key={i} cx={x(i)} cy={y(v)} r={i === values.length - 1 ? 3 : 1.8}
          fill={v > LATENCY_NORMAL_MS ? 'var(--warning)' : 'var(--primary)'} />
      ))}
    </svg>
  );
}

function RetrievalSection() {
  const { data } = useSystemStatus();
  const history = useProbeHistory(data?.searchProbeMs);
  const latest = history[history.length - 1] ?? data?.searchProbeMs ?? null;
  const slow = latest != null && latest > LATENCY_NORMAL_MS;
  return (
    <Section
      title="检索服务"
      aside={(
        <span className={cn('text-xs font-medium', data?.qdrantAlive == null ? 'text-muted-foreground' : data.qdrantAlive ? 'text-success' : 'text-destructive')}>
          Qdrant {data?.qdrantAlive == null ? '状态未知' : data.qdrantAlive ? '在线' : '离线'}
        </span>
      )}
    >
      <div className="flex items-baseline gap-2">
        <span className="text-xs text-muted-foreground">检索探测延迟</span>
        <span className={cn('text-2xl font-bold tabular-nums', slow && 'text-warning')}>
          {latest == null ? '—' : Math.round(latest)}
        </span>
        <span className="text-xs text-muted-foreground">ms</span>
      </div>
      <div className="mt-2">
        <LatencyChart values={history} />
        <div className="text-right text-[11px] text-muted-foreground">最近 {history.length} 次采样，每 10 秒一次</div>
      </div>
      <div className="mt-2 border-t pt-2">
        <Row label="服务已运行" value={fmtUptime(data?.processUptimeS)} />
      </div>
    </Section>
  );
}

function StorageSection() {
  const { data } = useSystemStatus();
  const sqlite = useSqliteDetail();
  const system = useSystem();
  const total = data?.diskTotalBytes ?? null;
  const free = data?.diskFreeBytes ?? null;
  const used = total != null && free != null ? total - free : null;
  const usedRatio = total ? (used ?? 0) / total : null;
  const dirs = (system.data?.disk ?? []).filter((d) => d.exists).sort((a, b) => b.bytes - a.bytes);
  const dirMax = Math.max(1, ...dirs.map((d) => d.bytes));
  const sqliteFiles = sqlite.data ? [
    { label: '数据库', value: sqlite.data.db_size_bytes },
    { label: 'WAL', value: sqlite.data.wal_size_bytes },
    { label: 'SHM', value: sqlite.data.shm_size_bytes },
  ] : [];
  const sqliteMax = Math.max(1, ...sqliteFiles.map((f) => f.value));

  return (
    <Section
      title="存储"
      aside={<span className={cn('text-xs font-medium', usedRatio != null && usedRatio > 0.9 ? 'text-destructive' : 'text-muted-foreground')}>已用 {fmtPercent(usedRatio, 0)}</span>}
    >
      <div className="text-xs text-muted-foreground">数据盘</div>
      <div className="mt-1 h-2.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn('h-full rounded-full', usedRatio != null && usedRatio > 0.9 ? 'bg-destructive' : 'bg-primary')}
          style={{ width: `${(usedRatio ?? 0) * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-xs tabular-nums">
        <span>已用 {fmtBytes(used)} / 共 {fmtBytes(total)}</span>
        <span className="font-semibold">可用 {fmtBytes(free)}</span>
      </div>
      {dirs.length > 0 && (
        <div className="mt-3 grid gap-1.5 border-t pt-2">
          <div className="text-xs text-muted-foreground">数据目录</div>
          {dirs.slice(0, 4).map((d) => (
            <LabelledBar
              key={d.path}
              label={d.path.split(/[\\/]/).filter(Boolean).pop() ?? d.path}
              value={d.bytes} max={dirMax} color="var(--primary)" display={fmtBytes(d.bytes)}
            />
          ))}
        </div>
      )}
      {sqliteFiles.length > 0 && (
        <div className="mt-3 grid gap-1.5 border-t pt-2">
          <div className="text-xs text-muted-foreground">SQLite 文件</div>
          {sqliteFiles.map((f) => (
            <LabelledBar key={f.label} label={f.label} value={f.value} max={sqliteMax} color="var(--muted-foreground)" display={fmtBytes(f.value)} />
          ))}
        </div>
      )}
    </Section>
  );
}

/**
 * Header status strip: every figure carries its own label; clicking it opens
 * the detail panel as a popover (no page dimming). Public non-admin callers
 * only see the paper count, from the public `/stats`.
 */
export default function StatusBar({ compact = false }: { compact?: boolean }) {
  const { data } = useSystemStatus();
  const activity = useActivity(10);
  const [open, setOpen] = useState(false);

  if (!data) {
    return <div className="h-8 w-48 animate-pulse rounded-full bg-muted" />;
  }

  const delta = data.vectorDelta;
  const running = activity.data?.active.running ?? data.runningTasks ?? 0;
  const pending = activity.data?.active.pending ?? 0;

  const trigger = (
    <button
      type="button"
      className={cn(
        'flex items-center gap-4 rounded-full border px-3.5 py-1.5 transition-colors hover:bg-accent',
        open && 'bg-accent',
      )}
    >
      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <span className={cn('size-2 rounded-full', HEALTH_DOT[data.health])} />
        <span className={cn('text-xs font-medium', HEALTH_TEXT[data.health])}>{HEALTH_LABEL[data.health]}</span>
      </span>
      <StatItem label="论文" value={fmtCount(data.totalPapers)} />
      {!compact && data.hasDetail && (
        <>
          <StatItem
            className="hidden xl:flex"
            label="向量"
            value={(
              <>
                {fmtCount(data.vectorPoints)}
                {delta != null && delta !== 0 && <span className="ml-1 text-xs text-warning">{fmtSigned(delta)}</span>}
              </>
            )}
          />
          <StatItem
            label="检索延迟"
            value={data.searchProbeMs == null ? '—' : `${Math.round(data.searchProbeMs)} ms`}
            tone={data.searchProbeMs != null && data.searchProbeMs > LATENCY_NORMAL_MS ? 'text-warning' : undefined}
          />
          <StatItem className="hidden 2xl:flex" label="磁盘可用" value={fmtBytes(data.diskFreeBytes)} />
          <StatItem label="运行中任务" value={running} tone={running > 0 ? 'text-primary' : undefined} />
        </>
      )}
      {data.hasDetail && <ChevronDown className={cn('size-3.5 text-muted-foreground transition-transform', open && 'rotate-180')} />}
    </button>
  );

  if (!data.hasDetail) return trigger;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      <PopoverContent align="end" sideOffset={8} className="w-[min(820px,94vw)] max-h-[82vh] overflow-auto p-4">
        <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="flex items-center gap-1.5">
            <span className={cn('size-2.5 rounded-full', HEALTH_DOT[data.health])} />
            <span className="text-sm font-semibold">系统{HEALTH_LABEL[data.health]}</span>
          </span>
          {data.healthReasons.map((r) => (
            <span key={r} className="rounded-md bg-warning/15 px-2 py-0.5 text-xs text-warning">{r}</span>
          ))}
          <span className="ml-auto text-xs text-muted-foreground">
            任务：运行中 {running} · 排队 {pending}
          </span>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <CorpusSection />
          <IndexSyncSection />
          <RetrievalSection />
          <StorageSection />
        </div>
      </PopoverContent>
    </Popover>
  );
}
