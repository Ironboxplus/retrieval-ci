/**
 * CodeScopePanel — code mode's advanced-filter panel, the counterpart to
 * PaperFilterPanel: a host → project checkbox tree over the caller's own
 * code inventory and, admin-only, the include-public-code switch with a
 * source picker.
 *
 * `FederatedCodeSearchRequest.projects` has a backend `max_length=50`, and an
 * owner can have hundreds of projects, so selection is real: the ticked set
 * is exactly `value.projects`, and empty means "search everything owned"
 * (federated-search's contract for an empty `projects[]`). That is shown by
 * every size bar reading primary while nothing is ticked; once anything is
 * ticked, unticked rows dim. `isProjectCapped` / `isHostCapped` disable a
 * checkbox before a click could produce a >50 request.
 *
 * Stays controlled — parent (Search.tsx code mode) owns the CodeUIState slice.
 */
import { useMemo, useState } from 'react';
import { ChevronRight, Filter, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Switch } from '@/components/ui/switch';
import { buildInventoryTree, maxProjectChunkCount, type InventoryTreeNode } from '../inventoryTree';
import { pathTail, isProjectCapped, isHostCapped, MAX_SELECTED_PROJECTS } from '../../pages/codeSearchModel';
import { useMyCodeInventory, useCodeIndexJobs } from '../../api/hooks/useCodeSearch';
import { usePublicCodeSources } from '../../api/hooks/usePublicCode';
import { useConsoleAccess } from '../../api/hooks/useAdminConsole';

export interface CodeScopeValue {
  projects: string[];
  includePublic: boolean;
  sourceIds: string[];
  domains: string[];
}

export interface CodeScopePanelProps {
  value: CodeScopeValue;
  onChange: (next: CodeScopeValue) => void;
  expanded?: boolean;
}

function projectsUnder(n: InventoryTreeNode): string[] {
  if (n.type === 'project') return [n.label];
  return (n.children ?? []).flatMap(projectsUnder);
}

function ProjectRow({
  n, projectMax, jobProgress, dim, checked, disabled, onToggle,
}: {
  n: InventoryTreeNode; projectMax: number; jobProgress?: number; dim: boolean;
  checked: boolean; disabled: boolean; onToggle: () => void;
}) {
  const active = n.codeActiveIndexJobId != null;
  const chunks = n.chunkCount ?? 0;
  const pct = projectMax > 0 ? Math.min(1, chunks / projectMax) : 0;
  return (
    <label
      title={n.label}
      className={cn(
        'grid h-8 cursor-pointer grid-cols-[16px_minmax(0,1fr)_64px_72px] items-center gap-2 rounded-md pl-7 pr-2 text-sm hover:bg-accent',
        disabled && 'cursor-not-allowed',
      )}
    >
      <Checkbox checked={checked} disabled={disabled} onCheckedChange={onToggle} />
      <span className={cn('truncate', dim && 'text-muted-foreground/60')}>{pathTail(n.label)}</span>
      {active ? (
        <span className="col-span-2 flex items-center justify-end gap-1.5 text-xs text-primary">
          <Loader2 className="size-3 animate-spin" />
          索引中 {Math.round((jobProgress ?? 0) * 100)}%
        </span>
      ) : (
        <>
          <span className="h-1.5 overflow-hidden rounded-full bg-muted">
            <span className={cn('block h-full rounded-full', dim ? 'bg-muted-foreground/30' : 'bg-primary')} style={{ width: `${pct * 100}%` }} />
          </span>
          <span className={cn('text-right text-xs tabular-nums', dim ? 'text-muted-foreground/60' : 'text-muted-foreground')}>
            {chunks.toLocaleString()}
          </span>
        </>
      )}
    </label>
  );
}

export default function CodeScopePanel({ value, onChange, expanded = true }: CodeScopePanelProps) {
  const isAdmin = useConsoleAccess();
  const { data: inventory, isLoading: inventoryLoading } = useMyCodeInventory();
  const { data: jobs = [] } = useCodeIndexJobs();
  const jobProgressByJobId = useMemo(() => new Map(jobs.map((j) => [j.job_id, j.progress])), [jobs]);
  // This endpoint is always exactly the caller's own inventory (one owner) —
  // flatten past that single user row straight to hosts.
  const { nodes: pureNodes } = useMemo(() => buildInventoryTree(inventory), [inventory]);
  const hostNodes = useMemo(() => pureNodes.flatMap((u) => u.children ?? []), [pureNodes]);
  const projectMax = useMemo(() => maxProjectChunkCount(hostNodes), [hostNodes]);
  const selectedSet = useMemo(() => new Set(value.projects), [value.projects]);
  const [collapsedHosts, setCollapsedHosts] = useState<Set<string>>(new Set());

  const { data: sources } = usePublicCodeSources(isAdmin && value.includePublic);

  const patch = (delta: Partial<CodeScopeValue>) => onChange({ ...value, ...delta });
  const setProjects = (projects: string[]) => patch({ projects: projects.slice(0, MAX_SELECTED_PROJECTS) });
  const selCount = value.projects.length;
  const toggleSource = (id: string) =>
    patch({ sourceIds: value.sourceIds.includes(id) ? value.sourceIds.filter((s) => s !== id) : [...value.sourceIds, id] });

  const renderHost = (host: InventoryTreeNode) => {
    const all = projectsUnder(host);
    const picked = all.filter((p) => selectedSet.has(p)).length;
    const open = !collapsedHosts.has(host.key);
    const state: boolean | 'indeterminate' = picked === 0 ? false : picked === all.length ? true : 'indeterminate';
    const hostDisabled = picked === 0 && isHostCapped(all.length, selCount, picked);
    const toggleHost = () => {
      if (picked > 0) setProjects(value.projects.filter((p) => !all.includes(p)));
      else setProjects([...value.projects, ...all]);
    };
    return (
      <div key={host.key}>
        <div className="flex h-8 items-center gap-2 rounded-md px-1 text-sm hover:bg-accent">
          <button
            type="button"
            aria-label={open ? '收起' : '展开'}
            className="flex size-5 items-center justify-center text-muted-foreground"
            onClick={() => setCollapsedHosts((prev) => {
              const next = new Set(prev);
              if (next.has(host.key)) next.delete(host.key); else next.add(host.key);
              return next;
            })}
          >
            <ChevronRight className={cn('size-3.5 transition-transform', open && 'rotate-90')} />
          </button>
          <Checkbox checked={state} disabled={hostDisabled} onCheckedChange={toggleHost} aria-label={host.label} />
          <span className="font-medium">{host.label}</span>
          <span className="text-xs text-muted-foreground">
            {all.length} 个项目{picked > 0 ? `，已选 ${picked}` : ''}
          </span>
        </div>
        {open && (host.children ?? []).map((n) => (n.type === 'project' ? (
          <ProjectRow
            key={n.key}
            n={n}
            projectMax={projectMax}
            dim={selCount > 0 && !selectedSet.has(n.label)}
            jobProgress={n.codeActiveIndexJobId ? jobProgressByJobId.get(n.codeActiveIndexJobId) : undefined}
            checked={selectedSet.has(n.label)}
            disabled={isProjectCapped(selCount, selectedSet.has(n.label))}
            onToggle={() => setProjects(selectedSet.has(n.label)
              ? value.projects.filter((p) => p !== n.label)
              : [...value.projects, n.label])}
          />
        ) : renderHost(n)))}
      </div>
    );
  };

  return (
    <AnimatePresence initial={false}>
      {expanded && (
        <motion.div
          key="code-scope-panel"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
          className="overflow-hidden"
        >
          <section className="rounded-xl border bg-muted/30 px-4 py-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Filter className="size-3.5 text-primary" />
                <span className="text-sm font-semibold">检索范围</span>
                <Badge variant="secondary" className="font-normal">
                  {selCount > 0 ? `已选 ${selCount} / 上限 ${MAX_SELECTED_PROJECTS} 个项目` : '全部项目'}
                </Badge>
                {value.includePublic && <Badge variant="secondary" className="bg-success/10 font-normal text-success">含公共代码</Badge>}
              </div>
              {(selCount > 0 || value.includePublic) && (
                <Button variant="ghost" size="sm" onClick={() => onChange({ projects: [], includePublic: false, sourceIds: [], domains: [] })}>
                  清除
                </Button>
              )}
            </div>

            <div className={cn('grid gap-4', isAdmin && 'md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]')}>
              <div className="min-w-0">
                <div className="grid grid-cols-[16px_minmax(0,1fr)_64px_72px] gap-2 border-b pb-1 pl-7 pr-2 text-xs text-muted-foreground">
                  <span />
                  <span>主机 / 项目</span>
                  <span className="col-span-2 text-right">已索引分块</span>
                </div>
                {inventoryLoading ? (
                  <div className="flex items-center justify-center gap-2 py-6 text-sm text-muted-foreground">
                    <Loader2 className="size-4 animate-spin" /> 加载项目…
                  </div>
                ) : hostNodes.length === 0 ? (
                  <div className="py-6 text-center text-sm text-muted-foreground">暂无已索引项目</div>
                ) : (
                  <div className="max-h-64 overflow-auto py-1">{hostNodes.map(renderHost)}</div>
                )}
              </div>

              {isAdmin && (
                <div className="min-w-0 space-y-2">
                  <label className="flex items-center gap-2 text-sm">
                    <Switch
                      checked={value.includePublic}
                      onCheckedChange={(v) => patch({ includePublic: v, sourceIds: v ? value.sourceIds : [], domains: v ? value.domains : [] })}
                    />
                    纳入公共代码
                  </label>
                  {value.includePublic && (
                    <div className="rounded-md border bg-background">
                      <div className="border-b px-2.5 py-1 text-xs text-muted-foreground">
                        公共源 · {value.sourceIds.length > 0 ? `已选 ${value.sourceIds.length}` : '全部'}
                      </div>
                      <div className="max-h-48 overflow-auto p-1">
                        {(sources ?? []).map((s) => (
                          <label key={s.source_id} className="flex h-7 cursor-pointer items-center gap-2 rounded px-1.5 text-sm hover:bg-accent">
                            <Checkbox checked={value.sourceIds.includes(s.source_id)} onCheckedChange={() => toggleSource(s.source_id)} />
                            <span className="truncate">{s.source_id}</span>
                            <span className="ml-auto shrink-0 text-xs text-muted-foreground">{s.domain}</span>
                          </label>
                        ))}
                        {sources && sources.length === 0 && <div className="px-1.5 py-2 text-xs text-muted-foreground">没有公共源</div>}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
