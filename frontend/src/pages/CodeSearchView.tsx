/**
 * CodeSearchView — the 代码 half of /search?mode=code. Mounted by Search.tsx
 * only when the mode switch is on "代码"; paper mode's own JSX is untouched.
 *
 * Structure mirrors paper mode's shape (search bar card -> filter panel ->
 * timings -> distribution -> result list) using the code-mode counterparts:
 * CodeScopePanel, CodeTimingsPanel, CodeDistributionPanel, CodeJobBoard,
 * CodeHitCard.
 */
import { useState, useMemo, useEffect, useRef } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { CircleAlert, Filter, SearchX } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { readCodeUI, writeCodeUI, toggleProjectSelection, type CodeUIState } from './codeSearchModel';
import { useFederatedCodeSearch, type FederatedCodeSearchResponse } from '../api/hooks/useCodeSearch';
import CodeScopePanel from '../components/code/CodeScopePanel';
import CodeTimingsPanel from '../components/code/CodeTimingsPanel';
import CodeDistributionPanel from '../components/code/CodeDistributionPanel';
import CodeJobBoard from '../components/code/CodeJobBoard';
import CodeHitCard from '../components/code/CodeHitCard';
import ModeSwitch, { type SearchMode } from '../components/ModeSwitch';
import Segmented from '../components/common/Segmented';
import SearchBox from '../components/common/SearchBox';

const RETRIEVAL_OPTIONS: { value: CodeUIState['mode']; label: string }[] = [
  { value: 'hybrid', label: '混合' },
  { value: 'dense', label: '语义' },
  { value: 'keyword', label: '关键词' },
];

const SCOPE_STATE_LABEL: Record<string, string> = {
  preparing: '索引中',
  recovery_required: '需要恢复',
  unavailable: '不可用',
  completed: '已完成',
};

// FederatedLegDiagnostics.excluded_scopes: scopes the backend withheld from
// this search, each with the reason it was withheld.
function ExcludedScopesRow({ data }: { data: FederatedCodeSearchResponse }) {
  const excluded = [...data.private.excluded_scopes, ...data.public.excluded_scopes];
  if (excluded.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="font-medium text-muted-foreground">未参与本次检索 {excluded.length} 个</span>
      {excluded.map((s) => (
        <span key={s.project} title={s.project} className="flex max-w-[280px] items-center gap-1 rounded-md bg-muted px-2 py-0.5 text-muted-foreground">
          <span className="truncate">{s.project}</span>
          <span className="shrink-0 text-warning">{SCOPE_STATE_LABEL[s.state] ?? s.state}</span>
        </span>
      ))}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
      {label}
      {children}
    </label>
  );
}

export default function CodeSearchView({
  sp, setSp, showFilters, setShowFilters, mode, setMode,
}: {
  sp: URLSearchParams;
  setSp: (next: URLSearchParams, opts?: { replace?: boolean }) => void;
  showFilters: boolean;
  setShowFilters: Dispatch<SetStateAction<boolean>>;
  mode: SearchMode;
  setMode: (next: SearchMode) => void;
}) {
  const codeUi = useMemo(() => readCodeUI(sp), [sp]);

  const patchCodeUi = (delta: Partial<CodeUIState>) => {
    const next = writeCodeUI({ ...codeUi, ...delta });
    next.set('mode', 'code');
    setSp(next, { replace: true });
  };

  const [inputValue, setInputValue] = useState(codeUi.query);
  const lastSyncedQuery = useRef(codeUi.query);
  useEffect(() => {
    if (codeUi.query !== lastSyncedQuery.current) {
      lastSyncedQuery.current = codeUi.query;
      setInputValue(codeUi.query);
    }
  }, [codeUi.query]);

  // useQuery keyed on the built request body: a filter change already
  // refetches; the 检索 button only commits the typed text, or forces a
  // refetch when the text is unchanged.
  const search = useFederatedCodeSearch(codeUi);

  const handleSearch = (raw: string) => {
    const q = raw.trim();
    if (!q) return;
    if (q === codeUi.query) search.refetch();
    else patchCodeUi({ query: q });
  };

  const hits = useMemo(() => search.data?.hits ?? [], [search.data]);
  const maxScore = useMemo(() => (hits.length ? Math.max(...hits.map((h) => h.score)) : 0), [hits]);

  // Same pure toggle and 50-project cap as CodeScopePanel's tree, so a
  // distribution-bar click can't do anything the tree couldn't.
  const toggleProject = (project: string) => patchCodeUi({ projects: toggleProjectSelection(codeUi.projects, project) });
  const toggleSource = (sourceId: string) => patchCodeUi({
    sourceIds: codeUi.sourceIds.includes(sourceId)
      ? codeUi.sourceIds.filter((s) => s !== sourceId)
      : [...codeUi.sourceIds, sourceId],
  });

  const filterCount = codeUi.projects.length + (codeUi.includePublic ? 1 : 0);

  return (
    <div className="flex flex-col gap-4">
      <section className="flex flex-col gap-3 rounded-xl border bg-card px-5 py-4">
        <div className="flex flex-col gap-2 sm:flex-row">
          <ModeSwitch mode={mode} onChange={setMode} />
          <SearchBox
            value={inputValue}
            onChange={setInputValue}
            onSearch={handleSearch}
            loading={search.isFetching}
            placeholder="描述代码做什么（如：带指数退避的 Qdrant 批量 upsert），或用关键词模式查确切符号名"
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Field label="检索方式">
              <Segmented
                ariaLabel="检索方式"
                value={codeUi.mode}
                options={RETRIEVAL_OPTIONS}
                onChange={(v) => patchCodeUi({ mode: v })}
              />
            </Field>
            <Field label="语言">
              <Input
                className="h-8 w-28 text-xs"
                placeholder="全部"
                value={codeUi.language}
                onChange={(e) => patchCodeUi({ language: e.target.value })}
              />
            </Field>
            <Field label="路径">
              <Input
                className="h-8 w-40 font-mono text-xs"
                placeholder="src/**/*.rs"
                value={codeUi.pathGlob}
                onChange={(e) => patchCodeUi({ pathGlob: e.target.value })}
              />
            </Field>
            <Field label="返回条数">
              <Input
                type="number"
                min={1}
                max={50}
                className="h-8 w-16 text-xs tabular-nums"
                value={codeUi.topK}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n) && n >= 1) patchCodeUi({ topK: Math.min(50, Math.round(n)) });
                }}
              />
            </Field>
            <Field label="重排">
              <Switch checked={codeUi.rerank} onCheckedChange={(v) => patchCodeUi({ rerank: v })} />
            </Field>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowFilters((s) => !s)}
            className={cn(showFilters ? 'text-primary' : 'text-muted-foreground')}
          >
            <Filter className="size-3.5" />
            {showFilters ? '收起检索范围' : '检索范围'}
            {filterCount > 0 && <Badge className="ml-1 h-4 px-1.5 text-[10px]">{filterCount}</Badge>}
          </Button>
        </div>

        <CodeScopePanel
          value={{
            projects: codeUi.projects, includePublic: codeUi.includePublic,
            sourceIds: codeUi.sourceIds, domains: codeUi.domains,
          }}
          onChange={(next) => patchCodeUi(next)}
          expanded={showFilters}
        />
      </section>

      <CodeJobBoard />

      {search.isError && (
        <Alert variant="destructive">
          <CircleAlert />
          <AlertTitle>检索失败</AlertTitle>
          <AlertDescription>{(search.error as Error)?.message}</AlertDescription>
        </Alert>
      )}

      <CodeTimingsPanel timings={search.data?.timings_ms} searching={search.isFetching} />

      {search.data && (
        <>
          <CodeDistributionPanel
            hits={hits}
            selectedProjects={codeUi.projects}
            selectedSourceIds={codeUi.sourceIds}
            onToggleProject={toggleProject}
            onToggleSource={toggleSource}
          />

          <ExcludedScopesRow data={search.data} />

          <div className="flex items-center gap-2 text-sm">
            <span>找到 <strong className="text-primary tabular-nums">{search.data.total}</strong> 条结果</span>
            <Badge variant="secondary" className="bg-primary/10 font-normal text-primary">
              {search.data.fusion === 'rerank' ? '已重排' : 'RRF 融合'}
            </Badge>
          </div>

          {hits.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-12 text-sm text-muted-foreground">
              <SearchX className="size-8" />
              没有匹配的代码，先建索引或换个描述
            </div>
          ) : (
            <div>
              {hits.map((h, i) => (
                <CodeHitCard key={`${h.origin}:${h.path}:${h.start_line}:${i}`} hit={h} index={i} maxScore={maxScore} />
              ))}
            </div>
          )}
        </>
      )}

      {!search.isFetching && !search.data && !search.isError && (
        <div className="py-16 text-center text-sm text-muted-foreground">输入描述后按回车开始检索</div>
      )}
    </div>
  );
}
