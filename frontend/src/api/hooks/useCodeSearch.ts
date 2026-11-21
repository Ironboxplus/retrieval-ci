import { useMutation, useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import api from '../client';
import type { InventoryResponse } from './useAdminConsole';
import { buildFederatedSearchRequest, type CodeSearchTimingsMsLike, type CodeUIState } from '../../pages/codeSearchModel';

// F2-c folded the single-leg /code/search page (CodeSearch.tsx) into
// /search?mode=code, which only ever calls federated-search (below) — the
// plain CodeHit/CodeSearchResponse/CodeSearchArgs/useCodeSearch() that used
// to wrap /code/search directly were removed here since nothing in the
// frontend called them any more (the backend route itself is untouched —
// MCP still uses it).

// ---- Federated search (private Code KB + curated Public Code, F2-c) ----

export interface FederatedCodeHit {
  origin: 'private' | 'public';
  path: string;
  start_line: number;
  end_line: number;
  label: string;
  language?: string | null;
  score: number;
  retrieval_score: number;
  fusion_rank_score: number;
  reranker_score?: number | null;
  code: string;
  project?: string | null;
  host?: string | null;
  source_id?: string | null;
  revision?: string | null;
  domain?: string | null;
}

export interface FederatedLegDiagnostics {
  included: boolean;
  candidate_count: number;
  effective_mode?: string | null;
  notice?: string | null;
  scope_snapshot_state?: string | null;
  excluded_scopes: { project: string; state: string; active_index_job_id: string | null }[];
  searched_projects: string[];
  selected_source_ids: string[];
}

export interface FederatedCodeSearchResponse {
  query: string;
  hits: FederatedCodeHit[];
  total: number;
  fusion: 'rrf' | 'rerank';
  rerank_applied: boolean;
  private: FederatedLegDiagnostics;
  public: FederatedLegDiagnostics;
  timings_ms: CodeSearchTimingsMsLike;
}

// The private leg always runs (R1) — this is the one call code mode uses for both
// "search my own code" and "search my code + curated public code" (include_public).
//
// A code search is an idempotent read, so this is a useQuery keyed on the
// built request body (F2-c review round 2) rather than a useMutation fired
// from an effect. That removes a real bug at its root: useMutation's
// mutate()-in-an-effect interacted badly with React StrictMode's dev-only
// mount->cleanup->mount double-invoke — the request went out for real, but
// its result landed on an already-unsubscribed observer and was silently
// dropped (confirmed via an onSuccess/onSettled trace: neither ever fired,
// even though the network response came back 200). useQuery has no such
// per-call observer lifecycle to lose. It also gives free dedup/caching, and
// `placeholderData: keepPreviousData` keeps the previous hits on screen
// while a new query (a filter change, a new search term) is in flight,
// instead of the results area blanking out and back.
export function useFederatedCodeSearch(ui: CodeUIState) {
  const request = buildFederatedSearchRequest(ui);
  return useQuery({
    queryKey: ['code-federated', request],
    queryFn: () =>
      api
        .post<FederatedCodeSearchResponse>('/code/federated-search', request, { timeout: 120000 })
        .then((r) => r.data),
    enabled: !!ui.query.trim(),
    placeholderData: keepPreviousData,
  });
}

export interface CodeIndexJob {
  job_id: string;
  project: string;
  host: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  files_total: number;
  chunks_total: number;
  removed_total: number;
  chunks_done: number;
  indexed: number;
  queue_position: number;
  error?: string | null;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
}

// The caller's recent code-index jobs (async queue, W156). Polls while any job
// is still queued/running so progress bars advance live.
export function useCodeIndexJobs() {
  return useQuery<CodeIndexJob[]>({
    queryKey: ['code', 'jobs'],
    queryFn: () =>
      api.get<{ jobs: CodeIndexJob[] }>('/code/jobs').then((r) => r.data.jobs ?? []),
    refetchInterval: (q) =>
      (q.state.data ?? []).some(
        (j) => j.status === 'queued' || j.status === 'running',
      )
        ? 2000
        : false,
  });
}

export interface CancelCodeIndexJobError {
  status: number;
  code?: string;
}

// F2-a: cancel a queued/not-yet-writing job. 409 code_index_committing (a write
// already started) / code_index_already_ended surface as a typed error so the
// card can render "正在提交，无法取消" instead of a generic failure toast.
export function useCancelCodeIndexJob() {
  const qc = useQueryClient();
  return useMutation<CodeIndexJob, CancelCodeIndexJobError, string>({
    mutationFn: (jobId: string) =>
      api.post<CodeIndexJob>(`/code/jobs/${jobId}/cancel`).then((r) => r.data)
        .catch((err) => {
          const status = err?.response?.status;
          const code = err?.response?.data?.detail?.code;
          throw { status, code } as CancelCodeIndexJobError;
        }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['code', 'jobs'] }),
  });
}

// Full tree (user -> host -> project) for the code-mode project TreeSelect
// (components/inventoryTree.ts's projectScopeOptions consumes this). Compact
// summary (detail=false) is enough — the picker stops at the project level.
export function useMyCodeInventory() {
  return useQuery<InventoryResponse>({
    queryKey: ['code', 'inventory'],
    queryFn: () =>
      api
        .get<InventoryResponse>('/index/inventory', { params: { kind: 'code' } })
        .then((r) => r.data),
  });
}
