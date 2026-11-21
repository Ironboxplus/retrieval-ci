import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import api from '../client';

export type PublicCodeUpdatePolicy = 'manual' | 'scheduled';
export type PublicCodeJobStatus = 'queued' | 'running' | 'completed' | 'failed';
export type PublicCodeSearchMode = 'hybrid' | 'dense' | 'keyword';
export type PublicCodeSearchEffectiveMode = PublicCodeSearchMode | 'unavailable';

export interface PublicCodeSource {
  source_id: string;
  domain: string;
  relative_path: string;
  upstream_url: string;
  default_ref: string;
  proxy_profile: string | null;
  credential_profile: string | null;
  language_overrides: Record<string, string>;
  source_origin: string;
  created_by_user_id: string | null;
  active_manifest_file_count: number;
  active_chunk_count: number;
  active_vector_count: number;
  active_promoted_at: string | null;
  created_at: string | null;
  updated_at: string | null;
  update_policy: PublicCodeUpdatePolicy;
  update_interval_minutes: number | null;
  active_revision: string | null;
  last_mirrored_revision: string | null;
  last_remote_revision: string | null;
  last_check_error: string | null;
  last_checked_at: string | null;
}

export interface PublicCodeJob {
  job_id: string;
  operation: string;
  source_ids: string[];
  total_sources: number;
  status: PublicCodeJobStatus;
  completed_sources: number;
  progress: number;
  phase: string;
  current_source_id: string | null;
  current_path: string | null;
  files_done: number;
  files_total: number;
  chunks_done: number;
  message: string | null;
  source_progress: Record<string, Record<string, unknown>>;
  results: Record<string, unknown>[];
  errors: Record<string, string>;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface PublicCodeSourceCreateRequest {
  source_id: string;
  domain: string;
  upstream_url: string;
  relative_path: string;
  default_ref?: string;
  proxy_profile?: string | null;
  credential_profile?: string | null;
  language_overrides?: Record<string, string>;
  update_policy?: PublicCodeUpdatePolicy;
  update_interval_minutes?: number | null;
  index_now?: boolean;
}

export interface PublicCodeSourceCreateResponse {
  source: PublicCodeSource;
  index_job: PublicCodeJob | null;
}

export interface PublicCodeSourceSelection {
  source_ids: string[];
  domains: string[];
}

export interface PublicCodeSearchRequest extends PublicCodeSourceSelection {
  query: string;
  top_k?: number;
  mode?: PublicCodeSearchMode;
  language?: string | null;
  rerank?: boolean;
}

export interface PublicCodeSearchHit {
  score: number;
  retrieval_score: number;
  reranker_score: number | null;
  source_id: string;
  revision: string;
  domain: string;
  path: string;
  language: string;
  start_line: number;
  end_line: number;
  label: string;
  content: string;
  code: string;
}

export interface PublicCodeSearchResponse {
  query: string;
  selected_source_ids: string[];
  hits: PublicCodeSearchHit[];
  requested_mode: PublicCodeSearchMode;
  effective_mode: PublicCodeSearchEffectiveMode;
  notice: string | null;
  rerank_applied: boolean;
  candidate_count: number;
  rerank_candidate_count: number;
}

const SOURCE_QUERY_KEY = ['public-code', 'sources'] as const;
const JOB_QUERY_KEY = ['public-code', 'jobs'] as const;

export function hasActivePublicCodeJobs(jobs: PublicCodeJob[] | undefined): boolean {
  return (jobs ?? []).some((job) => job.status === 'queued' || job.status === 'running');
}

// `enabled` defaults true (PublicCodeConsole's own admin-gated page always
// wants it); F2-c's CodeScopePanel passes isAdmin && includePublic so a
// non-admin viewer never fires the admin-only GET (would just 403).
export function usePublicCodeSources(enabled = true) {
  return useQuery({
    queryKey: SOURCE_QUERY_KEY,
    queryFn: () => api.get<PublicCodeSource[]>('/public-code/sources').then((response) => response.data),
    enabled,
  });
}

export function usePublicCodeJobs(limit = 50) {
  return useQuery({
    queryKey: [...JOB_QUERY_KEY, limit],
    queryFn: () =>
      api.get<PublicCodeJob[]>('/public-code/jobs', { params: { limit } }).then((response) => response.data),
    refetchInterval: (query) => hasActivePublicCodeJobs(query.state.data) ? 2_000 : false,
  });
}

function useInvalidatePublicCode() {
  const queryClient = useQueryClient();
  return () => Promise.all([
    queryClient.invalidateQueries({ queryKey: SOURCE_QUERY_KEY }),
    queryClient.invalidateQueries({ queryKey: JOB_QUERY_KEY }),
  ]);
}

export function useCreatePublicCodeSource() {
  const invalidate = useInvalidatePublicCode();
  return useMutation({
    mutationFn: (request: PublicCodeSourceCreateRequest) =>
      api.post<PublicCodeSourceCreateResponse>('/public-code/sources', request).then((response) => response.data),
    onSuccess: invalidate,
  });
}

export function useCheckPublicCodeUpdates() {
  const invalidate = useInvalidatePublicCode();
  return useMutation({
    mutationFn: (selection: PublicCodeSourceSelection) =>
      api.post<PublicCodeJob>('/public-code/sources/check-updates', selection).then((response) => response.data),
    onSuccess: invalidate,
  });
}

export function useIndexPublicCodeSources() {
  const invalidate = useInvalidatePublicCode();
  return useMutation({
    mutationFn: (selection: PublicCodeSourceSelection) =>
      api.post<PublicCodeJob>('/public-code/sources/index', selection).then((response) => response.data),
    onSuccess: invalidate,
  });
}

export function useUpdatePublicCodePolicy() {
  const invalidate = useInvalidatePublicCode();
  return useMutation({
    mutationFn: ({
      sourceId,
      updatePolicy,
      updateIntervalMinutes,
    }: {
      sourceId: string;
      updatePolicy: PublicCodeUpdatePolicy;
      updateIntervalMinutes: number | null;
    }) => api.put<PublicCodeSource>(`/public-code/sources/${encodeURIComponent(sourceId)}/update-policy`, {
      update_policy: updatePolicy,
      update_interval_minutes: updateIntervalMinutes,
    }).then((response) => response.data),
    onSuccess: invalidate,
  });
}

export function useSearchPublicCode() {
  return useMutation({
    mutationFn: (request: PublicCodeSearchRequest) =>
      api.post<PublicCodeSearchResponse>('/public-code/search', request, { timeout: 120_000 })
        .then((response) => response.data),
  });
}
