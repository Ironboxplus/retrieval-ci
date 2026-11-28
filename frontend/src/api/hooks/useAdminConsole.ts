import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import api from '../client';
import { PUBLIC_MODE } from '../../constants';
import { useMe } from './useAuth';

/**
 * Admin-only management console for ACE Journal + the index inventory.
 *
 * Visibility: the internal 4090:8000 build (PUBLIC_MODE off) IS the admin panel →
 * always visible. The public build shows it only to `tier==='admin'`. The backend
 * enforces the same (resolve_admin: gateway admin tier, or LAN internal-admin), so
 * the frontend gate is convenience, not the security boundary.
 *
 * All calls are same-origin under `/api/admin/console/*`: on the internal build
 * they hit 4090 directly (LAN → internal-admin); on public they ride the gateway
 * session cookie through require_admin.
 */
export type ConsoleAccessState = 'loading' | 'allowed' | 'denied';

export function resolveConsoleAccess(
  publicMode: boolean,
  authenticationResolved: boolean,
  user: { tier?: string } | null | undefined,
): ConsoleAccessState {
  if (!publicMode) return 'allowed';
  if (!authenticationResolved) return 'loading';
  return user?.tier === 'admin' ? 'allowed' : 'denied';
}

export function useConsoleAccessState(): ConsoleAccessState {
  const { data: me, isFetched } = useMe();
  return resolveConsoleAccess(PUBLIC_MODE, isFetched, me);
}

export function useConsoleAccess(): boolean {
  return useConsoleAccessState() === 'allowed';
}

export interface ConsoleOverview {
  journal: { entries: number; indexed: number; projects: number; owners: number };
  index: { files: number; users: number; hosts: number; projects: number; chunks: number };
}

export interface ConsoleUser {
  owner_user_id: string;
  index_files: number;
  index_chunks: number;
  index_hosts: number;
  index_projects: number;
  has_journal: boolean;
}

export interface JournalChunk {
  chunk_id: string;
  chunk_index: number;
  level: number;
  heading?: string | null;
  content: string;
  score?: number | null;
}

export interface JournalEntry {
  entry_id: string;
  project_id: number;
  project_slug: string;
  client: string;
  source_kind: string;
  title?: string | null;
  summary?: string | null;
  body?: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
  chunk_count: number;
  indexed: boolean;
  index_error?: string | null;
  created_at?: string | null;
  chunks: JournalChunk[];
}

export interface JournalHit {
  entry: JournalEntry;
  score: number;
  best_chunk?: JournalChunk | null;
  chunks: JournalChunk[];
}

export interface JournalSearchResponse {
  query: string;
  mode: string;
  total: number;
  hits: JournalHit[];
  metrics: Record<string, unknown>;
}

export interface InventoryFile {
  rel_path: string;
  kind: string;
  collection?: string | null;
  language?: string | null;
  chunk_count: number;
  indexed_at?: string | null;
}
export interface InventoryDir { dir: string; file_count: number; chunk_count: number; files: InventoryFile[] }
export interface InventoryProject {
  project: string; host: string; root_dir?: string | null;
  file_count: number; chunk_count: number; dirs: InventoryDir[];
  // Code-scope visibility state (index_catalog's code_scope_states overlay,
  // W150/F2-c) — undefined for a doc-only project, non-null only while a
  // code index job is actually running/queued for this scope.
  code_active_index_job_id?: string | null;
  code_snapshot_state?: string | null;
}
export interface InventoryHost { host: string; file_count: number; chunk_count: number; projects: InventoryProject[] }
export interface InventoryUser { owner_user_id: string; file_count: number; chunk_count: number; hosts: InventoryHost[] }
export interface InventoryResponse { users: InventoryUser[]; totals: Record<string, number> }

const BASE = '/admin/console';

export function useConsoleOverview(enabled = true) {
  return useQuery<ConsoleOverview>({
    queryKey: ['console', 'overview'],
    enabled,
    queryFn: () => api.get<ConsoleOverview>(`${BASE}/overview`).then((r) => r.data),
  });
}

export function useConsoleUsers(enabled = true) {
  return useQuery<{ users: ConsoleUser[]; total: number }>({
    queryKey: ['console', 'users'],
    enabled,
    queryFn: () => api.get(`${BASE}/users`).then((r) => r.data),
  });
}

export interface JournalSearchArgs {
  query: string;
  mode: 'keyword' | 'hybrid' | 'dense' | 'sparse';
  top_k?: number;
  owner_user_id?: string | null;
  source_kind?: string | null;
  include_chunks?: boolean;
  chunk_limit?: number | null;
}

export function useJournalSearch() {
  return useMutation<JournalSearchResponse, unknown, JournalSearchArgs>({
    mutationFn: (args) =>
      api.post<JournalSearchResponse>(`${BASE}/journal/search`, {
        query: args.query,
        mode: args.mode,
        top_k: args.top_k ?? 20,
        owner_user_id: args.owner_user_id || null,
        source_kind: args.source_kind || null,
        include_chunks: args.include_chunks ?? false,
        chunk_limit: args.chunk_limit ?? 5,
      }).then((r) => r.data),
  });
}

export function useJournalEntries(params: {
  owner_user_id?: string | null;
  project_slug?: string | null;
  q?: string | null;
  limit?: number;
}, enabled = true) {
  return useQuery<{ entries: JournalEntry[]; total: number }>({
    queryKey: ['console', 'journal', 'entries', params],
    enabled,
    queryFn: () =>
      api.get(`${BASE}/journal/entries`, {
        params: {
          owner_user_id: params.owner_user_id || undefined,
          project_slug: params.project_slug || undefined,
          q: params.q || undefined,
          limit: params.limit ?? 50,
        },
      }).then((r) => r.data),
  });
}

export function useJournalEntry(entryId: string | null) {
  return useQuery<JournalEntry>({
    queryKey: ['console', 'journal', 'entry', entryId],
    enabled: !!entryId,
    queryFn: () => api.get(`${BASE}/journal/entries/${entryId}`).then((r) => r.data),
  });
}

export function useDeleteJournalEntry() {
  const qc = useQueryClient();
  return useMutation<unknown, unknown, { entryId: string; reason?: string }>({
    mutationFn: ({ entryId, reason }) =>
      api.delete(`${BASE}/journal/entries/${entryId}`, {
        params: { reason: reason || undefined },
      }).then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['console', 'journal'] });
      qc.invalidateQueries({ queryKey: ['console', 'overview'] });
    },
  });
}

export interface ReindexResult { reindexed: number; failed: number; remaining: number }

/** The reindex queue (entries with missing vectors). Polls for a live view. */
export function useJournalPending(enabled = true, pollMs = 4000) {
  return useQuery<{ entries: JournalEntry[]; total: number }>({
    queryKey: ['console', 'journal', 'pending'],
    enabled,
    refetchInterval: enabled ? pollMs : false,
    refetchOnWindowFocus: true,
    queryFn: () => api.get(`${BASE}/journal/pending`, { params: { limit: 200 } }).then((r) => r.data),
  });
}

export function useReindexPending() {
  const qc = useQueryClient();
  return useMutation<ReindexResult, unknown, void>({
    mutationFn: () => api.post<ReindexResult>(`${BASE}/journal/reindex`).then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['console', 'journal', 'pending'] });
      qc.invalidateQueries({ queryKey: ['console', 'overview'] });
    },
  });
}

// ---------------------------- ACE playbook ---------------------------------- //

export interface PlaybookBullet {
  id: number;
  bullet_key: string;
  owner_user_id: string;
  project_slug: string;
  content: string;
  section: string;
  domain?: string | null;
  helpful: number;
  harmful: number;
  score?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export function usePlaybookBullets(params: {
  owner_user_id?: string | null;
  project_slug?: string | null;
}, enabled = true) {
  return useQuery<{ bullets: PlaybookBullet[]; total: number }>({
    queryKey: ['console', 'playbook', params],
    enabled,
    queryFn: () =>
      api.get(`${BASE}/playbook/bullets`, {
        params: {
          owner_user_id: params.owner_user_id || undefined,
          project_slug: params.project_slug || undefined,
        },
      }).then((r) => r.data),
  });
}

export function useDeletePlaybookBullet() {
  const qc = useQueryClient();
  return useMutation<unknown, unknown, { id: number; owner_user_id: string }>({
    mutationFn: ({ id, owner_user_id }) =>
      api.delete(`${BASE}/playbook/bullets/${id}`, { params: { owner_user_id } }).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['console', 'playbook'] }),
  });
}

export function useInventory(params: {
  owner_user_id?: string | null;
  host?: string | null;
  project?: string | null;
  detail?: boolean;
}, enabled = true) {
  return useQuery<InventoryResponse>({
    queryKey: ['console', 'inventory', params],
    enabled,
    queryFn: () =>
      api.get<InventoryResponse>(`${BASE}/index/inventory`, {
        params: {
          owner_user_id: params.owner_user_id || undefined,
          host: params.host || undefined,
          project: params.project || undefined,
          detail: params.detail || undefined,
        },
      }).then((r) => r.data),
  });
}
