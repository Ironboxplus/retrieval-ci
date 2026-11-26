import { useQuery } from '@tanstack/react-query';
import api from '../client';
import { useAuthenticatedApiEnabled } from './useAuth';

export interface OverviewResp {
  ts: number;
  uptime_s: number;
  sql: { papers: number; documents: number; chunks: number };
  qdrant: {
    status?: 'green' | 'yellow' | 'red' | 'grey';
    points?: number;
    indexed_vectors?: number;
    segments?: number;
    optimizer_status?: unknown;
    error?: string;
  };
  coverage: number | null;
  /** qdrant.points - sql.chunks. null when Qdrant is unreachable. F1-b. */
  vector_delta: number | null;
}

export interface QdrantDetailResp {
  status: 'green' | 'yellow' | 'red' | 'grey';
  points_count: number;
  indexed_vectors_count: number;
  segments_count: number;
  vectors_count: number | null;
  optimizer_status: unknown;
  payload_schema: Record<string, { type: string; points: number }>;
  hnsw_config: {
    m: number;
    ef_construct: number;
    full_scan_threshold?: number;
    max_indexing_threads?: number;
    on_disk?: boolean;
  } | null;
  // W39: per-vector dense HNSW override 合并顶层 default。多 named-vector
  // collection 下顶层 hnsw_config 仅作 fallback；effective_hnsw_config
  // 才是真正在用的配置（segment 创建时定型）。
  effective_hnsw_config: {
    m?: number;
    ef_construct?: number;
    full_scan_threshold?: number;
    max_indexing_threads?: number;
    on_disk?: boolean;
  } | null;
  optimizer_config: {
    deleted_threshold?: number;
    vacuum_min_vector_number?: number;
    default_segment_number?: number;
    max_segment_size?: number | null;
    memmap_threshold?: number;
    indexing_threshold?: number;
    flush_interval_sec?: number;
    max_optimization_threads?: number;
  } | null;
  quantization: {
    scalar?: { type?: string; quantile?: number; always_ram?: boolean };
  } | null;
  vectors: {
    dense_on_disk: boolean | null;
    sparse_on_disk: boolean | null;
    payload_on_disk: boolean | null;
  };
  search_probe_ms: number | null;
  search_probe_error?: string;
}

export interface SqliteDetailResp {
  db_path: string;
  db_size_bytes: number;
  wal_size_bytes: number;
  shm_size_bytes: number;
  tables: { name: string; rows: number }[];
  indexes: { name: string; table: string }[];
  pragma: {
    journal_mode: string;
    synchronous: number;
    page_size: number;
    cache_size: number;
  };
}

export interface DistributionResp {
  chunk_levels: Record<string, number>;
  venues: { venue: string; papers: number }[];
  years: { year: number; papers: number }[];
  venue_year: { venue: string; year: number; papers: number }[];
  data_sources: { data_source: string; documents: number }[];
}

export interface SystemResp {
  process_uptime_s: number;
  ts: number;
  fastapi: { alive: boolean; uptime_s: number };
  qdrant: { alive: boolean; ping_ms?: number; status_code?: number; error?: string };
  disk: { path: string; bytes: number; exists: boolean }[];
  filesystem: {
    free_bytes?: number;
    total_bytes?: number;
    used_bytes?: number;
    error?: string;
  };
}

export interface ActivityResp {
  active: { running: number; pending: number };
  recent_tasks: {
    id: number;
    task_type: string;
    status: string;
    progress: number | null;
    total: number | null;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    last_heartbeat: string | null;
  }[];
}

const FAST = 10_000; // ops-grade refresh: 10s for live counters
const SLOW = 60_000; // 1min for the SQLite detail card
const DISTRIBUTION_REFRESH = 600_000; // 10min: matches the backend stale-while-revalidate TTL of the heavy GROUP BYs

export function useOverview() {
  return useQuery({
    queryKey: ['dash', 'overview'],
    queryFn: () => api.get<OverviewResp>('/dashboard/overview').then((r) => r.data),
    refetchInterval: FAST,
  });
}

export function useQdrantDetail() {
  return useQuery({
    queryKey: ['dash', 'qdrant'],
    queryFn: () => api.get<QdrantDetailResp>('/dashboard/qdrant').then((r) => r.data),
    refetchInterval: FAST,
  });
}

export function useSqliteDetail() {
  return useQuery({
    queryKey: ['dash', 'sqlite'],
    queryFn: () => api.get<SqliteDetailResp>('/dashboard/sqlite').then((r) => r.data),
    refetchInterval: SLOW,
  });
}

export function useDistribution() {
  const enabled = useAuthenticatedApiEnabled();
  return useQuery({
    queryKey: ['dash', 'distribution'],
    queryFn: () => api.get<DistributionResp>('/dashboard/distribution').then((r) => r.data),
    enabled,
    refetchInterval: DISTRIBUTION_REFRESH,
  });
}

export function useSystem() {
  return useQuery({
    queryKey: ['dash', 'system'],
    queryFn: () => api.get<SystemResp>('/dashboard/system').then((r) => r.data),
    refetchInterval: FAST,
  });
}

export function useActivity(limit = 20) {
  return useQuery({
    queryKey: ['dash', 'activity', limit],
    queryFn: () => api.get<ActivityResp>(`/dashboard/activity?limit=${limit}`).then((r) => r.data),
    refetchInterval: FAST,
  });
}
