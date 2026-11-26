import { useMemo } from 'react';

import { useConsoleAccess } from './useAdminConsole';
import { useOverview, useQdrantDetail, useSystem, useActivity } from './useDashboard';
import { useStats } from './useStats';
import { deriveHealthLevel, healthReasons, type HealthInputs, type HealthLevel } from './systemStatusModel';

/**
 * The single data source for the status capsule and its four MorphCards
 * (frontend refactor F1, §2.3). Internal build or public admin: pulls the
 * `/dashboard/*` operations endpoints (full detail). Public non-admin: only
 * the public `/stats` endpoint — no admin-only dashboard calls fire for them.
 *
 * `Search.tsx`'s corpus stats line reads this same hook instead of running
 * its own `useStats()` poll, so there is exactly one live query for "how
 * many papers" on screen at once.
 */
export interface SystemStatusData {
  hasDetail: boolean;
  health: HealthLevel;
  /** Each health rule currently firing, named with its value. */
  healthReasons: string[];
  totalPapers: number | null;
  totalDocuments: number | null;
  totalChunks: number | null;
  vectorPoints: number | null;
  vectorDelta: number | null;
  runningTasks: number | null;
  qdrantAlive: boolean | null;
  searchProbeMs: number | null;
  diskFreeBytes: number | null;
  diskTotalBytes: number | null;
  segments: number | null;
  indexedVectors: number | null;
  optimizerStatus: unknown;
  processUptimeS: number | null;
}

export function useSystemStatus() {
  const hasDetail = useConsoleAccess();

  const overview = useOverview();
  const qdrant = useQdrantDetail();
  const system = useSystem();
  const activity = useActivity(10);
  const stats = useStats();

  const data: SystemStatusData | undefined = useMemo(() => {
    if (hasDetail) {
      if (!overview.data) return undefined;
      const fs = system.data?.filesystem;
      const diskFreeRatio =
        fs?.free_bytes != null && fs?.total_bytes ? fs.free_bytes / fs.total_bytes : null;
      const inputs: HealthInputs = {
        qdrantReachable: !('error' in overview.data.qdrant),
        vectorDelta: overview.data.vector_delta,
        optimizerStatus: overview.data.qdrant.optimizer_status,
        diskFreeRatio,
        latencyMs: qdrant.data?.search_probe_ms ?? null,
      };
      return {
        hasDetail: true,
        health: deriveHealthLevel(inputs),
        healthReasons: healthReasons(inputs),
        totalPapers: overview.data.sql.papers,
        totalDocuments: overview.data.sql.documents,
        totalChunks: overview.data.sql.chunks,
        vectorPoints: 'points' in overview.data.qdrant ? overview.data.qdrant.points ?? null : null,
        vectorDelta: overview.data.vector_delta,
        runningTasks: activity.data?.active.running ?? null,
        qdrantAlive: system.data?.qdrant.alive ?? null,
        searchProbeMs: qdrant.data?.search_probe_ms ?? null,
        diskFreeBytes: fs?.free_bytes ?? null,
        diskTotalBytes: fs?.total_bytes ?? null,
        segments: qdrant.data?.segments_count ?? null,
        indexedVectors: qdrant.data?.indexed_vectors_count ?? null,
        optimizerStatus: qdrant.data?.optimizer_status ?? null,
        processUptimeS: system.data?.process_uptime_s ?? null,
      };
    }
    if (!stats.data) return undefined;
    return {
      hasDetail: false,
      health: deriveHealthLevel({
        qdrantReachable: true, // /stats only returns a body when the search pipeline is up
        vectorDelta: null,
        optimizerStatus: null,
        diskFreeRatio: null,
        latencyMs: null,
      }),
      healthReasons: [],
      totalPapers: stats.data.total_papers ?? null,
      totalDocuments: stats.data.total_documents,
      totalChunks: stats.data.total_chunks,
      vectorPoints: stats.data.vector_index_size,
      vectorDelta: null,
      runningTasks: null,
      qdrantAlive: null,
      searchProbeMs: null,
      diskFreeBytes: null,
      diskTotalBytes: null,
      segments: null,
      indexedVectors: null,
      optimizerStatus: null,
      processUptimeS: null,
    };
  }, [hasDetail, overview.data, qdrant.data, system.data, activity.data, stats.data]);

  return {
    data,
    isLoading: hasDetail ? overview.isLoading : stats.isLoading,
  };
}
