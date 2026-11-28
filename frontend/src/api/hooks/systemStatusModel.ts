/**
 * Pure health-derivation for the status capsule / health dot (frontend
 * refactor F1, §2.3). Kept separate from useSystemStatus.ts so the
 * documented rules are unit-testable without mounting react-query:
 *   - Qdrant unreachable            -> down
 *   - vector_delta != 0             -> warn
 *   - optimizer status isn't 'ok'   -> warn
 *   - disk free ratio < 10%         -> warn
 *   - latency above LATENCY_NORMAL_MS -> warn
 *   - none of the above             -> ok
 *
 * `LATENCY_NORMAL_MS` is a fixed threshold, not derived from the recent
 * sample window: a self-derived band (e.g. mean ± 1sd of the last N probes)
 * drifts with the data, so a search backend that is *persistently* slow
 * would read as "normal" forever. The retrieval sparkline in StatusCapsule
 * uses this same constant for its band, so the dot and the chart never
 * disagree about what counts as slow.
 */

export type HealthLevel = 'ok' | 'warn' | 'down';

/** Search-probe latency, ms, above which the retrieval service reads as slow. */
export const LATENCY_NORMAL_MS = 100;

export interface HealthInputs {
  qdrantReachable: boolean;
  /** qdrant.points - sql.chunks, or null when unknown (treated as no signal). */
  vectorDelta: number | null;
  /** Qdrant's `optimizer_status` — 'ok' when healthy, an object/string otherwise. */
  optimizerStatus: unknown;
  /** free / total disk bytes on the data volume, or null when unknown. */
  diskFreeRatio: number | null;
  /** latest search-probe round-trip time, ms, or null when unknown. */
  latencyMs: number | null;
}

export function deriveHealthLevel(inputs: HealthInputs): HealthLevel {
  if (!inputs.qdrantReachable) return 'down';

  const deltaWarn = inputs.vectorDelta != null && inputs.vectorDelta !== 0;
  const optimizerWarn = inputs.optimizerStatus != null && inputs.optimizerStatus !== 'ok';
  const diskWarn = inputs.diskFreeRatio != null && inputs.diskFreeRatio < 0.10;
  const latencyWarn = inputs.latencyMs != null && inputs.latencyMs > LATENCY_NORMAL_MS;

  return deltaWarn || optimizerWarn || diskWarn || latencyWarn ? 'warn' : 'ok';
}

/** The same rules as `deriveHealthLevel`, each firing rule named — shown
 * next to the health state so a warning always says what is wrong. */
export function healthReasons(inputs: HealthInputs): string[] {
  if (!inputs.qdrantReachable) return ['Qdrant 不可达'];
  const reasons: string[] = [];
  if (inputs.vectorDelta != null && inputs.vectorDelta !== 0) {
    reasons.push(inputs.vectorDelta > 0
      ? `向量比分块多 ${inputs.vectorDelta.toLocaleString()}`
      : `向量比分块少 ${Math.abs(inputs.vectorDelta).toLocaleString()}`);
  }
  if (inputs.optimizerStatus != null && inputs.optimizerStatus !== 'ok') reasons.push('Qdrant 优化器异常');
  if (inputs.diskFreeRatio != null && inputs.diskFreeRatio < 0.10) {
    reasons.push(`磁盘可用不足 10%（${(inputs.diskFreeRatio * 100).toFixed(1)}%）`);
  }
  if (inputs.latencyMs != null && inputs.latencyMs > LATENCY_NORMAL_MS) {
    reasons.push(`检索延迟 ${Math.round(inputs.latencyMs)} ms，超过 ${LATENCY_NORMAL_MS} ms`);
  }
  return reasons;
}

export const HEALTH_LABEL: Record<HealthLevel, string> = { ok: '正常', warn: '有告警', down: '不可用' };
