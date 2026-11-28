import type { PublicCodeSource } from '../api/hooks/usePublicCode';

export interface PublicCodeSummary {
  configuredSources: number;
  indexedSources: number;
  scheduledSources: number;
  sourcesWithErrors: number;
  files: number;
  chunks: number;
  vectors: number;
}

export function buildPublicCodeSummary(sources: PublicCodeSource[]): PublicCodeSummary {
  return sources.reduce<PublicCodeSummary>((summary, source) => {
    summary.configuredSources += 1;
    summary.indexedSources += source.active_revision ? 1 : 0;
    summary.scheduledSources += source.update_policy === 'scheduled' ? 1 : 0;
    summary.sourcesWithErrors += source.last_check_error ? 1 : 0;
    summary.files += source.active_manifest_file_count;
    summary.chunks += source.active_chunk_count;
    summary.vectors += source.active_vector_count;
    return summary;
  }, {
    configuredSources: 0,
    indexedSources: 0,
    scheduledSources: 0,
    sourcesWithErrors: 0,
    files: 0,
    chunks: 0,
    vectors: 0,
  });
}

export function filterPublicCodeSources(sources: PublicCodeSource[], query: string): PublicCodeSource[] {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  if (!normalizedQuery) return sources;
  return sources.filter((source) => [
    source.source_id,
    source.domain,
    source.relative_path,
    source.upstream_url,
    source.active_revision,
    source.last_mirrored_revision,
    source.last_remote_revision,
  ].some((value) => value?.toLocaleLowerCase().includes(normalizedQuery)));
}

export type SourceStateKey = 'checkFailed' | 'unconfigured' | 'hasUpdate' | 'active';

/**
 * A source's state as a single mutually-exclusive key (frontend refactor F1
 * design addendum, §1.1) — this is what makes `summarizeSourceStates` below
 * a true partition, so its counts can be drawn as ONE proportional segmented
 * bar (segment widths sum to the total) instead of a row of KPI tiles whose
 * numbers silently overlap (a source can be both "indexed" and "scheduled").
 */
export function sourceStateKey(source: PublicCodeSource): SourceStateKey {
  if (source.last_check_error) return 'checkFailed';
  if (!source.active_revision) return 'unconfigured';
  if (source.last_remote_revision && source.last_remote_revision !== source.active_revision) {
    return 'hasUpdate';
  }
  return 'active';
}

export function summarizeSourceStates(
  sources: PublicCodeSource[],
): Record<SourceStateKey, number> {
  const counts: Record<SourceStateKey, number> = {
    checkFailed: 0, unconfigured: 0, hasUpdate: 0, active: 0,
  };
  for (const source of sources) counts[sourceStateKey(source)] += 1;
  return counts;
}

export interface RevisionGaps {
  /** 1 when 镜像 has moved since 生效 last promoted (needs promotion). */
  activeMirrorGap: 0 | 1;
  /** 1 when 上游 has commits not yet mirrored (needs a refresh). */
  mirrorRemoteGap: 0 | 1;
}

/**
 * Structural staleness for the compact source card's three-dot revision line
 * (生效 ← 镜像 ← 上游, design addendum §1.1): each gap is a binary "did this
 * pair of revisions diverge", not a fabricated continuous distance between
 * two git hashes — dot spacing in the card is these gaps times a fixed unit,
 * so two dots sit together when their revisions agree and separate the
 * instant they don't.
 */
export function revisionGaps(source: PublicCodeSource): RevisionGaps {
  return {
    activeMirrorGap: source.active_revision === source.last_mirrored_revision ? 0 : 1,
    mirrorRemoteGap: source.last_mirrored_revision === source.last_remote_revision ? 0 : 1,
  };
}

/** Largest `active_chunk_count` across all sources — the shared scale a
 * single source's expanded-card mini-bar is drawn against (design addendum:
 * "one proportional mini-bar sized against sibling sources", not a bare
 * "N 分块" number). 0 for an empty list so callers can safely divide without
 * a NaN/-Infinity guard. */
export function maxSourceChunkCount(sources: PublicCodeSource[]): number {
  return sources.reduce((max, s) => Math.max(max, s.active_chunk_count), 0);
}
