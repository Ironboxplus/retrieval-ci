import { describe, expect, it } from 'vitest';

import {
  buildPublicCodeSummary,
  filterPublicCodeSources,
  sourceStateKey,
  summarizeSourceStates,
  revisionGaps,
  maxSourceChunkCount,
} from './publicCodeViewModel';
import {
  hasActivePublicCodeJobs,
  type PublicCodeJob,
  type PublicCodeSource,
} from '../api/hooks/usePublicCode';

const source = (overrides: Partial<PublicCodeSource>): PublicCodeSource => ({
  source_id: 'ffmpeg',
  domain: 'multimedia',
  relative_path: 'multimedia/ffmpeg',
  upstream_url: 'https://github.com/FFmpeg/FFmpeg.git',
  default_ref: 'HEAD',
  proxy_profile: null,
  credential_profile: null,
  language_overrides: {},
  source_origin: 'bootstrap',
  created_by_user_id: null,
  active_manifest_file_count: 100,
  active_chunk_count: 200,
  active_vector_count: 200,
  active_promoted_at: null,
  created_at: null,
  updated_at: null,
  update_policy: 'manual',
  update_interval_minutes: null,
  active_revision: 'abc123',
  last_mirrored_revision: 'abc123',
  last_remote_revision: 'abc123',
  last_check_error: null,
  last_checked_at: null,
  ...overrides,
});

describe('public code console view model', () => {
  it('summarizes active coverage without counting configured-only sources as indexed', () => {
    const summary = buildPublicCodeSummary([
      source({}),
      source({
        source_id: 'pion',
        domain: 'networking',
        active_revision: null,
        active_manifest_file_count: 0,
        active_chunk_count: 0,
        active_vector_count: 0,
        last_check_error: 'upstream unavailable',
      }),
    ]);

    expect(summary).toEqual({
      configuredSources: 2,
      indexedSources: 1,
      scheduledSources: 0,
      sourcesWithErrors: 1,
      files: 100,
      chunks: 200,
      vectors: 200,
    });
  });

  it('filters by source id, domain, path, URL, and revision without case sensitivity', () => {
    const sources = [
      source({}),
      source({
        source_id: 'Pion-WebRTC',
        domain: 'Networking',
        relative_path: 'networking/pion-webrtc',
        upstream_url: 'https://github.com/pion/webrtc.git',
        active_revision: 'DEF456',
      }),
    ];

    expect(filterPublicCodeSources(sources, 'webrtc')).toHaveLength(1);
    expect(filterPublicCodeSources(sources, 'NETWORKING')).toHaveLength(1);
    expect(filterPublicCodeSources(sources, 'def456')).toHaveLength(1);
    expect(filterPublicCodeSources(sources, '')).toHaveLength(2);
  });

  it('polls only while at least one job is queued or running', () => {
    const job = (status: PublicCodeJob['status']): PublicCodeJob => ({
      job_id: status,
      operation: 'index',
      source_ids: ['ffmpeg'],
      total_sources: 1,
      status,
      completed_sources: status === 'completed' ? 1 : 0,
      progress: status === 'completed' ? 1 : 0,
      phase: status,
      current_source_id: null,
      current_path: null,
      files_done: 0,
      files_total: 0,
      chunks_done: 0,
      message: null,
      source_progress: {},
      results: [],
      errors: {},
      created_at: 1,
      started_at: null,
      finished_at: null,
    });

    expect(hasActivePublicCodeJobs([job('completed'), job('failed')])).toBe(false);
    expect(hasActivePublicCodeJobs([job('completed'), job('queued')])).toBe(true);
  });

  describe('sourceStateKey', () => {
    it('is checkFailed when the last check errored, even if otherwise active', () => {
      expect(sourceStateKey(source({ last_check_error: 'timeout' }))).toBe('checkFailed');
    });

    it('is unconfigured when there is no active revision yet', () => {
      expect(sourceStateKey(source({ active_revision: null }))).toBe('unconfigured');
    });

    it('is hasUpdate when the remote has moved past the active revision', () => {
      expect(sourceStateKey(source({ active_revision: 'a', last_remote_revision: 'b' }))).toBe('hasUpdate');
    });

    it('is active when active and remote agree and there is no error', () => {
      expect(sourceStateKey(source({ active_revision: 'a', last_remote_revision: 'a' }))).toBe('active');
    });

    it('is a true partition — every source falls into exactly one state', () => {
      const sources = [
        source({ last_check_error: 'x' }),
        source({ active_revision: null }),
        source({ active_revision: 'a', last_remote_revision: 'b' }),
        source({ active_revision: 'a', last_remote_revision: 'a' }),
      ];
      const keys = new Set(sources.map(sourceStateKey));
      expect(keys).toEqual(new Set(['checkFailed', 'unconfigured', 'hasUpdate', 'active']));
    });
  });

  describe('summarizeSourceStates', () => {
    it('counts sum to the total source count', () => {
      const sources = [
        source({ last_check_error: 'x' }),
        source({ active_revision: null }),
        source({ active_revision: 'a', last_remote_revision: 'b' }),
        source({ active_revision: 'a', last_remote_revision: 'a' }),
        source({ active_revision: 'c', last_remote_revision: 'c' }),
      ];
      const counts = summarizeSourceStates(sources);
      expect(counts).toEqual({ checkFailed: 1, unconfigured: 1, hasUpdate: 1, active: 2 });
      expect(Object.values(counts).reduce((a, b) => a + b, 0)).toBe(sources.length);
    });

    it('is all zero for an empty source list', () => {
      expect(summarizeSourceStates([])).toEqual({ checkFailed: 0, unconfigured: 0, hasUpdate: 0, active: 0 });
    });
  });

  describe('revisionGaps', () => {
    it('has zero gaps when active, mirror, and remote all agree', () => {
      expect(revisionGaps(source({
        active_revision: 'a', last_mirrored_revision: 'a', last_remote_revision: 'a',
      }))).toEqual({ activeMirrorGap: 0, mirrorRemoteGap: 0 });
    });

    it('separates 镜像 from 生效 when the mirror has moved but active has not been promoted', () => {
      expect(revisionGaps(source({
        active_revision: 'a', last_mirrored_revision: 'b', last_remote_revision: 'b',
      }))).toEqual({ activeMirrorGap: 1, mirrorRemoteGap: 0 });
    });

    it('separates 上游 from 镜像 when the remote has new commits not yet mirrored', () => {
      expect(revisionGaps(source({
        active_revision: 'a', last_mirrored_revision: 'a', last_remote_revision: 'b',
      }))).toEqual({ activeMirrorGap: 0, mirrorRemoteGap: 1 });
    });

    it('separates both when active, mirror, and remote are all different', () => {
      expect(revisionGaps(source({
        active_revision: 'a', last_mirrored_revision: 'b', last_remote_revision: 'c',
      }))).toEqual({ activeMirrorGap: 1, mirrorRemoteGap: 1 });
    });

    it('treats a null revision as distinct from any real revision (never accidentally equal)', () => {
      expect(revisionGaps(source({
        active_revision: null, last_mirrored_revision: null, last_remote_revision: 'a',
      }))).toEqual({ activeMirrorGap: 0, mirrorRemoteGap: 1 });
    });
  });

  describe('maxSourceChunkCount', () => {
    it('is the largest active_chunk_count among sibling sources', () => {
      const sources = [
        source({ active_chunk_count: 200 }),
        source({ source_id: 'pion', active_chunk_count: 9000 }),
        source({ source_id: 'zlib', active_chunk_count: 50 }),
      ];
      expect(maxSourceChunkCount(sources)).toBe(9000);
    });

    it('is 0 for an empty source list, never NaN or -Infinity', () => {
      expect(maxSourceChunkCount([])).toBe(0);
    });
  });
});
