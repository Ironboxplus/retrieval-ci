import { describe, expect, it } from 'vitest';

import {
  DEFAULT_CODE_UI, readCodeUI, writeCodeUI, buildFederatedSearchRequest,
  timingsSegments, originLabel, shortRevision, originDistribution, scopeDistribution, pathTail,
  MAX_SELECTED_PROJECTS, toggleProjectSelection, isProjectCapped, isHostCapped,
  type CodeUIState,
} from './codeSearchModel';

describe('readCodeUI / writeCodeUI', () => {
  it('round-trips every field through the URL', () => {
    const ui: CodeUIState = {
      query: 'qdrant upsert', projects: ['C:/repo/a', 'C:/repo/b'],
      language: 'rust', pathGlob: 'src/**/*.rs', maxPerFile: 3, topK: 25,
      mode: 'keyword', rerank: true, includePublic: true,
      sourceIds: ['aiter'], domains: ['ml'],
    };
    const sp = writeCodeUI(ui);
    expect(readCodeUI(sp)).toEqual(ui);
  });

  it('reads all-default state from an empty URL', () => {
    expect(readCodeUI(new URLSearchParams())).toEqual(DEFAULT_CODE_UI);
  });

  it('does not write params that equal the default', () => {
    const sp = writeCodeUI(DEFAULT_CODE_UI);
    expect([...sp.keys()]).toEqual([]);
  });

  it('falls back to hybrid for an unknown cmode value', () => {
    const sp = new URLSearchParams('cmode=bogus');
    expect(readCodeUI(sp).mode).toBe('hybrid');
  });

  it('ignores a non-numeric cmax/ctopk rather than throwing', () => {
    const sp = new URLSearchParams('cmax=abc&ctopk=xyz');
    const ui = readCodeUI(sp);
    expect(ui.maxPerFile).toBeNull();
    expect(ui.topK).toBe(DEFAULT_CODE_UI.topK);
  });
});

describe('buildFederatedSearchRequest', () => {
  it('maps UI state to the request body', () => {
    const ui: CodeUIState = {
      ...DEFAULT_CODE_UI, query: 'q', projects: ['/repo'], language: 'go',
      pathGlob: 'src/', maxPerFile: 5, topK: 15, mode: 'dense', rerank: true,
    };
    const body = buildFederatedSearchRequest(ui);
    expect(body).toMatchObject({
      query: 'q', top_k: 15, mode: 'dense', rerank: true, language: 'go',
      projects: ['/repo'], path_glob: 'src/', max_per_file: 5,
    });
  });

  it('blanks language/path_glob to null instead of empty string', () => {
    const body = buildFederatedSearchRequest({ ...DEFAULT_CODE_UI, language: '  ', pathGlob: '' });
    expect(body.language).toBeNull();
    expect(body.path_glob).toBeNull();
  });

  it('strips source_ids/domains when includePublic is false — mirrors the backend 422 rule', () => {
    const body = buildFederatedSearchRequest({
      ...DEFAULT_CODE_UI, includePublic: false, sourceIds: ['aiter'], domains: ['ml'],
    });
    expect(body.include_public).toBe(false);
    expect(body.source_ids).toEqual([]);
    expect(body.domains).toEqual([]);
  });

  it('keeps source_ids/domains when includePublic is true', () => {
    const body = buildFederatedSearchRequest({
      ...DEFAULT_CODE_UI, includePublic: true, sourceIds: ['aiter'], domains: ['ml'],
    });
    expect(body.source_ids).toEqual(['aiter']);
    expect(body.domains).toEqual(['ml']);
  });
});

describe('timingsSegments', () => {
  it('is empty when total is null/zero', () => {
    expect(timingsSegments({})).toEqual([]);
    expect(timingsSegments({ total: 0, embed: 5 })).toEqual([]);
  });

  it('places each present phase end-to-end proportional to total', () => {
    const segs = timingsSegments({ embed: 10, retrieve: 20, fuse: null, rerank: null, total: 100 });
    expect(segs).toEqual([
      { key: 'embed', label: '编码', ms: 10, startPct: 0, widthPct: 10 },
      { key: 'retrieve', label: '检索', ms: 20, startPct: 10, widthPct: 20 },
    ]);
  });

  it('skips null and non-positive phases without gaps in the remaining sequence', () => {
    const segs = timingsSegments({ embed: 10, retrieve: null, fuse: 0, rerank: 5, total: 100 });
    expect(segs.map((s) => s.key)).toEqual(['embed', 'rerank']);
    expect(segs[1].startPct).toBe(10);
  });

  it('never lets the segments overrun 100% even if phases sum past total', () => {
    const segs = timingsSegments({ embed: 60, retrieve: 60, total: 100 });
    const end = segs[segs.length - 1].startPct + segs[segs.length - 1].widthPct;
    expect(end).toBeLessThanOrEqual(100);
  });
});

describe('toggleProjectSelection / isProjectCapped / isHostCapped', () => {
  it('ticks an empty selection to a single project', () => {
    expect(toggleProjectSelection([], 'p')).toEqual(['p']);
  });

  it('unticks a selected project back to empty', () => {
    expect(toggleProjectSelection(['p'], 'p')).toEqual([]);
  });

  it('adds a second project alongside an existing one', () => {
    expect(toggleProjectSelection(['a'], 'b')).toEqual(['a', 'b']);
  });

  it('is a no-op past the cap for a project not already selected', () => {
    const fifty = Array.from({ length: MAX_SELECTED_PROJECTS }, (_, i) => `p${i}`);
    expect(toggleProjectSelection(fifty, 'p50')).toBe(fifty);
  });

  it('still allows unticking one exactly at the cap', () => {
    const fifty = Array.from({ length: MAX_SELECTED_PROJECTS }, (_, i) => `p${i}`);
    const next = toggleProjectSelection(fifty, 'p0');
    expect(next).toHaveLength(MAX_SELECTED_PROJECTS - 1);
    expect(next).not.toContain('p0');
  });

  it('isProjectCapped: disabled only when the cap is reached and this one is not already selected', () => {
    expect(isProjectCapped(49, false)).toBe(false);
    expect(isProjectCapped(50, false)).toBe(true);
    expect(isProjectCapped(50, true)).toBe(false);
  });

  it('isHostCapped: a host whose own subtree exceeds the cap is always disabled', () => {
    expect(isHostCapped(60, 0, 0)).toBe(true);
  });

  it('isHostCapped: a host that fits under the cap given the current selection is allowed', () => {
    expect(isHostCapped(10, 30, 0)).toBe(false);
  });

  it('isHostCapped: a host that would push the total over the cap is disabled', () => {
    expect(isHostCapped(10, 45, 0)).toBe(true);
  });

  it('isHostCapped: projects already selected within the host do not count twice toward the new total', () => {
    // 10-project host, 45 already selected total, 5 of which are already
    // inside this host — ticking the remaining 5 new ones lands at exactly 50.
    expect(isHostCapped(10, 45, 5)).toBe(false);
  });
});

describe('shortRevision / originLabel', () => {
  it('shortens a revision to 10 chars, null for no revision', () => {
    expect(shortRevision('abcdef0123456789')).toBe('abcdef0123');
    expect(shortRevision(null)).toBeNull();
    expect(shortRevision(undefined)).toBeNull();
  });

  it('labels a private hit by its project', () => {
    expect(originLabel({ origin: 'private', project: 'C:/repo/aiter' })).toBe('C:/repo/aiter');
  });

  it('labels a public hit by source + short revision', () => {
    expect(originLabel({
      origin: 'public', source_id: 'aiter', revision: 'abcdef0123456789',
    })).toBe('aiter · abcdef0123');
  });

  it('omits the revision segment entirely when a public hit has none', () => {
    expect(originLabel({ origin: 'public', source_id: 'aiter', revision: null })).toBe('aiter');
  });
});

describe('pathTail', () => {
  it('keeps a short path unchanged', () => {
    expect(pathTail('a/b', 2)).toBe('a/b');
    expect(pathTail('a', 2)).toBe('a');
  });

  it('truncates to the last N segments with a leading ellipsis marker', () => {
    expect(pathTail('C:/Users/Arc/Prd/retrival/core/storage', 2)).toBe('…/core/storage');
    expect(pathTail('E:/JETB/pythons/playground/AI/LangGraphProjects', 2)).toBe('…/AI/LangGraphProjects');
  });

  it('tolerates backslash paths and a trailing slash', () => {
    expect(pathTail('C:\\Users\\Arc\\Prd\\retrival', 2)).toBe('…/Prd/retrival');
    expect(pathTail('a/b/c/', 2)).toBe('…/b/c');
  });

  it('honors a custom segment count', () => {
    expect(pathTail('a/b/c/d', 1)).toBe('…/d');
    expect(pathTail('a/b/c/d', 3)).toBe('…/b/c/d');
  });
});

describe('originDistribution / scopeDistribution', () => {
  const hits = [
    { origin: 'private' as const, project: 'a' },
    { origin: 'private' as const, project: 'a' },
    { origin: 'private' as const, project: 'b' },
    { origin: 'public' as const, source_id: 's1' },
  ];

  it('splits by origin with percentages summing to 100', () => {
    const dist = originDistribution(hits);
    expect(dist).toEqual([
      { key: 'private', label: '私有', count: 3, pct: 75 },
      { key: 'public', label: '公共', count: 1, pct: 25 },
    ]);
  });

  it('is empty for no hits', () => {
    expect(originDistribution([])).toEqual([]);
    expect(scopeDistribution([])).toEqual([]);
  });

  it('groups scopes by project (private) / source_id (public), sorted by count desc', () => {
    const dist = scopeDistribution(hits);
    expect(dist[0]).toEqual({ key: 'a', label: 'a', count: 2, pct: 50, origin: 'private' });
    expect(dist.map((d) => d.key)).toContain('b');
    expect(dist.map((d) => d.key)).toContain('s1');
  });

  it('keeps a private project and a public source of the same name in separate bars', () => {
    const collision = [
      { origin: 'private' as const, project: 'aiter' },
      { origin: 'public' as const, source_id: 'aiter' },
    ];
    const dist = scopeDistribution(collision);
    expect(dist).toHaveLength(2);
    expect(dist.find((d) => d.origin === 'private')).toEqual({ key: 'aiter', label: 'aiter', count: 1, pct: 50, origin: 'private' });
    expect(dist.find((d) => d.origin === 'public')).toEqual({ key: 'aiter', label: 'aiter', count: 1, pct: 50, origin: 'public' });
  });
});
