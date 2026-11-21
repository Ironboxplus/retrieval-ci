import { describe, expect, it } from 'vitest';

import {
  DEFAULT_UI, readUI, writeUI, uiToRequest, searchEndpoint, searchTrigger,
  ARXIV_SEARCH_ENDPOINT, PAPER_SEARCH_ENDPOINT,
  type SearchUIState,
} from './searchModel';
import { searchCacheKey } from '../api/hooks/useSearchStream';

const base: SearchUIState = { ...DEFAULT_UI, query: 'retrieval augmented generation' };

describe('readUI / writeUI source', () => {
  it('round-trips source=arxiv', () => {
    const ui: SearchUIState = { ...base, sourceType: 'arxiv', years: [2024, 2025] };
    const sp = writeUI(ui);
    expect(sp.get('source')).toBe('arxiv');
    expect(readUI(sp)).toEqual(ui);
  });

  it('keeps the paper default for an old URL without a source param', () => {
    const ui = readUI(new URLSearchParams('q=x&method=vector'));
    expect(ui.sourceType).toBe('paper');
    expect(writeUI(ui).has('source')).toBe(false);
  });
});

describe('uiToRequest', () => {
  it('arxiv sends year/title_only/include_withdrawn, no venue and no source_type', () => {
    const req = uiToRequest({
      ...base, sourceType: 'arxiv', venues: ['ICML'], years: [2025], titleOnly: true, includeWithdrawn: true,
    });
    expect(req.year).toEqual([2025]);
    expect(req.title_only).toBe(true);
    expect(req.include_withdrawn).toBe(true);
    expect(req.venue).toBeUndefined();
    expect(req.source_type).toBeUndefined();
    expect(req.include_preprints).toBeUndefined();
  });

  it('all sends include_preprints: true', () => {
    const req = uiToRequest({ ...base, sourceType: 'all' });
    expect(req.include_preprints).toBe(true);
    expect(req.source_type).toBeUndefined();
  });

  it('paper stays field-for-field what it was before the change', () => {
    const req = uiToRequest({
      ...base, sourceType: 'paper', venues: ['ICML'], years: [2024], titleOnly: true,
      includeWithdrawn: true, chunkMode: true, topK: 30, method: 'vector',
    });
    expect(req).toEqual({
      query: 'retrieval augmented generation', top_k: 30, method: 'vector',
      source_type: 'paper', venue: ['ICML'], year: [2024], title_only: true,
      include_withdrawn: undefined, dedup_by_paper: false,
    });
    expect((req).include_preprints).toBeUndefined();
  });

  it('upload and webpage still send neither venue, year nor title_only', () => {
    for (const sourceType of ['upload', 'webpage'] as const) {
      const req = uiToRequest({ ...base, sourceType, venues: ['ICML'], years: [2024], titleOnly: true });
      expect(req.source_type).toBe(sourceType);
      expect(req.venue).toBeUndefined();
      expect(req.year).toBeUndefined();
      expect(req.title_only).toBeUndefined();
    }
  });
});

describe('searchEndpoint / cache key', () => {
  it('routes arxiv to the arxiv stream and everything else to the default stream', () => {
    expect(searchEndpoint('arxiv')).toBe('/arxiv/search/stream');
    expect(ARXIV_SEARCH_ENDPOINT).toBe('/arxiv/search/stream');
    for (const s of ['all', 'paper', 'webpage', 'upload'] as const) {
      expect(searchEndpoint(s)).toBe('/search/stream');
    }
    expect(PAPER_SEARCH_ENDPOINT).toBe('/search/stream');
  });

  it('gives arxiv and paper different cache keys even for the same body', () => {
    const body = { query: 'x', top_k: 20 };
    expect(searchCacheKey(body, ARXIV_SEARCH_ENDPOINT)).not.toBe(searchCacheKey(body, PAPER_SEARCH_ENDPOINT));
  });

  it('keeps the default-endpoint key identical to the body-only key', () => {
    const body = { query: 'x', top_k: 20 };
    expect(searchCacheKey(body)).toBe(searchCacheKey(body, PAPER_SEARCH_ENDPOINT));
    expect(searchCacheKey(body)).toBe('{"query":"x","top_k":20}');
  });

  it('arxiv and paper ui states for the same query produce different cache keys', () => {
    const a = searchCacheKey(uiToRequest({ ...base, sourceType: 'arxiv' }), searchEndpoint('arxiv'));
    const p = searchCacheKey(uiToRequest({ ...base, sourceType: 'paper' }), searchEndpoint('paper'));
    expect(a).not.toBe(p);
  });
});

describe('searchTrigger', () => {
  it('is null for an empty or blank query', () => {
    expect(searchTrigger({ ...base, query: '' })).toBeNull();
    expect(searchTrigger({ ...base, query: '   ' })).toBeNull();
  });

  it('changes when any request-affecting control changes', () => {
    const t0 = searchTrigger(base);
    expect(t0).not.toBeNull();
    const changed: Partial<SearchUIState>[] = [
      { sourceType: 'arxiv' }, { sourceType: 'all' }, { method: 'sparse' }, { years: [2024] },
      { venues: ['ICML'] }, { titleOnly: true }, { chunkMode: true }, { topK: 50 },
    ];
    for (const delta of changed) {
      expect(searchTrigger({ ...base, ...delta }), JSON.stringify(delta)).not.toBe(t0);
    }
    expect(searchTrigger({ ...base, sourceType: 'arxiv' }))
      .not.toBe(searchTrigger({ ...base, sourceType: 'arxiv', includeWithdrawn: true }));
  });

  it('does not change when only sortBy changes', () => {
    expect(searchTrigger({ ...base, sortBy: 'year' })).toBe(searchTrigger(base));
    expect(searchTrigger({ ...base, sortBy: 'citations' })).toBe(searchTrigger(base));
  });

  it('does not change for a control the current source does not send', () => {
    // 论文来源下 includeWithdrawn 不进请求体，所以不应触发重查。
    expect(searchTrigger({ ...base, includeWithdrawn: true })).toBe(searchTrigger(base));
  });
});
