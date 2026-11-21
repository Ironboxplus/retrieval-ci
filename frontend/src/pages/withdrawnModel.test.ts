import { describe, expect, it } from 'vitest';

import {
  includeWithdrawnField,
  readIncludeWithdrawn,
  withdrawnMarks,
  writeIncludeWithdrawn,
} from './withdrawnModel';

describe('withdrawnMarks', () => {
  it('marks a withdrawn arXiv paper with its content version and time', () => {
    expect(withdrawnMarks({ withdrawn: true, source_version: 'v1', withdrawn_at: '2026-11-01T03:00:00' }))
      .toEqual({ withdrawn: true, version: 'v1', versionLabel: '正文版本 v1', withdrawnAt: '2026-11-01T03:00:00' });
  });

  it('keeps the version of a live arXiv paper without a withdrawn mark', () => {
    expect(withdrawnMarks({ withdrawn: false, source_version: 'v2', withdrawn_at: null }))
      .toEqual({ withdrawn: false, version: 'v2', versionLabel: '正文版本 v2', withdrawnAt: null });
  });

  it('shows nothing for a conference paper', () => {
    expect(withdrawnMarks({ withdrawn: false })).toEqual({ withdrawn: false, version: null, versionLabel: null, withdrawnAt: null });
    expect(withdrawnMarks(undefined)).toEqual({ withdrawn: false, version: null, versionLabel: null, withdrawnAt: null });
    expect(withdrawnMarks({})).toEqual({ withdrawn: false, version: null, versionLabel: null, withdrawnAt: null });
  });

  it('only a literal true counts as withdrawn', () => {
    for (const value of ['true', 1, 'yes', null]) {
      expect(withdrawnMarks({ withdrawn: value }).withdrawn).toBe(false);
    }
  });

  it('ignores blank version strings and a withdrawn time on a live paper', () => {
    expect(withdrawnMarks({ source_version: '  ' }).versionLabel).toBeNull();
    expect(withdrawnMarks({ withdrawn: false, withdrawn_at: '2026-11-01T03:00:00' }).withdrawnAt).toBeNull();
  });
});

describe('search URL state', () => {
  it('is off by default and only "1" turns it on', () => {
    expect(readIncludeWithdrawn(new URLSearchParams(''))).toBe(false);
    expect(readIncludeWithdrawn(new URLSearchParams('withdrawn=0'))).toBe(false);
    expect(readIncludeWithdrawn(new URLSearchParams('withdrawn=1'))).toBe(true);
  });

  it('writes nothing when off and round-trips when on', () => {
    const off = new URLSearchParams('q=x');
    writeIncludeWithdrawn(off, false);
    expect(off.toString()).toBe('q=x');
    const on = new URLSearchParams('q=x');
    writeIncludeWithdrawn(on, true);
    expect(on.toString()).toBe('q=x&withdrawn=1');
    expect(readIncludeWithdrawn(on)).toBe(true);
  });
});

describe('includeWithdrawnField', () => {
  it('is sent only for arxiv and only when on', () => {
    expect(includeWithdrawnField('arxiv', true)).toBe(true);
    expect(includeWithdrawnField('arxiv', false)).toBeUndefined();
    expect(includeWithdrawnField('paper', true)).toBeUndefined();
    expect(includeWithdrawnField('paper', false)).toBeUndefined();
    expect(includeWithdrawnField('upload', true)).toBeUndefined();
    expect(includeWithdrawnField('webpage', true)).toBeUndefined();
    expect(includeWithdrawnField('all', true)).toBeUndefined();
  });
});
