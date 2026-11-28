import { describe, expect, it } from 'vitest';

import { selectedTaskIdFromSearch } from './taskSelection';

describe('selectedTaskIdFromSearch', () => {
  it('accepts the positive task id emitted by Crawl task links', () => {
    expect(selectedTaskIdFromSearch('?selected=42')).toBe(42);
  });

  it('rejects missing, zero, negative and non-integer values', () => {
    expect(selectedTaskIdFromSearch('')).toBeUndefined();
    expect(selectedTaskIdFromSearch('?selected=0')).toBeUndefined();
    expect(selectedTaskIdFromSearch('?selected=-4')).toBeUndefined();
    expect(selectedTaskIdFromSearch('?selected=1.5')).toBeUndefined();
    expect(selectedTaskIdFromSearch('?selected=paper-42')).toBeUndefined();
  });
});
