import { describe, expect, it } from 'vitest';

import { canQueryAuthenticatedApi } from './useAuth';


describe('canQueryAuthenticatedApi', () => {
  it('keeps trusted internal queries enabled without a gateway session', () => {
    expect(canQueryAuthenticatedApi(false, false, undefined)).toBe(true);
  });

  it('waits for the public authentication query to resolve', () => {
    expect(canQueryAuthenticatedApi(true, false, undefined)).toBe(false);
  });

  it('keeps authenticated public APIs disabled for anonymous users', () => {
    expect(canQueryAuthenticatedApi(true, true, null)).toBe(false);
  });

  it('enables authenticated public APIs after a user is resolved', () => {
    expect(canQueryAuthenticatedApi(true, true, { id: 1 })).toBe(true);
  });
});
