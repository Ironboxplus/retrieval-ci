import { describe, expect, it } from 'vitest';

import { resolveConsoleAccess } from './useAdminConsole';


describe('resolveConsoleAccess', () => {
  it('always allows the trusted internal console', () => {
    expect(resolveConsoleAccess(false, false, undefined)).toBe('allowed');
  });

  it('waits for public authentication before deciding', () => {
    expect(resolveConsoleAccess(true, false, undefined)).toBe('loading');
  });

  it('allows a resolved public admin', () => {
    expect(resolveConsoleAccess(true, true, { tier: 'admin' })).toBe('allowed');
  });

  it('denies anonymous and non-admin public users', () => {
    expect(resolveConsoleAccess(true, true, null)).toBe('denied');
    expect(resolveConsoleAccess(true, true, { tier: 'member' })).toBe('denied');
  });
});
