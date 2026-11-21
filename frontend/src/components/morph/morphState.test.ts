import { describe, expect, it } from 'vitest';

import { morphReducer, initialMorphState, type MorphState } from './morphState';

describe('initialMorphState', () => {
  it('starts with nothing expanded', () => {
    expect(initialMorphState<string>()).toEqual({ expandedKey: null });
  });

  it('seeds an already-expanded key for deep links', () => {
    expect(initialMorphState<string>('task-42')).toEqual({ expandedKey: 'task-42' });
  });
});

describe('morphReducer', () => {
  it('select opens a card from the closed state', () => {
    const s0 = initialMorphState<string>();
    const s1 = morphReducer(s0, { type: 'select', key: 'a' });
    expect(s1.expandedKey).toBe('a');
  });

  it('selecting the open card again closes it (toggle)', () => {
    const s0: MorphState<string> = { expandedKey: 'a' };
    const s1 = morphReducer(s0, { type: 'select', key: 'a' });
    expect(s1.expandedKey).toBeNull();
  });

  it('selecting a different card swaps the expansion — never two at once', () => {
    const s0: MorphState<string> = { expandedKey: 'a' };
    const s1 = morphReducer(s0, { type: 'select', key: 'b' });
    expect(s1.expandedKey).toBe('b');
  });

  it('close collapses whatever is open', () => {
    const s0: MorphState<string> = { expandedKey: 'a' };
    expect(morphReducer(s0, { type: 'close' }).expandedKey).toBeNull();
  });

  it('close on an already-closed state is a no-op (same reference)', () => {
    const s0 = initialMorphState<string>();
    expect(morphReducer(s0, { type: 'close' })).toBe(s0);
  });

  it('Escape keydown closes an open card', () => {
    const s0: MorphState<string> = { expandedKey: 'a' };
    expect(morphReducer(s0, { type: 'keydown', code: 'Escape' }).expandedKey).toBeNull();
  });

  it('a non-Escape keydown is a no-op', () => {
    const s0: MorphState<string> = { expandedKey: 'a' };
    const s1 = morphReducer(s0, { type: 'keydown', code: 'Enter' });
    expect(s1).toBe(s0);
  });

  it('Escape on a closed state is a no-op', () => {
    const s0 = initialMorphState<string>();
    expect(morphReducer(s0, { type: 'keydown', code: 'Escape' })).toBe(s0);
  });

  it('works with non-string keys (numbers)', () => {
    const s0 = initialMorphState<number>();
    const s1 = morphReducer(s0, { type: 'select', key: 42 });
    expect(s1.expandedKey).toBe(42);
    expect(morphReducer(s1, { type: 'select', key: 42 }).expandedKey).toBeNull();
  });
});
