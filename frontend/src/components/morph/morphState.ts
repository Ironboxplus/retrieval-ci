/**
 * Pure expand/collapse state machine for MorphGrid (frontend refactor F1,
 * §2.1). Exactly one card can be expanded at a time; selecting the already-
 * expanded key collapses it back (toggle), selecting a different key swaps
 * the expansion straight to it. Kept as a plain reducer so it can be unit
 * tested without mounting framer-motion/antd.
 */

export interface MorphState<K> {
  expandedKey: K | null;
}

export type MorphAction<K> =
  | { type: 'select'; key: K }
  | { type: 'close' }
  | { type: 'keydown'; code: string };

/** `seedKey` lets a deep link (e.g. Tasks `?selected=<id>`) open a card
 * already expanded on first render, without any extra "open on mount"
 * effect — the reducer's initial state already reflects it. */
export function initialMorphState<K>(seedKey: K | null = null): MorphState<K> {
  return { expandedKey: seedKey };
}

export function morphReducer<K>(state: MorphState<K>, action: MorphAction<K>): MorphState<K> {
  switch (action.type) {
    case 'select':
      // Selecting the already-open card closes it; selecting any other card
      // replaces the open one — never two expanded at once.
      return { expandedKey: state.expandedKey === action.key ? null : action.key };
    case 'close':
      return state.expandedKey === null ? state : { expandedKey: null };
    case 'keydown':
      return action.code === 'Escape' && state.expandedKey !== null
        ? { expandedKey: null }
        : state;
    default:
      return state;
  }
}
