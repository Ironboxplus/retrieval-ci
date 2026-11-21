import { useCallback, useEffect, useReducer } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, LayoutGroup, motion, useReducedMotion } from 'framer-motion';
import { X } from 'lucide-react';

import { Button } from '@/components/ui/button';

import { morphReducer, initialMorphState } from './morphState';

export interface MorphGridProps<T> {
  items: T[];
  getKey: (item: T) => string;
  renderCompact: (item: T) => ReactNode;
  renderExpanded: (item: T, close: () => void) => ReactNode;
  /** CSS grid-template-columns for the compact grid. */
  columns?: string;
  emptyState?: ReactNode;
  /** Open this card already-expanded on mount (e.g. a `?selected=` deep
   * link). Only consulted on the initial render. */
  initialExpandedKey?: string | null;
  /**
   * 'overlay' (default): expand into a centered, portaled modal over a
   * dimmed backdrop — for boards where the grid fills the page (Tasks,
   * public-code sources) and an expanded card reasonably covers it.
   * 'inline': expand IN PLACE of the grid, inside the same parent box — no
   * portal, no second backdrop. For a small anchored panel (the status
   * capsule) a competing full-viewport layer over the still-visible panel
   * behind it reads as two overlapping UIs, not one that grew; inline swaps
   * the grid for the single expanded card within that one box, still using
   * the same shared `layoutId` for the morph.
   */
  variant?: 'overlay' | 'inline';
}

/**
 * Compact-grid-of-cards with click-to-morph-into-one-big-card (frontend
 * refactor F1, §2.1). Only one card is expanded at a time (morphState.ts);
 * Esc, the backdrop or the close button all collapse it. Falls back to a
 * plain opacity fade when the OS requests reduced motion.
 */
export default function MorphGrid<T>({
  items, getKey, renderCompact, renderExpanded, columns, emptyState, initialExpandedKey,
  variant = 'overlay',
}: MorphGridProps<T>) {
  const [state, dispatch] = useReducer(
    morphReducer<string>,
    initialExpandedKey ?? null,
    initialMorphState<string>,
  );
  const reduceMotion = useReducedMotion();
  const close = useCallback(() => dispatch({ type: 'close' }), []);

  useEffect(() => {
    if (state.expandedKey === null) return;
    const onKey = (e: KeyboardEvent) => dispatch({ type: 'keydown', code: e.code });
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [state.expandedKey]);

  const expandedItem = items.find((it) => getKey(it) === state.expandedKey);
  const layoutId = (key: string) => (reduceMotion ? undefined : `morph-card-${key}`);

  if (items.length === 0 && emptyState) return <>{emptyState}</>;

  const grid = (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: columns ?? 'repeat(auto-fill, minmax(240px, 1fr))',
        gap: 12,
      }}
    >
      {items.map((item) => {
        const key = getKey(item);
        return (
          <motion.div
            key={key}
            layoutId={layoutId(key)}
            onClick={() => dispatch({ type: 'select', key })}
            style={{ cursor: 'pointer', minWidth: 0 }}
            whileHover={reduceMotion ? undefined : { y: -2 }}
            transition={{ duration: reduceMotion ? 0 : 0.2 }}
          >
            {renderCompact(item)}
          </motion.div>
        );
      })}
    </div>
  );

  if (variant === 'inline') {
    // No portal, no second backdrop: the grid and the expanded card never
    // coexist as separate layers — whichever is current occupies this one
    // box, and `layout` on the wrapper lets the box itself resize to fit.
    return (
      <LayoutGroup>
        <motion.div layout transition={{ duration: reduceMotion ? 0 : 0.2 }}>
          {expandedItem ? (
            <motion.div layoutId={layoutId(state.expandedKey as string)} style={{ position: 'relative' }}>
              <Button variant="ghost" size="icon" onClick={close} aria-label="关闭" className="absolute right-2 top-2 z-10">
                <X className="size-4" />
              </Button>
              {renderExpanded(expandedItem, close)}
            </motion.div>
          ) : grid}
        </motion.div>
      </LayoutGroup>
    );
  }

  return (
    <LayoutGroup>
      {grid}

      {/* Portaled to document.body: a `position: fixed` element confined to
          an ancestor with its own transform/filter/backdrop-filter/
          will-change becomes contained by THAT ancestor's box instead of
          the viewport (CSS spec, not a bug in this component) — e.g.
          Layout.tsx's sticky header uses `backdropFilter` for its glass
          effect, which silently clipped this overlay to the header's
          56px-tall box for any MorphGrid mounted under it (the status
          capsule). Escaping to body-level sidesteps every such ancestor,
          present or future, everywhere MorphGrid is used. */}
      {createPortal(
        <AnimatePresence>
          {expandedItem && (
            <motion.div
              key="morph-overlay-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: reduceMotion ? 0.12 : 0.2 }}
              onClick={close}
              style={{
                position: 'fixed', inset: 0, zIndex: 1000,
                background: 'rgba(0,0,0,0.45)',
                display: 'grid', placeItems: 'center', padding: 24,
              }}
            >
              <motion.div
                layoutId={layoutId(state.expandedKey as string)}
                onClick={(e) => e.stopPropagation()}
                transition={{ duration: reduceMotion ? 0.12 : undefined }}
                style={{
                  position: 'relative',
                  width: 'min(720px, 92vw)',
                  maxHeight: '85vh',
                  overflow: 'auto',
                  borderRadius: 12,
                }}
                className="border bg-card text-card-foreground shadow-2xl"
              >
                <Button variant="ghost" size="icon" onClick={close} aria-label="关闭" className="absolute right-2 top-2 z-10">
                  <X className="size-4" />
                </Button>
                {renderExpanded(expandedItem, close)}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body,
      )}
    </LayoutGroup>
  );
}
