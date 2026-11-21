import { useCallback, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import type { SearchRequest, SearchResponse } from '../../types';

export type StreamStage = 'idle' | 'embed' | 'retrieve' | 'rerank' | 'enrich' | 'complete' | 'error';

export type StreamProgress = {
  embed_ms?: number;
  retrieve_ms?: number;
  rerank_ms?: number;
  retrieve_candidates?: number;
  rerank_candidates_before?: number;
  rerank_candidates_after?: number;
  rerank_skipped?: boolean;
  enriched_count?: number;
  mode?: string;
};

type State = {
  stage: StreamStage;
  progress: StreamProgress;
  data?: SearchResponse;
  error?: string;
};

const INITIAL: State = { stage: 'idle', progress: {} };

function getApiBase(): string {
  const v = (import.meta as unknown as { env?: { VITE_API_BASE?: string } }).env?.VITE_API_BASE;
  return v ?? '/api';
}

// Stable cache key from request shape. Sorted so {a,b} === {b,a}.
export function searchCacheKey(req: SearchRequest): string {
  const ordered = Object.keys(req).sort().reduce<Record<string, unknown>>((acc, k) => {
    const v = (req as unknown as Record<string, unknown>)[k];
    if (v !== undefined && v !== null && v !== '') acc[k] = v;
    return acc;
  }, {});
  return JSON.stringify(ordered);
}

/**
 * Streaming search hook with persistent cache via React Query.
 *
 * On mount, rehydrates from `queryClient.getQueryData(['search', cacheKey])`
 * so navigating away (e.g. to paper detail) and back keeps the prior result
 * snapshot without re-running the search. Pipeline metrics + progress are
 * derived from the cached `SearchResponse`, but live stream stages are only
 * shown for fresh requests.
 */
export function useSearchStream() {
  const qc = useQueryClient();
  const [state, setState] = useState<State>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);
  const lastKeyRef = useRef<string | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL);
    lastKeyRef.current = null;
  }, []);

  // Hydrate from cache. Returns true if a cached snapshot was found.
  const hydrate = useCallback((req: SearchRequest): boolean => {
    const key = searchCacheKey(req);
    const cached = qc.getQueryData<SearchResponse>(['search', key]);
    if (cached) {
      lastKeyRef.current = key;
      setState({ stage: 'complete', progress: {}, data: cached });
      return true;
    }
    return false;
  }, [qc]);

  const mutate = useCallback(async (req: SearchRequest) => {
    const key = searchCacheKey(req);
    lastKeyRef.current = key;

    // Same request as cached → skip network entirely.
    const cached = qc.getQueryData<SearchResponse>(['search', key]);
    if (cached) {
      setState({ stage: 'complete', progress: {}, data: cached });
      return;
    }

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setState({ stage: 'embed', progress: {} });

    try {
      const resp = await fetch(`${getApiBase()}/search/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
        signal: ctrl.signal,
      });

      // Paid-tier gate (gateway). 401 = not logged in, 402 = out of credits.
      // Surface as stable sentinels so Search can prompt login / topup.
      if (resp.status === 401) throw new Error('unauthorized');
      if (resp.status === 402) throw new Error('insufficient_credits');
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      const dispatch = (line: string) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        let ev: Record<string, unknown>;
        try {
          ev = JSON.parse(trimmed);
        } catch {
          return;
        }
        setState((prev) => {
          const next = apply(prev, ev);
          if (next.stage === 'complete' && next.data) {
            // Cache forever — same key = same result. Manual invalidate via reset/refresh.
            qc.setQueryData(['search', key], next.data);
          }
          return next;
        });
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) dispatch(line);
      }
      if (buf.trim()) dispatch(buf);
    } catch (err: unknown) {
      if ((err as { name?: string })?.name === 'AbortError') return;
      const message = err instanceof Error ? err.message : String(err);
      setState((s) => ({ ...s, stage: 'error', error: message }));
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
    }
  }, [qc]);

  const isPending = state.stage !== 'idle' && state.stage !== 'complete' && state.stage !== 'error';

  return { ...state, mutate, reset, isPending, hydrate, lastKey: lastKeyRef.current };
}

function apply(prev: State, ev: Record<string, unknown>): State {
  const stage = ev.stage as StreamStage | undefined;
  if (!stage) return prev;
  const p = { ...prev.progress };
  switch (stage) {
    case 'embed':
      p.embed_ms = ev.embed_ms as number;
      return { ...prev, stage: 'embed', progress: p };
    case 'retrieve':
      p.retrieve_ms = ev.retrieve_ms as number;
      p.retrieve_candidates = ev.candidates as number;
      p.mode = ev.mode as string;
      return { ...prev, stage: 'retrieve', progress: p };
    case 'rerank':
      p.rerank_ms = ev.rerank_ms as number;
      p.rerank_candidates_before = ev.candidates_before as number;
      p.rerank_candidates_after = ev.candidates_after as number;
      p.rerank_skipped = Boolean(ev.skipped);
      return { ...prev, stage: 'rerank', progress: p };
    case 'enrich':
      p.enriched_count = ev.enriched_count as number;
      return { ...prev, stage: 'enrich', progress: p };
    case 'complete':
      return { ...prev, stage: 'complete', progress: p, data: ev.response as SearchResponse };
    case 'error':
      return { ...prev, stage: 'error', error: String(ev.error ?? 'stream error') };
    default:
      return prev;
  }
}

