import type { PaperGroup, SearchResult } from '../types';

/**
 * Client-side fallback for SearchResponse.papers — mirrors the backend
 * app/services/search_grouping.group_into_papers so a legacy cached response
 * (or any flat chunk list) can still be rendered as paper cards.
 *
 * Groups by paper_id (falling back to doc_id), ranks papers by their best
 * (max) chunk score, and sorts each paper's chunks score-desc.
 */
export function groupPapersClientSide(results: SearchResult[], keepBestOnly = false): PaperGroup[] {
  const groups = new Map<string, PaperGroup>();
  for (const r of results) {
    const meta = (r.metadata || {}) as Record<string, unknown>;
    const paperId = (meta.paper_id as string | undefined) || '';
    const docId = String(r.doc_id ?? '');
    const key = paperId || docId || String(r.chunk_id ?? '');
    let g = groups.get(key);
    if (!g) {
      g = {
        paper_id: paperId,
        doc_id: docId,
        doc_title: r.doc_title || '',
        source: r.source || '',
        max_score: r.score,
        chunk_count: 0,
        metadata: { ...meta },
        chunks: [],
      };
      groups.set(key, g);
    }
    g.chunks.push(r);
    if (r.score > g.max_score) {
      g.max_score = r.score;
      g.doc_title = r.doc_title || g.doc_title;
      g.source = r.source || g.source;
      g.metadata = { ...meta };
    }
  }
  const list = Array.from(groups.values());
  for (const g of list) {
    g.chunks.sort((a, b) => b.score - a.score);
    if (keepBestOnly) g.chunks = g.chunks.slice(0, 1);
    g.chunk_count = g.chunks.length;
  }
  list.sort((a, b) => b.max_score - a.max_score);
  return list;
}
