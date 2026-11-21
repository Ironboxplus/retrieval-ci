import { useId } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../client';
import type {
  PaperBrowseParams,
  PaperBrowseResponse,
  TagsResponse,
  PaperStats,
  PaperBrief,
  ActionResponse,
  PaperIndexStatus,
  IndexActionResponse,
} from '../../types';

// CSV-serialize array params so axios doesn't emit `venue[]=A` brackets,
// which FastAPI's default Query(List[str]) parsing doesn't accept. Backend
// `/papers/browse` accepts both `venue=A&venue=B` and `venue=A,B` (W22-style
// split). We pick CSV here to keep the URL short and round-trip predictable.
function flattenParams(params: PaperBrowseParams): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') continue;
    if (Array.isArray(v)) {
      if (v.length === 0) continue;
      out[k] = v.join(',');
    } else {
      out[k] = v;
    }
  }
  return out;
}

export function usePaperBrowse(params: PaperBrowseParams) {
  return useQuery({
    queryKey: ['papers', 'browse', params],
    queryFn: () =>
      api
        .get<PaperBrowseResponse>('/papers/browse', { params: flattenParams(params) })
        .then((r) => r.data),
    placeholderData: (prev) => prev,
  });
}

export function usePaperTags() {
  return useQuery({
    queryKey: ['papers', 'tags'],
    queryFn: () => api.get<TagsResponse>('/papers/tags').then((r) => r.data),
    staleTime: 60000,
  });
}

export function usePaperStats() {
  return useQuery({
    queryKey: ['papers', 'stats'],
    queryFn: () => api.get<PaperStats>('/papers/stats').then((r) => r.data),
  });
}

export function usePaperDetail(paperId: string | undefined) {
  const requestId = useId();
  return useQuery({
    queryKey: ['papers', paperId],
    queryFn: () =>
      api.get<PaperBrief>(`/papers/${paperId}`, {
        headers: { 'X-Idempotency-Key': `paper-detail-${paperId}-${requestId}` },
      }).then((r) => r.data),
    enabled: !!paperId,
    // Same reasoning as useDocumentDetail: a match_paper task's paper_id can
    // reference a paper that's since been removed from the corpus; retrying
    // a 404 doubles a request that was never going to succeed.
    retry: false,
  });
}

export function useDeletePaper() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (paperId: string) =>
      api.delete<ActionResponse>(`/papers/${paperId}`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['papers'] }),
  });
}

export function usePaperIndexStatus(paperId: string | undefined) {
  return useQuery({
    queryKey: ['papers', paperId, 'index-status'],
    queryFn: () =>
      api
        .get<PaperIndexStatus>(`/papers/${paperId}/index-status`)
        .then((r) => r.data),
    enabled: !!paperId,
    staleTime: 10000,
  });
}

export function usePaperBibtex(paperId: string | undefined) {
  return useQuery({
    queryKey: ['papers', paperId, 'bibtex'],
    queryFn: () =>
      api.get<string>(`/papers/${paperId}/bibtex`, { responseType: 'text' as never }).then((r) => r.data as unknown as string),
    enabled: false,
  });
}

export function usePaperFulltext(paperId: string | undefined) {
  const requestId = useId();
  return useQuery({
    queryKey: ['papers', paperId, 'fulltext'],
    queryFn: () =>
      api.get<{ paper_id: string; title: string; content: string; size: number }>(
        `/papers/${paperId}/fulltext`,
        { headers: { 'X-Idempotency-Key': `paper-fulltext-${paperId}-${requestId}` } },
      ).then((r) => r.data),
    enabled: !!paperId,
    staleTime: 300_000,
  });
}

export function useTriggerPaperIndex() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (paperId: string) =>
      api
        .post<IndexActionResponse>(`/papers/${paperId}/index`)
        .then((r) => r.data),
    onSuccess: (_data, paperId) => {
      qc.invalidateQueries({ queryKey: ['papers', paperId, 'index-status'] });
      qc.invalidateQueries({ queryKey: ['papers', paperId] });
    },
  });
}
