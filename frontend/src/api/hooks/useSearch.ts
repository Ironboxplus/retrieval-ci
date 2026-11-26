import { useMutation } from '@tanstack/react-query';
import api from '../client';
import type {
  SearchRequest,
  SearchResponse,
  PaperSearchRequest,
  PaperSearchResponse,
} from '../../types';

export function useSearch() {
  return useMutation({
    mutationFn: (req: SearchRequest) =>
      api.post<SearchResponse>('/search', req).then((r) => r.data),
  });
}

export function usePaperSearch() {
  return useMutation({
    mutationFn: (req: PaperSearchRequest) =>
      api.post<PaperSearchResponse>('/papers/search', req).then((r) => r.data),
  });
}
