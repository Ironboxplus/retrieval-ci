import { useQuery } from '@tanstack/react-query';
import api from '../client';
import { useAuthenticatedApiEnabled } from './useAuth';

export interface VenueYearCount {
  year: number;
  count: number;
}

export interface VenueCoverage {
  venue: string;
  count: number;
  min_year: number;
  max_year: number;
  /** Years inside [min_year, max_year] that were never indexed. */
  missing_years: number[];
  years: VenueYearCount[];
}

export interface IndexCoverageResponse {
  venues: VenueCoverage[];
  total_papers: number;
  /** Papers with no venue or no year — counted so the cards reconcile. */
  unclassified: number;
  venue_count: number;
  min_year: number | null;
  max_year: number | null;
}

export function useIndexCoverage(enabledWhen = true) {
  const authed = useAuthenticatedApiEnabled();
  return useQuery({
    queryKey: ['papers', 'coverage'],
    queryFn: () => api.get<IndexCoverageResponse>('/papers/coverage').then((r) => r.data),
    enabled: authed && enabledWhen,
    // The backend caches this for 5 minutes and it only moves after an ingest,
    // so there is nothing to gain from refetching on every window focus.
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}
