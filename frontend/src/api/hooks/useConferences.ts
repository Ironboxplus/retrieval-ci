import { useQuery } from '@tanstack/react-query';
import api from '../client';
import { useAuthenticatedApiEnabled } from './useAuth';

export interface ConferenceRank {
  ccf: string;
  core?: string;
  thcpl?: string;
}

export interface ConferenceInfo {
  title: string;
  description: string;
  sub: string;
  rank: ConferenceRank;
  dblp: string;
  venue_tier: string;
  years: number[];
  downloadable?: boolean;
}

export interface CategoryInfo {
  name: string;
  name_en: string;
}

export interface ConferencesResponse {
  categories: Record<string, CategoryInfo>;
  conferences: ConferenceInfo[];
}

export function useConferences() {
  const enabled = useAuthenticatedApiEnabled();
  return useQuery<ConferencesResponse>({
    queryKey: ['conferences'],
    queryFn: async () => (await api.get('/conferences')).data,
    enabled,
    staleTime: Infinity,
  });
}

export type VenueGroup = { label: string; sub: string; venues: { title: string; ccf: string }[] };

export function buildVenueGroups(data: ConferencesResponse | undefined): VenueGroup[] {
  if (!data) return [];
  const bySub: Record<string, VenueGroup> = {};
  for (const conf of data.conferences) {
    const cat = data.categories[conf.sub];
    if (!bySub[conf.sub]) {
      bySub[conf.sub] = {
        label: cat?.name_en ?? conf.sub,
        sub: conf.sub,
        venues: [],
      };
    }
    bySub[conf.sub].venues.push({ title: conf.title, ccf: conf.rank.ccf });
  }
  const order = ['AI', 'DB', 'NW', 'SC', 'SE', 'DS', 'CT', 'CG', 'HI', 'MX'];
  return order.filter((s) => bySub[s]).map((s) => bySub[s]);
}
