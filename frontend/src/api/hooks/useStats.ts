import { useQuery } from '@tanstack/react-query';
import api from '../client';
import type { StatsResponse } from '../../types';
import { useAuthenticatedApiEnabled } from './useAuth';

export function useStats() {
  const enabled = useAuthenticatedApiEnabled();
  return useQuery({
    queryKey: ['stats'],
    queryFn: () => api.get<StatsResponse>('/stats').then((r) => r.data),
    enabled,
    refetchInterval: 30000,
  });
}
