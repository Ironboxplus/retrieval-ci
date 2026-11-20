import { useMutation } from '@tanstack/react-query';
import api from '../client';
import type { MatchDocRequest, MatchDocResponse } from '../../types';

// 功能2：doc → 最相近论文匹配。逐块检索 + 论文级聚合，耗时偏长 → 单独放宽超时。
export function useMatchDoc() {
  return useMutation({
    mutationFn: (req: MatchDocRequest) =>
      api
        .post<MatchDocResponse>('/match-doc', req, { timeout: 180000 })
        .then((r) => r.data),
  });
}
