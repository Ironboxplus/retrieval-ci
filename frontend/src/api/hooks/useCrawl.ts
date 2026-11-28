import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../client';
import type {
  CrawlUrlsRequest,
  CrawlRecordListResponse,
  ActionResponse,
} from '../../types';

export function useCrawlRecords(page = 1, pageSize = 20, taskId?: number) {
  return useQuery({
    queryKey: ['crawl', 'records', page, pageSize, taskId ?? 'all'],
    queryFn: () =>
      api
        .get<CrawlRecordListResponse>('/crawl/records', {
          params: { page, page_size: pageSize, task_id: taskId },
        })
        .then((r) => r.data),
    placeholderData: (prev) => prev,
  });
}

export function useCrawlUrls() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: CrawlUrlsRequest) =>
      api.post<ActionResponse>('/crawl/urls', req).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['crawl'] }),
  });
}
