import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../client';
import type {
  DocumentListResponse,
  DocumentDetail,
  ActionResponse,
  DocumentIndexStatus,
} from '../../types';

export function useDocuments(
  page = 1,
  pageSize = 20,
  sourceType?: string,
  options?: { enabled?: boolean },
) {
  return useQuery({
    queryKey: ['documents', page, pageSize, sourceType ?? 'all'],
    queryFn: () =>
      api
        .get<DocumentListResponse>('/documents', {
          params: {
            page,
            page_size: pageSize,
            ...(sourceType ? { source_type: sourceType } : {}),
          },
        })
        .then((r) => r.data),
    placeholderData: (prev) => prev,
    enabled: options?.enabled ?? true,
  });
}

export function useDocumentDetail(docId: number | undefined) {
  return useQuery({
    queryKey: ['documents', docId],
    queryFn: () =>
      api.get<DocumentDetail>(`/documents/${docId}`).then((r) => r.data),
    enabled: !!docId,
    // A task can reference a doc_id whose document was since deleted
    // (Tasks.tsx's useTaskSubjectDisplay falls back to the pure subject
    // string when this 404s) — retrying a 404 can never succeed, it only
    // doubles a request that was never going to work.
    retry: false,
  });
}

export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { file: File; title?: string; autoIndex?: boolean }) => {
      const form = new FormData();
      form.append('file', data.file);
      if (data.title) form.append('title', data.title);
      if (data.autoIndex !== undefined)
        form.append('auto_index', String(data.autoIndex));
      return api
        .post('/documents/upload', form, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        .then((r) => r.data);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  });
}

export function useUpdateDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      docId: number;
      title?: string;
      metadata?: Record<string, unknown>;
    }) =>
      api
        .put<ActionResponse>(`/documents/${data.docId}`, {
          title: data.title,
          metadata: data.metadata,
        })
        .then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (docId: number) =>
      api.delete<ActionResponse>(`/documents/${docId}`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  });
}

export function useDocumentIndexStatus(docId: number | undefined) {
  return useQuery({
    queryKey: ['documents', docId, 'index-status'],
    queryFn: () =>
      api
        .get<DocumentIndexStatus>(`/documents/${docId}/index-status`)
        .then((r) => r.data),
    enabled: !!docId,
    staleTime: 10000,
  });
}
