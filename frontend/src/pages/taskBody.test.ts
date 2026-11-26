import { describe, expect, it } from 'vitest';

import {
  taskSubject, taskTypeLabel, paperVenueYearFallback, taskDocId, taskGroupKey, type TaskLike,
} from './taskBody';

describe('taskTypeLabel', () => {
  it('maps known task types to Chinese names', () => {
    expect(taskTypeLabel('ingest_pipeline')).toBe('论文入库');
    expect(taskTypeLabel('crawl_urls')).toBe('网页采集');
    expect(taskTypeLabel('match_paper')).toBe('论文匹配');
    expect(taskTypeLabel('upload_pipeline')).toBe('上传处理');
    expect(taskTypeLabel('medical_corpus_download')).toBe('医学语料下载');
    expect(taskTypeLabel('icml2026_external_pipeline')).toBe('论文入库');
  });

  it('falls back to the raw task_type for unknown values', () => {
    expect(taskTypeLabel('some_future_task')).toBe('some_future_task');
  });
});

describe('taskSubject', () => {
  it('ingest_pipeline: venue + year', () => {
    const t: TaskLike = { task_type: 'ingest_pipeline', params: { venue: 'IJCAI', year: 2026 } };
    expect(taskSubject(t)).toBe('IJCAI 2026');
  });

  it('icml2026_external_pipeline uses the same venue/year shape', () => {
    const t: TaskLike = { task_type: 'icml2026_external_pipeline', params: { venue: 'ICML', year: 2026, source: 'watcher' } };
    expect(taskSubject(t)).toBe('ICML 2026');
  });

  it('ingest_pipeline arXiv batch and month keys read as months', () => {
    expect(taskSubject({ task_type: 'ingest_pipeline', params: { venue: 'arXiv', year: 20260903 } }))
      .toBe('arXiv 2026-09 #3');
    expect(taskSubject({ task_type: 'ingest_pipeline', params: { venue: 'arXiv', year: 202609 } }))
      .toBe('arXiv 2026-09');
    expect(taskSubject({ task_type: 'ingest_pipeline', params: { venue: 'ACL', year: 2026 } }))
      .toBe('ACL 2026');
  });

  it('ingest_pipeline with only venue (no year yet)', () => {
    const t: TaskLike = { task_type: 'ingest_pipeline', params: { venue: 'AAAI' } };
    expect(taskSubject(t)).toBe('AAAI');
  });

  it('crawl_urls: single URL shown verbatim', () => {
    const t: TaskLike = { task_type: 'crawl_urls', params: { urls: ['https://example.com/a'] } };
    expect(taskSubject(t)).toBe('https://example.com/a');
  });

  it('crawl_urls: multiple URLs shown as a count', () => {
    const t: TaskLike = { task_type: 'crawl_urls', params: { urls: ['a', 'b', 'c'] } };
    expect(taskSubject(t)).toBe('3 个网址');
  });

  it('crawl_urls: no urls array', () => {
    const t: TaskLike = { task_type: 'crawl_urls', params: {} };
    expect(taskSubject(t)).toBe('');
  });

  it('match_paper: falls back to VENUE YEAR parsed from the paper_id slug — never the raw id (users do not care about ids)', () => {
    const t: TaskLike = { task_type: 'match_paper', params: { input_type: 'paper_id', paper_id: 'neurips-2024-123' } };
    expect(taskSubject(t)).toBe('NEURIPS 2024');
  });

  it('upload_pipeline: never shows the UUID-renamed file_path — the real title comes from a document lookup at the JSX layer', () => {
    const t: TaskLike = { task_type: 'upload_pipeline', params: { doc_id: 5, file_path: '/data/uploads/50ade8d6-e02f-40dc-b6cd-5bcaeefbc110.pdf' } };
    expect(taskSubject(t)).toBe('');
  });

  it('medical_corpus_download: explicit limit', () => {
    const t: TaskLike = { task_type: 'medical_corpus_download', params: { download_limit: 500, allow_cf_bypass: true } };
    expect(taskSubject(t)).toBe('限 500 篇');
  });

  it('medical_corpus_download: limit 0 means "all pending"', () => {
    const t: TaskLike = { task_type: 'medical_corpus_download', params: { download_limit: 0 } };
    expect(taskSubject(t)).toBe('全部待下载');
  });

  it('unknown task_type returns empty string, not a crash', () => {
    const t: TaskLike = { task_type: 'some_future_task', params: { anything: 1 } };
    expect(taskSubject(t)).toBe('');
  });

  it('tolerates null/undefined params', () => {
    expect(taskSubject({ task_type: 'ingest_pipeline', params: null })).toBe('');
    expect(taskSubject({ task_type: 'ingest_pipeline', params: undefined })).toBe('');
  });
});

describe('paperVenueYearFallback', () => {
  it('parses a simple venue-year-hash paper id', () => {
    expect(paperVenueYearFallback('iclr-2024-sLdVl0q68X')).toBe('ICLR 2024');
  });

  it('parses a multi-word venue', () => {
    expect(paperVenueYearFallback('usenix-security-2024-abc123')).toBe('USENIX SECURITY 2024');
  });

  it('parses the numeric-suffix corpus id shape', () => {
    expect(paperVenueYearFallback('ijcai-2026-0996')).toBe('IJCAI 2026');
  });

  it('returns empty string for an id with no discernible year', () => {
    expect(paperVenueYearFallback('not-a-recognizable-id')).toBe('');
  });

  it('returns empty string for an empty id', () => {
    expect(paperVenueYearFallback('')).toBe('');
  });
});

describe('taskDocId', () => {
  it('extracts params.doc_id for upload_pipeline', () => {
    expect(taskDocId({ task_type: 'upload_pipeline', params: { doc_id: 5, file_path: 'x' } })).toBe(5);
  });

  it('is null when doc_id is missing or not a number', () => {
    expect(taskDocId({ task_type: 'upload_pipeline', params: {} })).toBeNull();
    expect(taskDocId({ task_type: 'upload_pipeline', params: { doc_id: 'five' } })).toBeNull();
  });

  it('is null for non-upload task types', () => {
    expect(taskDocId({ task_type: 'ingest_pipeline', params: { doc_id: 5 } })).toBeNull();
  });
});

describe('taskGroupKey', () => {
  it('groups match_paper attempts against the same paper_id together', () => {
    const a: TaskLike = { task_type: 'match_paper', params: { paper_id: 'iclr-2024-x' } };
    const b: TaskLike = { task_type: 'match_paper', params: { paper_id: 'iclr-2024-x' } };
    const c: TaskLike = { task_type: 'match_paper', params: { paper_id: 'iclr-2024-y' } };
    expect(taskGroupKey(a)).toBe(taskGroupKey(b));
    expect(taskGroupKey(a)).not.toBe(taskGroupKey(c));
  });

  it('groups upload_pipeline attempts against the same doc_id together', () => {
    const a: TaskLike = { task_type: 'upload_pipeline', params: { doc_id: 5 } };
    const b: TaskLike = { task_type: 'upload_pipeline', params: { doc_id: 5 } };
    const c: TaskLike = { task_type: 'upload_pipeline', params: { doc_id: 6 } };
    expect(taskGroupKey(a)).toBe(taskGroupKey(b));
    expect(taskGroupKey(a)).not.toBe(taskGroupKey(c));
  });

  it('groups ingest_pipeline attempts against the same venue+year together', () => {
    const a: TaskLike = { task_type: 'ingest_pipeline', params: { venue: 'ACL', year: 2026 } };
    const b: TaskLike = { task_type: 'ingest_pipeline', params: { venue: 'ACL', year: 2026 } };
    const c: TaskLike = { task_type: 'ingest_pipeline', params: { venue: 'KDD', year: 2026 } };
    expect(taskGroupKey(a)).toBe(taskGroupKey(b));
    expect(taskGroupKey(a)).not.toBe(taskGroupKey(c));
  });

  it('never collapses two genuinely different task types into the same group', () => {
    const a: TaskLike = { task_type: 'match_paper', params: { paper_id: '1' } };
    const b: TaskLike = { task_type: 'ingest_pipeline', params: { paper_id: '1' } };
    expect(taskGroupKey(a)).not.toBe(taskGroupKey(b));
  });

  it('falls back to the subject string so unrecognized shapes still get a stable key', () => {
    const a: TaskLike = { task_type: 'crawl_urls', params: { urls: ['https://x.com'] } };
    const b: TaskLike = { task_type: 'crawl_urls', params: { urls: ['https://x.com'] } };
    expect(taskGroupKey(a)).toBe(taskGroupKey(b));
  });
});
