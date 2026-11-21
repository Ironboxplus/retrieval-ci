import { describe, expect, it } from 'vitest';

import { stageLabel, nodeNameLabel, workflowNodeLabel, taskLevelEventLabel, taskStatusLabel, taskErrorLabel, pipelineProgressView } from './taskStageLabels';

describe('stageLabel', () => {
  it('translates known phase suffixes to Chinese, ignoring the task-type prefix', () => {
    expect(stageLabel('match_paper.queued')).toBe('排队中');
    expect(stageLabel('match_paper.embedding')).toBe('向量化');
    expect(stageLabel('task.completed')).toBe('已完成');
    expect(stageLabel('workflow.finalize.running')).toBe('运行中');
  });

  it('falls back to the raw suffix for an unrecognized phase, never throws', () => {
    expect(stageLabel('some_task.a_future_phase')).toBe('a_future_phase');
  });

  it('is empty for null/undefined/empty input', () => {
    expect(stageLabel(null)).toBe('');
    expect(stageLabel(undefined)).toBe('');
    expect(stageLabel('')).toBe('');
  });
});

describe('nodeNameLabel', () => {
  it('translates known fixed workflow-node names', () => {
    expect(nodeNameLabel('Prepare Index Task')).toBe('准备索引');
    expect(nodeNameLabel('Finalize Index Task')).toBe('完成索引');
    expect(nodeNameLabel('Prepare Batch Index')).toBe('准备批量索引');
    expect(nodeNameLabel('Finalize Batch Index')).toBe('完成批量索引');
    expect(nodeNameLabel('Prepare Crawl Job')).toBe('准备采集');
    expect(nodeNameLabel('Finalize Crawl Job')).toBe('完成采集');
  });

  it('translates the "Index {x}" prefix pattern, keeping the real content', () => {
    expect(nodeNameLabel('Index https://example.com')).toBe('索引 https://example.com');
  });

  it('leaves real content (URLs, titles) untouched — not a code', () => {
    expect(nodeNameLabel('https://example.com/a')).toBe('https://example.com/a');
    expect(nodeNameLabel('Attention Is All You Need')).toBe('Attention Is All You Need');
  });

  // Real production ingest_pipeline / upload_pipeline WorkflowNode titles
  // (app/services/pipeline/stages.py, upload_stages.py) — a different,
  // parameterized vocabulary from the fixed "Prepare Index Task" names
  // above (venue/year/doc_id are dynamic, so these need pattern matching,
  // not a lookup table).
  it('translates ingest_pipeline stage titles, keeping venue/year as-is', () => {
    expect(nodeNameLabel('Download SIGIR 2026 PDFs')).toBe('下载 SIGIR 2026 PDF');
    expect(nodeNameLabel('Extract SIGIR 2026 (KSS marker)')).toBe('解析 SIGIR 2026（KSS 标记）');
    expect(nodeNameLabel('Extract SIGIR 2026 (local marker)')).toBe('解析 SIGIR 2026（本地标记）');
    expect(nodeNameLabel('Preindex SIGIR 2026 → SQL')).toBe('预索引 SIGIR 2026 → SQL');
    expect(nodeNameLabel('Embed SIGIR 2026 (KSS BGE-M3)')).toBe('向量化 SIGIR 2026（KSS BGE-M3）');
    expect(nodeNameLabel('Embed SIGIR 2026 (local BGE-M3)')).toBe('向量化 SIGIR 2026（本地 BGE-M3）');
    expect(nodeNameLabel('Ingest SIGIR 2026 → Qdrant')).toBe('写入 SIGIR 2026 → Qdrant');
  });

  it('translates upload_pipeline stage titles, keeping the doc id as-is', () => {
    expect(nodeNameLabel('Extract upload doc 123 (marker)')).toBe('解析上传文档 123（标记）');
    expect(nodeNameLabel('Index upload doc 123 -> Qdrant')).toBe('索引上传文档 123 → Qdrant');
  });
});

describe('workflowNodeLabel', () => {
  // The parameterized-title translation (nodeNameLabel) repeats the venue/
  // year the card header already shows and carries implementation jargon
  // ("KSS BGE-M3", "→ SQL", "(marker)") — this keys off the node's STABLE
  // `subtask_key` (app/services/pipeline/stages.py's `key=...`) instead,
  // which is never parameterized, for a fixed short label with none of that.
  it('gives ingest_pipeline nodes fixed short labels regardless of venue/year/tool', () => {
    expect(workflowNodeLabel('download', 'ingest_pipeline', 'Download SIGIR 2026 PDFs')).toBe('下载');
    expect(workflowNodeLabel('extract', 'ingest_pipeline', 'Extract SIGIR 2026 (KSS marker)')).toBe('转换');
    expect(workflowNodeLabel('extract', 'ingest_pipeline', 'Extract SIGIR 2026 (local marker)')).toBe('转换');
    expect(workflowNodeLabel('preindex', 'ingest_pipeline', 'Preindex SIGIR 2026 → SQL')).toBe('预索引');
    expect(workflowNodeLabel('embed', 'ingest_pipeline', 'Embed SIGIR 2026 (KSS BGE-M3)')).toBe('向量化');
    expect(workflowNodeLabel('embed', 'ingest_pipeline', 'Embed SIGIR 2026 (local BGE-M3)')).toBe('向量化');
    expect(workflowNodeLabel('ingest', 'ingest_pipeline', 'Ingest SIGIR 2026 → Qdrant')).toBe('入库');
  });

  it('applies the same fixed vocabulary to icml2026_external_pipeline (shares stage keys with ingest_pipeline)', () => {
    expect(workflowNodeLabel('download', 'icml2026_external_pipeline', 'Download ICML 2026 PDFs')).toBe('下载');
  });

  it('gives upload_pipeline nodes their own fixed short labels — "extract" means something different there', () => {
    expect(workflowNodeLabel('extract', 'upload_pipeline', 'Extract upload doc 123 (marker)')).toBe('解析');
    expect(workflowNodeLabel('index', 'upload_pipeline', 'Index upload doc 123 -> Qdrant')).toBe('入库');
  });

  it('falls back to nodeNameLabel for task types with no fixed vocabulary here', () => {
    expect(workflowNodeLabel('some-url', 'crawl_urls', 'Index https://example.com')).toBe('索引 https://example.com');
  });

  it('falls back to nodeNameLabel when the subtask_key is unrecognized even for a known task type', () => {
    expect(workflowNodeLabel('unknown_stage', 'ingest_pipeline', 'Some Future Stage')).toBe('Some Future Stage');
  });

  // Real production data: .88's TaskSubtask rows have subtask_key=None for
  // every ingest_pipeline node (confirmed via a live workflow response) —
  // whether from an older deploy or genuinely-old historical rows, this
  // must still resolve to the fixed short label from the TITLE, not fall
  // through to the parameterized nodeNameLabel translation.
  it('resolves the fixed short label from the title when subtask_key is null (real .88 data shape)', () => {
    expect(workflowNodeLabel(null, 'ingest_pipeline', 'Download SIGIR 2026 PDFs')).toBe('下载');
    expect(workflowNodeLabel(null, 'ingest_pipeline', 'Extract SIGIR 2026 (local marker)')).toBe('转换');
    expect(workflowNodeLabel(null, 'ingest_pipeline', 'Preindex SIGIR 2026 → SQL')).toBe('预索引');
    expect(workflowNodeLabel(null, 'ingest_pipeline', 'Embed SIGIR 2026 (local BGE-M3)')).toBe('向量化');
    expect(workflowNodeLabel(null, 'ingest_pipeline', 'Ingest SIGIR 2026 → Qdrant')).toBe('入库');
    expect(workflowNodeLabel(undefined, 'upload_pipeline', 'Extract upload doc 123 (marker)')).toBe('解析');
    expect(workflowNodeLabel(undefined, 'upload_pipeline', 'Index upload doc 123 -> Qdrant')).toBe('入库');
  });
});

describe('taskLevelEventLabel', () => {
  // app/services/pipeline/orchestrator.py's task-level (subtask_id=None)
  // lifecycle events.
  it('hides the pipeline.started event — it just restates the stage chain the card already renders as nodes', () => {
    expect(taskLevelEventLabel({ event_type: 'pipeline.started', message: 'pipeline started: download -> extract -> preindex -> embed -> ingest' })).toBeNull();
  });

  it('translates other known pipeline lifecycle event types to Chinese', () => {
    expect(taskLevelEventLabel({ event_type: 'pipeline.cancelled', message: 'pipeline cancelled' })).toBe('流程已取消');
    expect(taskLevelEventLabel({ event_type: 'pipeline.completed', message: 'pipeline completed' })).toBe('流程已完成');
  });

  it('passes through an unrecognized event type unchanged rather than guessing at a translation', () => {
    expect(taskLevelEventLabel({ event_type: 'pipeline.failed', message: "stage 'download' failed" }))
      .toBe("stage 'download' failed");
  });
});

describe('taskStatusLabel', () => {
  it('names every task status in Chinese', () => {
    expect(taskStatusLabel('pending')).toBe('排队中');
    expect(taskStatusLabel('running')).toBe('运行中');
    expect(taskStatusLabel('completed')).toBe('已完成');
    expect(taskStatusLabel('failed')).toBe('失败');
    expect(taskStatusLabel('cancelled')).toBe('已取消');
  });
});

describe('taskErrorLabel', () => {
  it('names the failed stage for the fixed backend shape', () => {
    expect(taskErrorLabel("stage 'preindex' failed", 'ingest_pipeline')).toBe('「预索引」阶段失败');
    expect(taskErrorLabel("stage 'index' failed", 'upload_pipeline')).toBe('「入库」阶段失败');
  });
  it('passes real exception messages through unchanged', () => {
    expect(taskErrorLabel('Qdrant timed out after 30s', 'ingest_pipeline')).toBe('Qdrant timed out after 30s');
  });
});

describe('pipelineProgressView', () => {
  // Live .88 /api/tasks shapes, 2026-09-25 17:4x.
  const embedding = {
    task_type: 'ingest_pipeline',
    pipeline: {
      fraction: 0.5531, eta_seconds: 922, waiting: false,
      stage: { key: 'embed', name: 'Embed arXiv 20260600 (BGE-M3 on arc99)', status: 'running', progress: 5, total: 8 },
    },
  };
  const parked = {
    task_type: 'ingest_pipeline',
    pipeline: {
      fraction: 0.1296, eta_seconds: null, waiting: true,
      stage: { key: 'embed', name: 'Embed arXiv 20260301 (BGE-M3 on arc99)', status: 'pending', progress: 0, total: 0 },
    },
  };

  it('shows the time-weighted percent, the stage with its count, and the ETA', () => {
    expect(pipelineProgressView(embedding)).toEqual({ pct: 55, stage: '向量化 5/8', eta: 922 });
  });

  it('names the stage a parked task is waiting for and gives no ETA', () => {
    expect(pipelineProgressView(parked)).toEqual({ pct: 13, stage: '等待向量化', eta: null });
  });

  it('is absent for tasks without pipeline progress', () => {
    expect(pipelineProgressView({ task_type: 'crawl_urls' })).toBeNull();
  });
});
