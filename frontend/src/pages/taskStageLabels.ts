/**
 * Chinese labels for backend stage/event codes and workflow-node names
 * (design addendum: "No raw event codes on screen" — the expanded task card
 * must never show strings like "match_paper.prepared" or "Prepare Index
 * Task" verbatim). Backend stage strings come from `stage=f"{type}.{phase}"`
 * across app/services/match_doc/jobs.py, app/tasks/indexing_tasks.py,
 * app/tasks/crawl_tasks.py, app/services/task_runtime.py — this table covers
 * every phase suffix currently emitted (grepped from those call sites).
 */

const STAGE_SUFFIX_LABEL: Record<string, string> = {
  queued: '排队中', pending: '等待中', running: '运行中', starting: '启动中',
  prepared: '准备完成', preparing: '准备中', prepare: '准备中',
  embedding: '向量化', qdrant: '向量检索', rerank: '重排序', aggregate: '汇总结果',
  completed: '已完成', failed: '失败', cancelled: '已取消', timeout: '超时',
  discarded: '已丢弃', progress: '进行中', retry: '重试', started: '已启动',
  index: '索引中', indexing: '索引中', index_failed: '索引失败',
  results: '采集结果', skipped: '已跳过', finalize: '收尾中', edge: '依赖检查',
};

// Canonical match_paper phase order (app/services/match_doc/jobs.py: the
// queued/running stages from its own emit_event calls, then the
// on_progress phase branches, in the order they're emitted). Used only to
// fill in the phases a cancelled/failed/timed-out run never reached as
// hollow/pending nodes after its terminal marker (taskFlow.ts
// buildSyntheticStageNodes) — not a general stage registry for other task
// types.
export const MATCH_PAPER_STAGE_ORDER = [
  'queued', 'running', 'prepared', 'embedding', 'qdrant', 'rerank', 'aggregate', 'completed',
];

/** `"match_paper.embedding"` -> `"向量化"`. The task-type prefix is dropped
 * because the card already names the task type; only the phase matters. */
export function stageLabel(stage: string | null | undefined): string {
  if (!stage) return '';
  const suffix = stage.split('.').pop() || stage;
  return STAGE_SUFFIX_LABEL[suffix] ?? suffix;
}

const NODE_NAME_LABEL: Record<string, string> = {
  'Prepare Index Task': '准备索引',
  'Finalize Index Task': '完成索引',
  'Prepare Batch Index': '准备批量索引',
  'Finalize Batch Index': '完成批量索引',
  'Prepare Crawl Job': '准备采集',
  'Finalize Crawl Job': '完成采集',
};

// ingest_pipeline / upload_pipeline WorkflowNode titles (app/services/
// pipeline/stages.py, upload_stages.py) are parameterized with the venue,
// year, or doc_id — a fixed lookup can't cover them, so these are patterns
// that translate the fixed English verb/suffix and keep the dynamic part
// (`$1`) exactly as the backend sent it.
const NODE_NAME_PATTERNS: [RegExp, string][] = [
  [/^Download (.+) PDFs$/, '下载 $1 PDF'],
  [/^Extract (.+) \(KSS marker\)$/, '解析 $1（KSS 标记）'],
  [/^Extract (.+) \(local marker\)$/, '解析 $1（本地标记）'],
  // The local worker names its stage after itself (`(local pvl_worker)`, stages.py _local_worker_label).
  [/^Extract (.+) \(local ([\w-]+)\)$/, '解析 $1（本地 $2）'],
  [/^Preindex (.+) → SQL$/, '预索引 $1 → SQL'],
  [/^Embed (.+) \(KSS BGE-M3\)$/, '向量化 $1（KSS BGE-M3）'],
  [/^Embed (.+) \(local BGE-M3\)$/, '向量化 $1（本地 BGE-M3）'],
  [/^Ingest (.+) → Qdrant$/, '写入 $1 → Qdrant'],
  [/^Extract upload doc (.+) \(marker\)$/, '解析上传文档 $1（标记）'],
  [/^Index upload doc (.+) -> Qdrant$/, '索引上传文档 $1 → Qdrant'],
];

/** Fixed English workflow-node names -> Chinese; real content (a URL, a
 * document title used as the node name) passes through unchanged — it's
 * data, not a code. */
export function nodeNameLabel(name: string): string {
  const fixed = NODE_NAME_LABEL[name];
  if (fixed) return fixed;
  for (const [pattern, replacement] of NODE_NAME_PATTERNS) {
    if (pattern.test(name)) return name.replace(pattern, replacement);
  }
  if (name.startsWith('Index ')) return `索引 ${name.slice('Index '.length)}`;
  return name;
}

// Fixed short Chinese stage names keyed by the workflow node's STABLE
// `subtask_key` (app/services/pipeline/stages.py / upload_stages.py's
// `key=...` — "download"/"extract"/"preindex"/"embed"/"ingest"/"index").
// Unlike the title (nodeNameLabel above), the key is never parameterized
// with venue/year/doc_id, so this can give a short label with no repeated
// header content and no implementation jargon (KSS/BGE-M3/SQL/Marker).
// upload_pipeline reuses the key "extract" for a different step (parsing
// the uploaded file, not converting a crawled PDF) — hence two separate
// tables instead of one keyed only by subtask_key.
const INGEST_STAGE_KEY_LABEL: Record<string, string> = {
  download: '下载', extract: '转换', preindex: '预索引', embed: '向量化', ingest: '入库', cleanup: '清理旧版本',
};
const UPLOAD_STAGE_KEY_LABEL: Record<string, string> = {
  extract: '解析', embed: '向量化', index: '入库',
};
const INGEST_TASK_TYPES = new Set(['ingest_pipeline', 'icml2026_external_pipeline']);

// Fallback when subtask_key isn't populated (real .88 data: every
// ingest_pipeline WorkflowNode currently comes back with subtask_key=null —
// whether an older deploy or pre-existing historical rows, either way it
// happens) — the same fixed short labels, matched against the TITLE's verb
// instead of the key. Mirrors NODE_NAME_PATTERNS' matching but maps to the
// short vocabulary, not the parameterized one.
const INGEST_TITLE_SHORT_PATTERNS: [RegExp, string][] = [
  [/^Download /, '下载'],
  [/^Extract .*\(KSS marker\)$/, '转换'],
  [/^Extract .*\(local [\w-]+\)$/, '转换'],
  [/^Preindex .* → SQL$/, '预索引'],
  [/^Embed .*\(KSS BGE-M3\)$/, '向量化'],
  [/^Embed .*\(local BGE-M3\)$/, '向量化'],
  [/^Ingest .* → Qdrant$/, '入库'],
];
const UPLOAD_TITLE_SHORT_PATTERNS: [RegExp, string][] = [
  [/^Extract upload doc .*\(marker\)$/, '解析'],
  [/^Index upload doc .* -> Qdrant$/, '入库'],
];

/** Prefers the fixed short subtask_key vocabulary for ingest/upload
 * pipelines; when subtask_key is missing, matches the same short label off
 * the title's verb instead; falls back to nodeNameLabel(name) for every
 * other task type (crawl_urls, batch_index, ...) and for anything neither
 * table recognizes (a future stage added to the pipeline). */
export function workflowNodeLabel(subtaskKey: string | null | undefined, taskType: string, name: string): string {
  const isUpload = taskType === 'upload_pipeline';
  const isIngest = INGEST_TASK_TYPES.has(taskType);
  const keyTable = isUpload ? UPLOAD_STAGE_KEY_LABEL : isIngest ? INGEST_STAGE_KEY_LABEL : null;
  const byKey = subtaskKey ? keyTable?.[subtaskKey] : undefined;
  if (byKey) return byKey;

  const titlePatterns = isUpload ? UPLOAD_TITLE_SHORT_PATTERNS : isIngest ? INGEST_TITLE_SHORT_PATTERNS : null;
  if (titlePatterns) {
    for (const [pattern, label] of titlePatterns) {
      if (pattern.test(name)) return label;
    }
  }
  return nodeNameLabel(name);
}

// Task-level (subtask_id=None) lifecycle events from
// app/services/pipeline/orchestrator.py's own emit_event calls — a
// different vocabulary from per-stage events, keyed by event_type since the
// message text alone isn't a stable code.
const TASK_LEVEL_EVENT_TYPE_LABEL: Record<string, string> = {
  'pipeline.cancelled': '流程已取消',
  'pipeline.completed': '流程已完成',
};

/** `null` means "don't show this event at all" — specifically
 * "pipeline.started", whose message ("pipeline started: download -> extract
 * -> ...") just restates in raw English the exact stage sequence the chain
 * above it already renders as nodes. Other known lifecycle event types get
 * a fixed Chinese label; anything unrecognized (e.g. a failure reason with
 * dynamic exception text) passes through unchanged rather than losing
 * information to a guessed-at translation. */
export function taskLevelEventLabel(event: { event_type: string; message: string }): string | null {
  if (event.event_type === 'pipeline.started') return null;
  return TASK_LEVEL_EVENT_TYPE_LABEL[event.event_type] ?? event.message;
}

const TASK_STATUS_LABEL: Record<string, string> = {
  pending: '排队中', running: '运行中', completed: '已完成',
  failed: '失败', cancelled: '已取消', timeout: '超时',
};

/** TaskBrief.status -> the word shown on its card and detail header. */
export function taskStatusLabel(status: string): string {
  return TASK_STATUS_LABEL[status] ?? stageLabel(status);
}

/** Backend error strings with a fixed shape get a Chinese sentence naming
 * the stage; anything else (a real exception message) passes through. */
export function taskErrorLabel(message: string, taskType: string): string {
  const m = /^stage '([\w-]+)' failed$/.exec(message.trim());
  if (!m) return message;
  const table = INGEST_TASK_TYPES.has(taskType) ? INGEST_STAGE_KEY_LABEL : UPLOAD_STAGE_KEY_LABEL;
  const stage = table[m[1]] ?? INGEST_STAGE_KEY_LABEL[m[1]] ?? m[1];
  return `「${stage}」阶段失败`;
}

/** Compact-card progress for a running ingest pipeline: time-weighted percent,
 *  the stage it is in (or waiting for), and the work-only ETA. */
export function pipelineProgressView(task: { task_type: string; pipeline?: {
  fraction: number; eta_seconds: number | null; waiting: boolean;
  stage: { key: string; name: string; progress: number; total: number } | null;
} | null }): { pct: number; stage: string; eta: number | null } | null {
  const p = task.pipeline;
  if (!p) return null;
  const pct = Math.round(p.fraction * 100);
  if (!p.stage) return { pct, stage: '', eta: p.eta_seconds };
  const label = workflowNodeLabel(p.stage.key, task.task_type, p.stage.name);
  const count = p.stage.total > 0 ? ` ${p.stage.progress}/${p.stage.total}` : '';
  return { pct, stage: p.waiting ? `等待${label}` : `${label}${count}`, eta: p.waiting ? null : p.eta_seconds };
}
