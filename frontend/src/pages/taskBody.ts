/**
 * Pure task-card "subject" extraction (frontend refactor F1-f). MorphGrid's
 * compact task card shows the Chinese task-type name + one subject string
 * pulled from `params` (e.g. "IJCAI 2026", a URL count, a paper id) instead
 * of the numeric task id. Extraction is per task_type because each backend
 * task creator (app/services/pipeline/service.py, app/api/routes/crawl.py,
 * app/services/match_doc/jobs.py, app/services/pipeline/upload_service.py,
 * app/services/medical_corpus/service.py,
 * scripts/collect/watch_icml2026_web_task.py) shapes `params` differently.
 */

export interface TaskLike {
  task_type: string;
  params?: Record<string, unknown> | null;
}

const TYPE_LABEL: Record<string, string> = {
  ingest_pipeline: '论文入库',
  icml2026_external_pipeline: '论文入库',
  crawl_urls: '网页采集',
  match_paper: '论文匹配',
  upload_pipeline: '上传处理',
  medical_corpus_download: '医学语料下载',
  batch_index: '批量索引',
  journal_index: 'Journal 索引',
  code_index: '代码索引',
  code_duplicates: '重复代码检测',
};

export function taskTypeLabel(taskType: string): string {
  return TYPE_LABEL[taskType] ?? taskType;
}

const PUBLIC_CODE_OPERATION_LABEL: Record<string, string> = {
  check_updates: '公开代码检查更新',
  index: '公开代码索引',
  import: '公开代码导入',
};

/** Card/detail title: `taskTypeLabel`, except public_code_job whose name
 * depends on `params.operation`. */
export function taskDisplayLabel(task: TaskLike): string {
  if (task.task_type === 'public_code_job') {
    return PUBLIC_CODE_OPERATION_LABEL[String((task.params ?? {}).operation)] ?? '公开代码';
  }
  return taskTypeLabel(task.task_type);
}

// These runners are backed by in-process queues (app/services/pipeline/
// queue_tasks.py): the client re-submits, the task API cannot restart them.
const NON_RETRYABLE_TYPES = new Set(['journal_index', 'code_index', 'code_duplicates', 'public_code_job']);

export function taskCanRetry(task: TaskLike): boolean {
  return !NON_RETRYABLE_TYPES.has(task.task_type);
}

// public_code_job cancel always answers 409 (app/services/public_code/jobs.py
// `_cancel_task_job` raises TaskNotCancellable), so the button is not offered.
export function taskCanCancel(task: TaskLike): boolean {
  return task.task_type !== 'public_code_job';
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

// Paper ids in this app are consistently "{venue-slug}-{year}-{suffix}"
// (e.g. "iclr-2024-sLdVl0q68X", "usenix-security-2024-abc123",
// "ijcai-2026-0996") — everything up to the last "-YYYY-" is the venue,
// hyphens become spaces, uppercased.
const PAPER_ID_VENUE_YEAR = /^(.+)-(\d{4})-[^-]+$/;

/** VENUE YEAR parsed out of a paper_id slug, or '' if it doesn't match the
 * shape — used as the match_paper task subject fallback when no title is
 * available (params never carries one; see taskSubject's match_paper case). */
export function paperVenueYearFallback(paperId: string): string {
  const m = PAPER_ID_VENUE_YEAR.exec(paperId);
  if (!m) return '';
  return `${m[1].split('-').map((s) => s.toUpperCase()).join(' ')} ${m[2]}`;
}

/** `params.doc_id` for an upload_pipeline task, or null — lets the JSX layer
 * fetch the real document title instead of showing the UUID-renamed
 * file_path (see taskSubject's upload_pipeline case). */
export function taskDocId(task: TaskLike): number | null {
  if (task.task_type !== 'upload_pipeline') return null;
  const docId = (task.params ?? {}).doc_id;
  return typeof docId === 'number' ? docId : null;
}

/**
 * Groups repeated attempts against the same real-world thing (design
 * addendum §1.1: "four identical 论文匹配 cards for the same paper" should be
 * one stacked card) — task_type plus the identifying field for that type
 * (paper_id / doc_id / venue+year), so two attempts against the same paper
 * group even if their status or timestamps differ. Falls back to the
 * (type, subject) pair for shapes without a stronger identity, which still
 * keeps genuinely repeated submissions (e.g. the same crawl URL) together.
 */
export function taskGroupKey(task: TaskLike): string {
  const p = task.params ?? {};
  switch (task.task_type) {
    case 'match_paper':
      return `match_paper:${p.paper_id ?? ''}`;
    case 'upload_pipeline':
      return `upload_pipeline:${p.doc_id ?? ''}`;
    case 'ingest_pipeline':
    case 'icml2026_external_pipeline':
      return `${task.task_type}:${p.venue ?? ''}:${p.year ?? ''}`;
    case 'journal_index':
    case 'code_index':
    case 'code_duplicates':
      return `${task.task_type}:${p.project ?? ''}`;
    case 'public_code_job':
      return `public_code_job:${p.operation ?? ''}:${stringList(p.source_ids).join(',')}`;
    default:
      return `${task.task_type}:${taskSubject(task)}`;
  }
}

/** Extract the one-line subject for a task card. Returns '' when params
 * don't carry enough information (never throws — params shapes drift). */
/** arXiv ingest tasks carry a batch key YYYYMMPP (or a month YYYYMM) in
 * `year`: "20260903" -> "2026-09 #3", "202609" -> "2026-09". */
export function arxivKeyLabel(venue: unknown, year: unknown): string | null {
  if (String(venue).toLowerCase() !== 'arxiv') return null;
  const key = String(year);
  if (!/^\d{6}(\d{2})?$/.test(key)) return null;
  const month = `${key.slice(0, 4)}-${key.slice(4, 6)}`;
  return key.length === 8 ? `${month} #${Number(key.slice(6))}` : month;
}

export function taskSubject(task: TaskLike): string {
  const p = task.params ?? {};
  switch (task.task_type) {
    case 'ingest_pipeline':
    case 'icml2026_external_pipeline': {
      const venue = p.venue;
      const year = p.year;
      if (venue != null && year != null) return `${venue} ${arxivKeyLabel(venue, year) ?? year}`;
      if (venue != null) return String(venue);
      return '';
    }
    case 'crawl_urls': {
      const urls = Array.isArray(p.urls) ? (p.urls as unknown[]) : [];
      if (urls.length === 1) return String(urls[0]);
      if (urls.length > 1) return `${urls.length} 个网址`;
      return '';
    }
    case 'match_paper': {
      // Users do not care about ids: the backend params only carry paper_id
      // (no title), so this is the venue+year fallback — see
      // `paperVenueYearFallback`. The JSX layer prefers a real title fetched
      // by paper_id when that request resolves (Tasks.tsx `useTaskSubjectDisplay`).
      const paperId = p.paper_id;
      return paperId != null ? paperVenueYearFallback(String(paperId)) : '';
    }
    case 'upload_pipeline': {
      // The backend renames every upload to a UUID before it ever reaches
      // `file_path` (app/services/pipeline/upload_service.py), so its
      // basename is never a meaningful subject — always empty here. The real
      // document title is fetched by `doc_id` at the JSX layer instead.
      return '';
    }
    case 'journal_index':
    case 'code_index':
    case 'code_duplicates':
      return typeof p.project === 'string' ? p.project : '';
    case 'public_code_job': {
      const ids = stringList(p.source_ids);
      if (ids.length === 1) return ids[0];
      if (ids.length > 1) return `${ids.length} 个来源`;
      return '';
    }
    case 'medical_corpus_download': {
      const limit = p.download_limit;
      return typeof limit === 'number' && limit > 0 ? `限 ${limit} 篇` : '全部待下载';
    }
    default:
      return '';
  }
}
