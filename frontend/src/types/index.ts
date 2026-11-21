// --- System ---
export interface StatsResponse {
  total_documents: number;
  total_papers?: number;
  total_chunks: number;
  coverage_ratio?: number;
  chunk_levels?: Record<string, number>;
  source_type_stats?: Record<string, number>;
  vector_index_size: number;
  embedding_model?: string;
  embedding_dimension?: number;
  supports_sparse?: boolean;
}

// --- Search ---
export type RetrievalMethod = 'vector' | 'hybrid' | 'sparse';

export interface SearchRequest {
  query: string;
  method?: RetrievalMethod;
  top_k?: number;
  filters?: Record<string, unknown>;
  source_type?: string;
  data_source?: string;
  venue?: string | string[];
  year?: number | number[];
  venue_tier?: string;
  title_only?: boolean;
  // 包含已撤稿的 arXiv 论文；默认（缺省）排除。按 paper_ids 限定的请求不受影响。
  include_withdrawn?: boolean;
  // 每篇论文卡片展示几个 chunk：true(默认)=只保留最高分；false=展开该论文全部命中片段。
  // 无论取值，结果始终按「论文最高分」分组排序（粒度=论文）。
  dedup_by_paper?: boolean;
  // 限定检索范围到指定论文（paper_id）。配合 dedup_by_paper=false + 大 top_k 可拿到
  // 单篇论文的全部命中片段及分数（「在此论文中检索」）。
  paper_ids?: string[];
}

export interface PipelineMetricsData {
  embed_ms: number;
  retrieve_ms: number;
  rerank_ms: number;
  total_ms: number;
  candidates_before_rerank: number;
  candidates_after_rerank: number;
  mode: string;
}

export interface VenueAggregation {
  venue: string;
  count: number;
  avg_score: number;
}

export interface YearAggregation {
  year: number;
  count: number;
  avg_score: number;
}

export interface SearchResult {
  doc_id: number;
  chunk_id: number;
  content: string;
  score: number;
  doc_title: string;
  source: string;
  metadata: Record<string, unknown>;
}

export interface AggregatedResult {
  doc_id: number;
  doc_title: string;
  source: string;
  hit_count: number;
  total_score: number;
  avg_score: number;
  max_score: number;
  chunks: SearchResult[];
}

// 按论文分组的检索结果（粒度=论文，按 max_score 排序）。前端以 papers 为主渲染卡片。
export interface PaperGroup {
  paper_id: string;
  doc_id: string;
  doc_title: string;
  source: string;
  max_score: number;      // 该论文命中 chunk 的最高分 —— 论文排序依据
  chunk_count: number;    // 本卡片携带的 chunk 数（= chunks.length）
  metadata: Record<string, unknown>;  // 论文级元数据（取自最高分 chunk）
  chunks: SearchResult[]; // 该论文命中的片段，已按分降序
}

export interface SearchResponse {
  query: string;
  method: string;
  total_results: number;          // = results.length（扁平 chunk 数）
  results: SearchResult[];        // 命中 chunk 扁平列表（向后兼容）
  papers?: PaperGroup[];          // 论文分组视图（前端主渲染）
  aggregated?: AggregatedResult[];
  pipeline_metrics?: PipelineMetricsData;
  venue_distribution?: VenueAggregation[];
  year_distribution?: YearAggregation[];
}

// --- Match Doc (功能2：doc → 最相近论文) ---
export type MatchInputType = 'text' | 'pdf' | 'paper_id';
export type MatchAgg = 'coverage' | 'maxsim';

export interface MatchDocRequest {
  input_type: MatchInputType;
  text?: string;
  upload_task_id?: string;
  paper_id?: string;
  top_k?: number;
  per_chunk_k?: number;
  agg?: MatchAgg;
  sim_threshold?: number;
  filters?: Record<string, unknown>;
  max_query_chunks?: number;
}

export interface QueryChunk {
  idx: number;
  text: string;
  level?: number | null;
}

export interface MatchedSpan {
  query_chunk_idx: number;
  paper_chunk: string;
  sim: number;
}

export interface PaperMatch {
  paper_id: string;
  doc_id: string;
  title: string;
  venue: string;
  year?: number | null;
  score: number;
  coverage: number;
  metadata: Record<string, unknown>;
  matched: MatchedSpan[];
}

export interface MatchDocResponse {
  input_type: string;
  query_chunks: QueryChunk[];
  papers: PaperMatch[];
  metrics: Record<string, unknown>;
}

// --- Papers ---
export interface PaperSearchRequest {
  query: string;
  top_k?: number;
  method?: RetrievalMethod;
  aggregate_by_paper?: boolean;
  filters?: Record<string, unknown>;
}

export interface PaperSearchResult {
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string;
  source: string;
  venue: string;
  year: number;
  paper_type: string;
  score: number;
  hit_count: number;
  pdf_url?: string;
  url?: string;
  matched_chunks: SearchResult[];
}

export interface PaperSearchResponse {
  query: string;
  method: string;
  total_results: number;
  papers: PaperSearchResult[];
}

export interface PaperBrowseParams {
  /** Single string or array (axios serializes arrays as repeated query keys). */
  venue?: string | string[];
  /** Single number or array. */
  year?: number | number[];
  venue_tier?: string;
  track_type?: string;
  impact_level?: string;
  /** Keyword — matched against title + abstract (case-insensitive substring). */
  q?: string;
  /** Include withdrawn arXiv papers; omitted = hidden. */
  include_withdrawn?: boolean;
  page?: number;
  page_size?: number;
  sort_by?: string;
}

export interface PaperBrief {
  paper_id: string;
  title: string;
  authors: string[];
  venue: string;
  year: number;
  paper_type?: string;
  venue_tier?: string;
  track_type?: string;
  impact_level?: string;
  source?: string;
  url?: string;
  pdf_url?: string;
  doc_id?: number;
  indexed?: boolean;
  /** arXiv only: withdrawn upstream (SQL is authoritative). */
  withdrawn?: boolean;
  /** Version the stored text belongs to, e.g. "v2"; absent for conference papers. */
  source_version?: string | null;
}

export interface PaperBrowseResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  results: PaperBrief[];
}

export interface TagsResponse {
  total_papers: number;
  venues: string[];
  years: number[];
  venue_tiers: string[];
  track_types: string[];
  impact_levels: string[];
  venue_counts: Record<string, number>;
  year_counts: Record<string, number>;
}

export interface PaperStats {
  total_papers: number;
  total_documents: number;
  total_chunks: number;
  chunk_levels: Record<string, number>;
  vector_index_size: number;
  bm25_index_size: number;
  collection_name: string;
}

// --- Documents ---
export interface DocumentBrief {
  id: number;
  title: string;
  source: string;
  source_type: string;
  data_source: string;
  status: string;
  url?: string;
  file_path?: string;
  chunk_count: number;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
  download_url?: string;
  source_file_path?: string;
  source_download_url?: string;
}

export interface DocumentDetail extends DocumentBrief {
  chunks: ChunkBrief[];
}

export interface ChunkBrief {
  id: number;
  level: number;
  chunk_index: number;
  content_preview: string;
  token_count: number;
}

export interface DocumentListResponse {
  total: number;
  page: number;
  page_size: number;
  documents: DocumentBrief[];
}

// --- Index Status (shared shape for /papers/{id}/index-status and /documents/{id}/index-status) ---
export interface PaperIndexStatus {
  paper_id: string;
  doc_id: number | null;
  sql_chunks_count: number;
  qdrant_points_count: number;
  indexed: boolean;
}

export interface DocumentIndexStatus {
  document_id: number;
  sql_chunks_count: number;
  qdrant_points_count: number;
  indexed: boolean;
}

export interface IndexActionResponse {
  paper_id?: string;
  doc_id?: number;
  status: string;
  chunks_embedded?: number;
  note?: string;
}

// --- Crawl ---
export interface CrawlUrlsRequest {
  urls: string[];
  use_proxy?: boolean;
  auto_index?: boolean;
  max_concurrent?: number;
}

export interface CrawlRecord {
  id: number;
  task_id: number | null;
  url: string;
  status: string;
  title?: string | null;
  content_length?: number | null;
  doc_id?: number | null;
  error_message?: string | null;
  status_code?: number | null;
  retry_count?: number | null;
  crawl_time?: number | null;
  proxy_used?: string | null;
  created_at: string;
}

export interface CrawlRecordListResponse {
  total: number;
  page: number;
  page_size: number;
  records: CrawlRecord[];
}

// --- Tasks ---
export interface TaskBrief {
  id: number;
  task_type: string;
  status: string;
  priority: number;
  progress: number;
  total: number;
  error_message?: string;
  retry_count: number;
  max_retries: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  worker_id?: string;
  last_heartbeat?: string;
  duration_seconds?: number;
  params: Record<string, unknown>;
  result?: Record<string, unknown>;
  /** Running ingest pipelines only (list view): see app/services/pipeline/progress.py. */
  pipeline?: PipelineProgress | null;
}

export interface PipelineProgress {
  /** 0..1, stages weighted by their typical duration. */
  fraction: number;
  /** Work-only remainder; null while parked on a stage lock. */
  eta_seconds: number | null;
  stage: { key: string; name: string; status: string; progress: number; total: number } | null;
  waiting: boolean;
}

export interface TaskHealth {
  status: string;
  heartbeat_age: number | null;
  is_healthy: boolean;
  issues: string[];
  duration_seconds?: number;
}

export interface TaskDetail extends TaskBrief {
  health: TaskHealth;
  resource_summary?: Record<string, unknown>;
  subtask_summary?: Record<string, unknown>;
  recent_events: TaskEvent[];
}

export interface TaskEvent {
  id: number;
  task_id: number;
  subtask_id?: number;
  event_type: string;
  level: string;
  stage?: string;
  message: string;
  payload?: Record<string, unknown>;
  created_at: string;
}

export interface TaskListResponse {
  total: number;
  page: number;
  page_size: number;
  tasks: TaskBrief[];
  /** Seconds until every running ingest pipeline is done (stage-lane backlog). */
  pipeline_eta_seconds?: number | null;
}

export interface TaskSubtask {
  id: number;
  task_id: number;
  subtask_key: string;
  subtask_type: string;
  name: string;
  status: string;
  progress: number;
  total: number;
  retry_count: number;
  max_retries: number;
  message?: string;
  stage?: string;
  error_message?: string;
  metadata?: Record<string, unknown>;
  result?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface WorkflowResponse {
  task_id: number;
  task_type: string;
  task_status: string;
  summary: Record<string, unknown>;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface WorkflowNode {
  id: number;
  subtask_key?: string | null;
  /** The stable stage key (`download`, `extract`, ...) the workflow endpoint sends. */
  key?: string | null;
  name: string;
  status: string;
  progress: number;
  total: number;
}

export interface WorkflowEdge {
  id: number;
  key: string;
  source: string;
  target: string;
  label: string;
  status: string;
  metadata?: Record<string, unknown>;
  allowed_actions: string[];
  created_at: string;
  updated_at: string;
}

// --- Generic ---
export interface ActionResponse {
  success: boolean;
  message: string;
  task_id?: string | number;
}

// --- Auth / Account (付费层 gateway, PUBLIC_MODE only) ---
// Session-cookie auth via the BWH gateway; the SPA never sees the sk-key hash.
// A single bound login method. One account may hold several (cdk / linuxdo /
// google) — all sharing one balance (账号绑定).
export interface Identity {
  id: number;
  provider: string;   // 'cdk' | 'linuxdo' | 'google'
  sub: string;
  created_at?: string | null;
}

export interface Me {
  id: number;
  username: string;
  tier: string;       // 'paying' | 'internal' | 'admin'
  balance: number;    // remaining credits
  identities?: Identity[];
}

export interface ApiKeyInfo {
  id: number;
  prefix: string;
  last4: string;
  revoked: boolean;
  last_used_at: string | null;
}

// Plaintext sk-key, shown exactly ONCE on creation.
export interface ApiKeyCreated {
  id: number;
  key: string;
  prefix: string;
  last4: string;
}

export interface TopupResult {
  out_trade_no: string;
  pay_url: string;
}
