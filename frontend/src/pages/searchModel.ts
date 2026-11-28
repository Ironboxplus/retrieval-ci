// 检索页（论文模式）的纯逻辑：URL 状态、请求体、端点选择、重查触发条件。
// 从 Search.tsx 拆出，便于 vitest 直接测；与 codeSearchModel.ts 同样的分工。
import type { RetrievalMethod, SearchRequest } from '../types';
import { searchCacheKey } from '../api/hooks/useSearchStream';
import { includeWithdrawnField, readIncludeWithdrawn, writeIncludeWithdrawn } from './withdrawnModel';

// 'upload' 是上传文档（整本书走这一类）。与文档库页的「论文/网页/上传」三类对齐。
// 'arxiv' 是预印本，走独立端点 /arxiv/search/stream（范围固定为 venue_tier=preprint）。
export type SourceTypeFilter = 'all' | 'paper' | 'arxiv' | 'webpage' | 'upload';
export type SortBy = 'score' | 'year' | 'citations';

export const PAPER_SEARCH_ENDPOINT = '/search/stream';
export const ARXIV_SEARCH_ENDPOINT = '/arxiv/search/stream';

export type SearchUIState = {
  query: string;
  method: RetrievalMethod;
  topK: number;
  venues: string[];
  years: number[];
  titleOnly: boolean;
  includeWithdrawn: boolean;
  chunkMode: boolean;
  sourceType: SourceTypeFilter;
  sortBy: SortBy;
};

// 请求体：SearchRequest 之外多一个后端已有的 include_preprints（types 里没有）。
export type PaperSearchBody = SearchRequest & { include_preprints?: boolean };

// sourceType 默认 'paper'：论文有 10.6 万篇、1394 万 chunk，upload/webpage 合计不到
// 4 千个点，混在一起检索时后者几乎永远进不了 top_k，'all' 这个默认值等于「只搜论文」
// 但不告诉用户。显式默认成 paper，让「搜书」成为一次有意识的切换。
// titleOnly 默认关闭。它只匹配 L1 块（标题+摘要，无摘要时回退成标题+正文前 N 个
// token），开着等于把正文整个排除在检索之外。以前 sourceType 默认是 'all'，
// uiToRequest 在非 paper 下根本不发 title_only，后端默认 false——所以默认关闭才是
// 一直以来的实际行为；把 sourceType 默认改成 'paper' 时它被顺带打开了。
export const DEFAULT_UI: SearchUIState = {
  query: '', method: 'hybrid', topK: 20, titleOnly: false, includeWithdrawn: false, chunkMode: false,
  venues: [], years: [], sourceType: 'paper', sortBy: 'score',
};

export function readUI(sp: URLSearchParams): SearchUIState {
  const num = (k: string, d?: number) => {
    const v = sp.get(k);
    if (v == null || v === '') return d;
    const n = Number(v);
    return Number.isFinite(n) ? n : d;
  };
  // Back-compat: legacy `year=2024` (single) still hydrates as [2024].
  const yearsParam = sp.get('years');
  let years: number[] = [];
  if (yearsParam) {
    years = yearsParam.split(',').map((s) => Number(s)).filter((n) => Number.isFinite(n));
  } else {
    const legacyYear = num('year');
    if (legacyYear != null) years = [legacyYear];
  }
  return {
    query: sp.get('q') ?? '',
    method: ((sp.get('method') as RetrievalMethod) || 'hybrid'),
    topK: num('topK', 20) ?? 20,
    venues: sp.get('venues')?.split(',').filter(Boolean) ?? [],
    years,
    // 跟着 DEFAULT_UI 翻过来：缺省即关闭，'1' 才打开。旧链接里的 titleOnly=0 仍然
    // 读作关闭，语义不变；变的只是「不带这个参数」从开变成了关。
    titleOnly: sp.get('titleOnly') === '1',
    includeWithdrawn: readIncludeWithdrawn(sp),
    chunkMode: sp.get('chunk') === '1',
    sourceType: ((sp.get('source') as SourceTypeFilter) || DEFAULT_UI.sourceType),
    sortBy: ((sp.get('sortBy') as SortBy) || 'score'),
  };
}

export function writeUI(ui: SearchUIState): URLSearchParams {
  const sp = new URLSearchParams();
  if (ui.query) sp.set('q', ui.query);
  if (ui.method !== DEFAULT_UI.method) sp.set('method', ui.method);
  if (ui.topK !== DEFAULT_UI.topK) sp.set('topK', String(ui.topK));
  if (ui.venues.length) sp.set('venues', ui.venues.join(','));
  if (ui.years.length) sp.set('years', ui.years.slice().sort((a, b) => a - b).join(','));
  if (ui.titleOnly) sp.set('titleOnly', '1');
  writeIncludeWithdrawn(sp, ui.includeWithdrawn);
  if (ui.chunkMode) sp.set('chunk', '1');
  if (ui.sourceType !== DEFAULT_UI.sourceType) sp.set('source', ui.sourceType);
  if (ui.sortBy !== DEFAULT_UI.sortBy) sp.set('sortBy', ui.sortBy);
  return sp;
}

export function uiToRequest(ui: SearchUIState): PaperSearchBody {
  const isPaper = ui.sourceType === 'paper';
  const isArxiv = ui.sourceType === 'arxiv';
  return {
    query: ui.query, top_k: ui.topK, method: ui.method,
    // arXiv 走独立端点，范围由后端固定；source_type 与 venue 会被它覆盖，不发。
    source_type: ui.sourceType === 'all' || isArxiv ? undefined : ui.sourceType,
    // 「全部」要真的包含 arXiv 预印本；后端默认排除。
    include_preprints: ui.sourceType === 'all' ? true : undefined,
    // venue 只有「论文」才发。上传文档和网页的 payload 里没有这个字段
    // （papers_payload.py 的 upload/webpage profile），正向过滤等于空集——
    // 实测 source_type=upload + venue=["ICML"] 返回 candidates:0、rerank skipped。
    // 筛选面板与覆盖卡片只在 paper 下渲染，用户切到「上传」后既看不见也清不掉残留的会议选择。
    venue: isPaper && ui.venues.length ? ui.venues : undefined,
    // year 论文与 arXiv 都有（arXiv 预印本带 year）。
    year: (isPaper || isArxiv) && ui.years.length ? ui.years : undefined,
    // title_only 匹配的是 L1 标题 chunk，对上传文档/网页没有对应物：实测
    // source_type=upload + title_only=true 只能返回封面页且 score 0.0，而 false 时
    // 同一 query 得到 0.983 的正文。开关只在 paper/arXiv 下渲染，其他来源下关不掉它。
    title_only: isPaper || isArxiv ? ui.titleOnly : undefined,
    // 撤稿只存在于 arXiv 预印本；后端默认排除，开关打开才带回。随请求体进缓存键。
    include_withdrawn: includeWithdrawnField(ui.sourceType, ui.includeWithdrawn),
    // 展开片段 = 关闭按论文去重（每篇返回全部命中 chunk）；默认 undefined（后端默认
    // dedup=true，每篇仅最高分）保持缓存键干净。结果始终按论文分组排序，与此开关无关。
    dedup_by_paper: ui.chunkMode ? false : undefined,
  };
}

/** 来源对应的流式检索端点。 */
export function searchEndpoint(sourceType: SourceTypeFilter): string {
  return sourceType === 'arxiv' ? ARXIV_SEARCH_ENDPOINT : PAPER_SEARCH_ENDPOINT;
}

/**
 * 「改控件就重查」的触发值：query 为空时为 null（不查）；否则是端点+请求体的序列化
 * （与缓存键同一个函数）。它变了就要重新发请求；只改 sortBy（前端排序）不会变。
 */
export function searchTrigger(ui: SearchUIState): string | null {
  if (!ui.query.trim()) return null;
  return searchCacheKey(uiToRequest(ui), searchEndpoint(ui.sourceType));
}
