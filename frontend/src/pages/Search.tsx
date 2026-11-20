import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { Link, useSearchParams, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Input, Card, Space, Tag, Typography, Slider, Row, Col,
  Empty, Progress, Tooltip, Button, Steps, Switch, Segmented, theme, message,
} from 'antd';
import {
  LinkOutlined, FileTextOutlined, UserOutlined, SearchOutlined,
  FilterOutlined, BarChartOutlined,
  RocketOutlined, CheckCircleOutlined, LoadingOutlined,
  ExperimentOutlined, SortAscendingOutlined, FontSizeOutlined,
  FilePdfOutlined, ReadOutlined, AppstoreOutlined, GlobalOutlined,
  CopyOutlined, HistoryOutlined, StopOutlined,
} from '@ant-design/icons';
import { useSearchStream } from '../api/hooks/useSearchStream';
import type { StreamStage, StreamProgress } from '../api/hooks/useSearchStream';
import { useAuthUI } from '../stores/authStore';
import { PUBLIC_MODE } from '../constants';
import api from '../api/client';
import type { RetrievalMethod, SearchResponse, SearchRequest, PipelineMetricsData, PaperBrief, PaperGroup } from '../types';
import ChunkScoreView from '../components/ChunkScoreView';
import MarkdownSnippet from '../components/MarkdownSnippet';
import { groupPapersClientSide } from '../components/groupPapers';
import PaperFilterPanel, { venueColors, type PaperFilterValue } from '../components/PaperFilterPanel';
import IndexCoveragePanel from '../components/IndexCoveragePanel';
import { copyText } from '../utils/clipboard';
import CodeSearchView from './CodeSearchView';
import ModeSwitch from '../components/ModeSwitch';
import { useIsWide } from '../hooks/useIsWide';
import {
  includeWithdrawnField, readIncludeWithdrawn, withdrawnMarks, writeIncludeWithdrawn,
} from './withdrawnModel';

// Same breakpoint and stacked-vs-same-row structure as CodeSearchView.tsx's
// own NARROW_BREAKPOINT (round-3 review, 390px) — kept identical between the
// two modes so switching 论文|代码 never jumps the header layout.
const NARROW_BREAKPOINT = 576;

const SEARCH_HISTORY_KEY = 'retrival_search_history';
const MAX_HISTORY = 20;
type HistoryEntry = { query: string; ts: number };
function getSearchHistory(): HistoryEntry[] {
  try { return JSON.parse(localStorage.getItem(SEARCH_HISTORY_KEY) || '[]'); } catch { return []; }
}
function saveSearchHistory(query: string) {
  const hist = getSearchHistory().filter((h) => h.query !== query);
  hist.unshift({ query, ts: Date.now() });
  localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(hist.slice(0, MAX_HISTORY)));
}
function clearSearchHistory() { localStorage.removeItem(SEARCH_HISTORY_KEY); }

// ─── URL sync helpers ───
// 'upload' 是上传文档（整本书走这一类）。与文档库页的「论文/网页/上传」三类对齐。
type SourceTypeFilter = 'all' | 'paper' | 'webpage' | 'upload';

type SearchUIState = {
  query: string;
  method: RetrievalMethod;
  topK: number;
  venues: string[];
  years: number[];
  titleOnly: boolean;
  includeWithdrawn: boolean;
  chunkMode: boolean;
  sourceType: SourceTypeFilter;
  sortBy: 'score' | 'year' | 'citations';
};

// sourceType 默认 'paper'：论文有 10.6 万篇、1394 万 chunk，upload/webpage 合计不到
// 4 千个点，混在一起检索时后者几乎永远进不了 top_k，'all' 这个默认值等于「只搜论文」
// 但不告诉用户。显式默认成 paper，让「搜书」成为一次有意识的切换。
// titleOnly 默认关闭。它只匹配 L1 块（标题+摘要，无摘要时回退成标题+正文前 N 个
// token），开着等于把正文整个排除在检索之外。以前 sourceType 默认是 'all'，
// uiToRequest 在非 paper 下根本不发 title_only，后端默认 false——所以默认关闭才是
// 一直以来的实际行为；把 sourceType 默认改成 'paper' 时它被顺带打开了。
const DEFAULT_UI: SearchUIState = {
  query: '', method: 'hybrid', topK: 20, titleOnly: false, includeWithdrawn: false, chunkMode: false,
  venues: [], years: [], sourceType: 'paper', sortBy: 'score',
};

function readUI(sp: URLSearchParams): SearchUIState {
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
    sortBy: ((sp.get('sortBy') as 'score' | 'year' | 'citations') || 'score'),
  };
}

function writeUI(ui: SearchUIState): URLSearchParams {
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

function uiToRequest(ui: SearchUIState): SearchRequest {
  return {
    query: ui.query, top_k: ui.topK, method: ui.method,
    source_type: ui.sourceType === 'all' ? undefined : ui.sourceType,
    // venue/year 同 title_only：只有「论文」才发。上传文档和网页的 payload 里没有
    // 这两个字段（papers_payload.py 的 upload/webpage profile），正向过滤等于空集——
    // 实测 source_type=upload + venue=["ICML"] 返回 candidates:0、rerank skipped。
    // 而筛选面板与覆盖卡片都只在 paper 下渲染，用户切到「上传」后既看不见也清不掉
    // 残留的会议选择，只会看到「什么都检索不出来」。
    venue: ui.sourceType === 'paper' && ui.venues.length ? ui.venues : undefined,
    year: ui.sourceType === 'paper' && ui.years.length ? ui.years : undefined,
    // 只有「论文」才发 title_only。它匹配的是 L1 标题 chunk，对上传文档/网页没有对应
    // 物：实测 source_type=upload + title_only=true 只能返回封面页且 score 0.0，而
    // false 时同一 query 得到 0.983 的正文。开关本身也只在 paper 下渲染，用户在其他
    // 来源下关不掉它——照发就是发一个必然为空的检索。
    title_only: ui.sourceType === 'paper' ? ui.titleOnly : undefined,
    // 撤稿只存在于 arXiv 论文；后端默认排除，开关打开才带回。随请求体进缓存键。
    include_withdrawn: includeWithdrawnField(ui.sourceType, ui.includeWithdrawn),
    // 展开片段 = 关闭按论文去重（每篇返回全部命中 chunk）；默认 undefined（后端默认
    // dedup=true，每篇仅最高分）保持缓存键干净。结果始终按论文分组排序，与此开关无关。
    dedup_by_paper: ui.chunkMode ? false : undefined,
  };
}

// ─── Pipeline Visualization ───
const STAGE_INDEX: Record<StreamStage, number> = {
  idle: -1, embed: 0, retrieve: 1, rerank: 2, enrich: 3, complete: 4, error: -1,
};

function PipelineVis({
  metrics, searching, stage, progress,
}: {
  metrics?: PipelineMetricsData;
  searching: boolean;
  stage: StreamStage;
  progress: StreamProgress;
}) {
  const { token } = theme.useToken();
  const currentStep = metrics ? 4 : STAGE_INDEX[stage] ?? -1;
  if (currentStep < 0 && !metrics) return null;

  const embedMs = metrics?.embed_ms ?? progress.embed_ms;
  const retrieveMs = metrics?.retrieve_ms ?? progress.retrieve_ms;
  const retrieveCandidates = metrics?.candidates_before_rerank ?? progress.retrieve_candidates;
  const rerankMs = metrics?.rerank_ms ?? progress.rerank_ms;
  const rerankBefore = metrics?.candidates_before_rerank ?? progress.rerank_candidates_before;
  const rerankAfter = metrics?.candidates_after_rerank ?? progress.rerank_candidates_after;
  const mode = metrics?.mode ?? progress.mode;
  const enriched = progress.enriched_count;
  const totalMs = metrics?.total_ms;

  const stages = [
    { title: '向量编码', desc: embedMs != null ? `${embedMs.toFixed(0)}ms` : (searching ? '编码查询...' : ''), icon: <ExperimentOutlined /> },
    {
      title: mode === 'dense' ? '语义检索' : mode === 'sparse' ? '稀疏检索' : '混合检索',
      desc: retrieveMs != null ? `${retrieveMs.toFixed(0)}ms · ${retrieveCandidates ?? '?'} 候选` : (searching ? 'Dense + Sparse RRF...' : ''),
      icon: <SearchOutlined />,
    },
    {
      title: '精排重排',
      desc: rerankMs != null
        ? (rerankMs > 0 ? `${rerankMs.toFixed(0)}ms · ${rerankBefore ?? '?'}→${rerankAfter ?? '?'}` : '跳过')
        : (searching ? '重排候选...' : ''),
      icon: <SortAscendingOutlined />,
    },
    {
      title: '结果返回',
      desc: totalMs != null ? `总耗时 ${totalMs.toFixed(0)}ms` : (enriched != null ? `回填 ${enriched} 条 metadata...` : (searching ? '回填 metadata...' : '')),
      icon: <CheckCircleOutlined />,
    },
  ];

  return (
    <Card size="small" style={{ borderRadius: 10, background: token.colorBgElevated }} styles={{ body: { padding: '12px 16px' } }}>
      <Typography.Text strong style={{ fontSize: 12, color: token.colorTextSecondary, marginBottom: 8, display: 'block' }}>
        <RocketOutlined style={{ marginRight: 4 }} /> 检索管线
      </Typography.Text>
      <Steps
        size="small"
        current={currentStep}
        status={searching ? 'process' : 'finish'}
        items={stages.map((s, i) => ({
          title: <span style={{ fontSize: 12 }}>{s.title}</span>,
          description: <span style={{ fontSize: 11, color: token.colorTextSecondary }}>{currentStep >= i ? s.desc : ''}</span>,
          icon: currentStep === i && searching ? <LoadingOutlined /> : s.icon,
        }))}
      />
    </Card>
  );
}

// ─── Distribution Summary ───
function DistributionBar({ data }: { data: SearchResponse }) {
  const { token } = theme.useToken();
  const venueDist = data.venue_distribution || [];
  const yearDist = data.year_distribution || [];
  const totalCount = venueDist.reduce((s, v) => s + v.count, 0) || 1;
  if (!venueDist.length && !yearDist.length) return null;

  return (
    <Card size="small" style={{ borderRadius: 10 }} styles={{ body: { padding: '12px 16px' } }}>
      <Row gutter={24}>
        <Col span={14}>
          <Typography.Text strong style={{ fontSize: 12, color: token.colorTextSecondary, marginBottom: 6, display: 'block' }}>
            <BarChartOutlined style={{ marginRight: 4 }} /> 会议分布
          </Typography.Text>
          <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', marginBottom: 6 }}>
            {venueDist.map((v) => (
              <Tooltip key={v.venue} title={`${v.venue}: ${v.count} 篇 (avg ${v.avg_score.toFixed(3)})`}>
                <div style={{
                  width: `${(v.count / totalCount) * 100}%`,
                  background: venueColors[v.venue] || token.colorPrimary,
                  minWidth: v.count > 0 ? 3 : 0,
                  transition: 'width 0.3s',
                }} />
              </Tooltip>
            ))}
          </div>
          <Space size={4} style={{ flexWrap: 'wrap' }}>
            {venueDist.slice(0, 6).map((v) => (
              // Bounded width + ellipsis + title: same defensive fix as
              // code mode's CodeDistributionPanel (F2-c round 2) — a venue
              // name is short in practice, but nothing upstream guarantees
              // that, and `Space wrap` alone doesn't shrink one chip that's
              // individually wider than the row.
              <Tag key={v.venue} color={venueColors[v.venue]} title={v.venue}
                   style={{ borderRadius: 4, fontSize: 11, margin: 0, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {v.venue} {v.count}
              </Tag>
            ))}
          </Space>
        </Col>
        <Col span={10}>
          <Typography.Text strong style={{ fontSize: 12, color: token.colorTextSecondary, marginBottom: 6, display: 'block' }}>
            年份分布
          </Typography.Text>
          <Space size={8}>
            {yearDist.map((y) => (
              <div key={y.year} style={{ textAlign: 'center' }}>
                <div style={{
                  width: 32,
                  height: Math.max(8, Math.min(40, y.count * 4)),
                  background: token.colorPrimary,
                  borderRadius: 4,
                  margin: '0 auto 4px',
                  opacity: 0.7 + (y.avg_score * 0.3),
                }} />
                <div style={{ fontSize: 10, color: token.colorTextSecondary }}>{y.year}</div>
                <div style={{ fontSize: 10, fontWeight: 600, color: token.colorText }}>{y.count}</div>
              </div>
            ))}
          </Space>
        </Col>
      </Row>
    </Card>
  );
}

// ─── Title highlight ───
function highlight(text: string, query: string): string {
  if (!query || !text) return text;
  const words = query.split(/\s+/).filter((w) => w.length > 1);
  if (!words.length) return text;
  const regex = new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  return text.replace(regex, '<mark style="background:rgba(79,70,229,0.2);padding:1px 3px;border-radius:3px">$1</mark>');
}

// ─── Paper Card (grouped: one card per paper, ranked by best chunk score) ───
function PaperCard({ p, rank, query }: { p: PaperGroup; rank: number; query: string }) {
  const { token } = theme.useToken();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [hoverDetail, setHoverDetail] = useState<PaperBrief | null>(null);
  const prefetchedRef = useRef(false);

  const meta = (p.metadata || {}) as Record<string, unknown>;
  const paperId = p.paper_id || (meta.paper_id as string | undefined) || undefined;
  const sourceType = (meta.source_type as string | undefined) ?? (paperId ? 'paper' : undefined);
  const isWebpage = sourceType === 'webpage';
  const isUpload = sourceType === 'upload';
  const venue = meta.venue as string | undefined;
  const year = meta.year as number | undefined;
  const authors = (meta.authors as string[] | undefined) ?? [];
  const venueTier = meta.venue_tier as string | undefined;
  const trackType = meta.track_type as string | undefined;
  const impactLevel = meta.impact_level as string | undefined;
  const citationCount = meta.citation_count as number | undefined;
  const withdrawnMark = withdrawnMarks(meta);
  const pdfUrl = meta.pdf_url as string | undefined;
  const crawledAt = meta.crawled_at as string | undefined;
  const sourceUrl = (meta.source as string | undefined) || p.source;
  const urlHost = (() => {
    if (!sourceUrl) return undefined;
    try { return new URL(sourceUrl).hostname; } catch { return undefined; }
  })();

  const score = p.max_score;
  const scorePercent = Math.min(Math.round(score * 100), 100);
  const strokeColor = score > 0.9 ? '#10b981' : score > 0.7 ? '#4F46E5' : '#f59e0b';

  // Prefetch paper detail on hover (network only once per card lifetime).
  // Skipped in public mode: opening a paper is metered (0.1), so a hover must NOT
  // silently spend credits — only an explicit click charges.
  const prefetchPaper = useCallback(() => {
    if (PUBLIC_MODE) return;
    if (!paperId || prefetchedRef.current) return;
    prefetchedRef.current = true;
    qc.prefetchQuery({
      queryKey: ['papers', paperId],
      queryFn: () => api.get<PaperBrief>(`/papers/${paperId}`).then((res) => res.data),
      staleTime: 60_000,
    }).then(() => {
      const cached = qc.getQueryData<PaperBrief>(['papers', paperId]);
      if (cached) setHoverDetail(cached);
    });
  }, [paperId, qc]);

  const abstract = ((hoverDetail as unknown as { metadata?: Record<string, unknown> })?.metadata?.abstract as string | undefined)
    ?? (hoverDetail as unknown as { abstract?: string })?.abstract;
  const [showAbstract, setShowAbstract] = useState(false);

  const highlightedTitle = useMemo(() => highlight(p.doc_title, query), [p.doc_title, query]);

  // Jump into the paper detail page and run an in-paper search for this query —
  // shows every chunk's match against the query with the score heatmap (W87).
  const goInPaper = useCallback(() => {
    if (paperId) navigate(`/papers/${paperId}?tab=chunks&q=${encodeURIComponent(query)}`);
  }, [paperId, navigate, query]);

  return (
    <Card
      size="small"
      className="result-card"
      style={{
        marginBottom: 10,
        borderLeft: `4px solid ${venue ? (venueColors[venue] || '#a5b4fc') : '#a5b4fc'}`,
        borderRadius: 10,
      }}
      onMouseEnter={prefetchPaper}
      styles={{ body: { padding: '12px 16px' } }}
    >
      {/* Row 1: rank + title + paper max-score */}
      <Row gutter={12} wrap={false} align="middle">
        <Col flex="0 0 36px" style={{ textAlign: 'center' }}>
          <Tag style={{ margin: 0, fontWeight: 700, fontSize: 11, background: token.colorPrimary, color: '#fff', border: 'none', borderRadius: 6 }}>
            #{rank}
          </Tag>
        </Col>
        <Col flex="auto" style={{ minWidth: 0 }}>
          {paperId ? (
            <Link to={`/papers/${paperId}`}>
              <Typography.Text strong style={{ fontSize: 15, color: token.colorText, lineHeight: 1.4 }} onMouseEnter={prefetchPaper}>
                <span dangerouslySetInnerHTML={{ __html: highlightedTitle }} />
              </Typography.Text>
            </Link>
          ) : isWebpage && sourceUrl ? (
            <a href={sourceUrl} target="_blank" rel="noreferrer">
              <Typography.Text strong style={{ fontSize: 15, color: token.colorText, lineHeight: 1.4 }}>
                <span dangerouslySetInnerHTML={{ __html: highlightedTitle || sourceUrl }} />
              </Typography.Text>
            </a>
          ) : (
            <Typography.Text strong style={{ fontSize: 15, color: token.colorText }}>
              <span dangerouslySetInnerHTML={{ __html: highlightedTitle }} />
            </Typography.Text>
          )}
        </Col>
        <Col flex="0 0 56px" style={{ textAlign: 'center' }}>
          <Tooltip title={`论文最高分: ${score.toFixed(4)}（共命中 ${p.chunk_count} 段）`}>
            <Progress type="circle" percent={scorePercent} size={44} format={() => score.toFixed(2)} strokeColor={strokeColor} trailColor={token.colorFillSecondary} />
          </Tooltip>
        </Col>
      </Row>

      {/* Row 2: meta tags + hit-count */}
      <Row style={{ marginTop: 8 }}>
        <Col flex="auto" style={{ paddingLeft: 48 }}>
          <Space size={4} style={{ flexWrap: 'wrap' }}>
            <Tag color="volcano" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>
              {p.chunk_count > 1 ? `命中 ${p.chunk_count} 段` : '最高分片段'}
            </Tag>
            {isWebpage && <Tag color="cyan" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>网页</Tag>}
            {/* 上传文档没有 venue/year/paper_id，下面那些标签全不渲染——不标一下就和论文难以区分。 */}
            {isUpload && <Tag color="purple" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>上传</Tag>}
            {isWebpage && urlHost && (
              <Tag style={{ borderRadius: 4, background: token.colorFillSecondary, border: 'none', color: token.colorTextSecondary, fontSize: 11, margin: 0 }}>{urlHost}</Tag>
            )}
            {isWebpage && crawledAt && (
              <Tooltip title={`抓取时间: ${crawledAt}`}>
                <Tag style={{ borderRadius: 4, background: token.colorFillTertiary, border: 'none', color: token.colorTextTertiary, fontSize: 11, margin: 0 }}>抓取于 {crawledAt.slice(0, 10)}</Tag>
              </Tooltip>
            )}
            {venue && <Tag color={venueColors[venue] || '#6366f1'} style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>{venue}</Tag>}
            {year && <Tag style={{ borderRadius: 4, background: token.colorFillSecondary, border: 'none', color: token.colorTextSecondary, fontSize: 11, margin: 0 }}>{year}</Tag>}
            {venueTier && <Tag color="gold" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>{venueTier}</Tag>}
            {withdrawnMark.withdrawn && <Tag color="red" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>已撤稿</Tag>}
            {withdrawnMark.versionLabel && <Tag style={{ borderRadius: 4, background: token.colorFillSecondary, border: 'none', color: token.colorTextSecondary, fontSize: 11, margin: 0 }}>{withdrawnMark.versionLabel}</Tag>}
            {(() => {
              const nonMain: Record<string, string> = {
                poster: '海报', demo: '演示', tutorial: '教程', keynote: '主旨演讲',
                doctoral: '博士论坛', workshop: '研讨会', invited: '特邀', panel: '专题讨论',
                'late-breaking': '最新进展', wip: '进行中工作', abstract: '摘要',
              };
              // Main-conference papers (main/research/short/industry) carry no badge — it's the norm.
              // Non-main tracks get a caution badge so a 2-3pp poster/demo isn't mistaken for a full paper.
              const label = trackType ? nonMain[trackType] : undefined;
              return label ? (
                <Tooltip title="非主会正文（短摘要 / 非研究论文，含金量较低）">
                  <Tag color="volcano" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>非主会 · {label}</Tag>
                </Tooltip>
              ) : null;
            })()}
            {impactLevel && impactLevel !== 'unknown' && <Tag color="purple" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>{impactLevel}</Tag>}
            {citationCount != null && citationCount > 0 && (
              <Tag color="orange" style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>{citationCount} 引用</Tag>
            )}
          </Space>
        </Col>
      </Row>

      {/* Row 3: authors */}
      {authors.length > 0 && (
        <Row style={{ marginTop: 6 }}>
          <Col flex="auto" style={{ paddingLeft: 48 }}>
            <Typography.Text type="secondary" style={{ fontSize: 11 }}>
              <UserOutlined style={{ marginRight: 4 }} />
              {authors.slice(0, 5).join(', ')}
              {authors.length > 5 && (
                <Tooltip title={authors.slice(5).join(', ')}>
                  <span style={{ color: token.colorPrimary, cursor: 'help' }}> +{authors.length - 5}</span>
                </Tooltip>
              )}
            </Typography.Text>
          </Col>
        </Row>
      )}

      {/* Body: matched chunks with score heat (markdown) */}
      <Row style={{ marginTop: 10 }}>
        <Col flex="auto" style={{ paddingLeft: 48, minWidth: 0 }}>
          <ChunkScoreView chunks={p.chunks} query={query} />
        </Col>
      </Row>

      {/* Action row */}
      <Row style={{ marginTop: 6 }}>
        <Col flex="auto" style={{ paddingLeft: 48 }}>
          <Space size={6} style={{ flexWrap: 'wrap' }}>
            {paperId && (
              <Link to={`/papers/${paperId}`}>
                <Button size="small" type="link" icon={<ReadOutlined />} style={{ padding: '0 4px', fontSize: 12 }}>详情</Button>
              </Link>
            )}
            {paperId && (
              <Button size="small" type="link" icon={<SearchOutlined />} style={{ padding: '0 4px', fontSize: 12 }} onClick={goInPaper}>
                在此论文中检索
              </Button>
            )}
            {isWebpage && sourceUrl && (
              <a href={sourceUrl} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                <Button size="small" type="link" icon={<GlobalOutlined />} style={{ padding: '0 4px', fontSize: 12 }}>访问原网页</Button>
              </a>
            )}
            {isWebpage && p.doc_id && (
              <Link to={`/documents?selected=${p.doc_id}`}>
                <Button size="small" type="link" icon={<FileTextOutlined />} style={{ padding: '0 4px', fontSize: 12 }}>文档详情</Button>
              </Link>
            )}
            {pdfUrl && (
              <a href={pdfUrl} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                <Button size="small" type="link" icon={<FilePdfOutlined />} style={{ padding: '0 4px', fontSize: 12 }}>PDF</Button>
              </a>
            )}
            {paperId && (
              <Button size="small" type="link" icon={<CopyOutlined />} style={{ padding: '0 4px', fontSize: 12 }}
                onClick={(e) => {
                  e.stopPropagation();
                  api.get(`/papers/${paperId}/bibtex`).then((res) => {
                    const text = (res.data as { bibtex: string }).bibtex;
                    return copyText(text);
                  }).then(() => message.success('BibTeX 已复制'))
                    .catch(() => message.error('BibTeX 获取或复制失败'));
                }}
              >BibTeX</Button>
            )}
            {paperId && abstract && (
              <Button size="small" type="link" onClick={() => setShowAbstract((v) => !v)} style={{ padding: '0 4px', fontSize: 12 }}>
                {showAbstract ? '收起摘要' : '摘要'}
              </Button>
            )}
            {p.source && !isWebpage && (
              <a href={p.source} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                <Tooltip title={p.source}>
                  <Button size="small" type="link" icon={<LinkOutlined />} style={{ padding: '0 4px', fontSize: 12 }}>来源</Button>
                </Tooltip>
              </a>
            )}
          </Space>
        </Col>
      </Row>

      {/* Optional abstract (from hover prefetch) */}
      {showAbstract && abstract && (
        <Row style={{ marginTop: 10 }}>
          <Col flex="auto" style={{ paddingLeft: 48 }}>
            <div style={{ padding: 10, borderRadius: 6, background: token.colorFillQuaternary, fontSize: 12, lineHeight: 1.7, color: token.colorTextSecondary }}>
              <Typography.Text strong style={{ fontSize: 11, color: token.colorPrimary, display: 'block', marginBottom: 4 }}>论文摘要</Typography.Text>
              <MarkdownSnippet source={abstract} max={600} />
            </div>
          </Col>
        </Row>
      )}
    </Card>
  );
}

// The corpus counts (papers/chunks/vectors) now live only on the status
// capsule's 语料 card and 索引同步 card (one concept, one place) — this page
// no longer repeats them in a text line.

// ─── Main ───
export default function Search() {
  const { token } = theme.useToken();
  const [sp, setSp] = useSearchParams();
  // F2-c: 论文|代码 segmented switch, `?mode=paper|code` (default paper, so
  // every pre-F2 link keeps working unchanged). Paper mode's own JSX below is
  // untouched — only wrapped in `mode === 'paper' &&` — so this can never
  // perturb paper mode's pixels or behavior.
  const mode = (sp.get('mode') === 'code' ? 'code' : 'paper') as 'paper' | 'code';
  const setMode = (next: 'paper' | 'code') => {
    const nsp = new URLSearchParams(sp);
    if (next === 'code') nsp.set('mode', 'code'); else nsp.delete('mode');
    setSp(nsp, { replace: true });
  };
  const isWide = useIsWide(NARROW_BREAKPOINT);
  const ui = useMemo(() => readUI(sp), [sp]);
  // Collapsed by default below the `lg` breakpoint (matches Layout.tsx's
  // sidebar-collapse point) — on a 390px phone the advanced filter panel
  // alone fills the whole first screen if it starts open.
  const [showFilters, setShowFilters] = useState(
    () => typeof window === 'undefined' || window.innerWidth >= 992,
  );

  const [inputValue, setInputValue] = useState(ui.query);
  const lastSyncedUiQuery = useRef(ui.query);
  useEffect(() => {
    if (ui.query !== lastSyncedUiQuery.current) {
      lastSyncedUiQuery.current = ui.query;
      setInputValue(ui.query);
    }
  }, [ui.query]);

  const search = useSearchStream();
  const composingRef = useRef(false);

  // Paid-tier gate (PUBLIC_MODE): the gateway answers 401 (anon) / 402 (no
  // credits) → useSearchStream surfaces them as sentinel errors. Prompt login
  // / open topup, de-duped so we don't re-fire on every render.
  const openAccount = useAuthUI((s) => s.openAccount);
  const openLogin = useAuthUI((s) => s.openLogin);
  const authPromptedRef = useRef<string | null>(null);
  useEffect(() => {
    if (search.stage !== 'error') { authPromptedRef.current = null; return; }
    const err = search.error ?? null;
    if (authPromptedRef.current === err) return;
    authPromptedRef.current = err;
    if (err === 'unauthorized') {
      message.info('检索需登录后使用，新用户登录即注册');
      openLogin();
    } else if (err === 'insufficient_credits') {
      message.warning('余额不足，请充值后再检索');
      openAccount('topup');
    }
  }, [search.stage, search.error, openAccount, openLogin]);

  const patchUI = useCallback((delta: Partial<SearchUIState>) => {
    const next = { ...ui, ...delta };
    setSp(writeUI(next), { replace: true });
  }, [ui, setSp]);

  const lastFiredQuery = useRef<string | null>(null);
  useEffect(() => {
    // F2-c: `q` is shared with code mode's own URL state (codeSearchModel.ts).
    // Every hook in this component still runs regardless of which mode's JSX
    // is shown (rules of hooks), so without this guard, opening a code-mode
    // link fires this paper-mode SSE stream too — wasted, and its long-lived
    // connection was observed to stall the concurrent federated-search
    // request in the same browser tab (shared connection budget).
    if (mode !== 'paper') return;
    if (!ui.query.trim()) return;
    if (lastFiredQuery.current === ui.query) return;
    lastFiredQuery.current = ui.query;
    const req = uiToRequest(ui);
    if (!search.hydrate(req)) {
      search.mutate(req);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ui.query, mode]);

  const [history, setHistory] = useState<HistoryEntry[]>(getSearchHistory);

  const handleSearch = (raw?: string) => {
    if (composingRef.current) return;
    const q = (raw ?? inputValue).trim();
    if (!q) return;
    saveSearchHistory(q);
    setHistory(getSearchHistory());
    if (q === ui.query) {
      search.mutate(uiToRequest(ui));
    } else {
      patchUI({ query: q });
    }
  };

  // Papers (grouped). Backend sends data.papers; fall back to client-side
  // grouping for any legacy cached response that predates the papers field.
  const papers: PaperGroup[] = useMemo(() => {
    const data = search.data;
    if (!data) return [];
    const base = (data.papers && data.papers.length)
      ? data.papers
      : groupPapersClientSide(data.results || []);
    const list = [...base];
    if (ui.sortBy === 'year') {
      list.sort((a, b) => ((b.metadata?.year as number) || 0) - ((a.metadata?.year as number) || 0));
    } else if (ui.sortBy === 'citations') {
      list.sort((a, b) => ((b.metadata?.citation_count as number) || 0) - ((a.metadata?.citation_count as number) || 0));
    } else {
      list.sort((a, b) => b.max_score - a.max_score);
    }
    return list;
  }, [search.data, ui.sortBy]);

  const totalChunks = search.data?.total_results ?? 0;

  return (
    <Space orientation="vertical" size="middle" style={{ width: '100%' }}>
      {mode === 'code' && (
        <CodeSearchView sp={sp} setSp={setSp} showFilters={showFilters} setShowFilters={setShowFilters} mode={mode} setMode={setMode} />
      )}

      {mode === 'paper' && (
      <>
      {/* Search Bar + Filters */}
      <Card style={{ borderRadius: 12 }} styles={{ body: { padding: '16px 20px' } }}>
        <Space orientation="vertical" size="small" style={{ width: '100%' }}>
          {/* F2-c review round 2: the 论文|代码 switch moved inside the card,
              left of the input, same row — it used to float above the card
              as its own element. Everything from here down is byte-for-byte
              what paper mode rendered before F2-c. */}
          <div style={{ display: 'flex', flexDirection: isWide ? 'row' : 'column', gap: 8 }}>
            <ModeSwitch mode={mode} onChange={setMode} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <Input.Search
                placeholder="输入检索关键词，回车或点检索触发..."
                enterButton={<><SearchOutlined /> 检索</>}
                size="large"
                loading={search.isPending}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onCompositionStart={() => { composingRef.current = true; }}
                onCompositionEnd={() => { composingRef.current = false; }}
                onSearch={handleSearch}
                allowClear
              />
            </div>
          </div>
          <AnimatePresence>
          {history.length > 0 && !ui.query && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}
            >
              <HistoryOutlined style={{ color: token.colorTextQuaternary, fontSize: 12 }} />
              {history.slice(0, 8).map((h) => (
                <Tag
                  key={h.ts}
                  style={{ cursor: 'pointer', borderRadius: 4, fontSize: 12 }}
                  onClick={() => { setInputValue(h.query); patchUI({ query: h.query }); }}
                  closable
                  onClose={(e) => {
                    e.preventDefault();
                    const next = getSearchHistory().filter((x) => x.query !== h.query);
                    localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(next));
                    setHistory(next);
                  }}
                >
                  {h.query.length > 30 ? h.query.slice(0, 30) + '…' : h.query}
                </Tag>
              ))}
              <Button type="link" size="small" style={{ fontSize: 11, padding: 0 }}
                onClick={() => { clearSearchHistory(); setHistory([]); }}
              >清除</Button>
            </motion.div>
          )}
          </AnimatePresence>

          <Row justify="space-between" align="middle">
            <Space size="middle" wrap>
              <Segmented
                size="small"
                value={ui.method}
                onChange={(v) => patchUI({ method: v as RetrievalMethod })}
                options={[
                  { value: 'hybrid', label: '混合检索' },
                  { value: 'vector', label: '语义检索' },
                  { value: 'sparse', label: '稀疏检索' },
                ]}
              />
              <Segmented
                size="small"
                value={ui.sourceType}
                onChange={(v) => patchUI({ sourceType: v as SourceTypeFilter })}
                options={[
                  { value: 'all', label: '全部' },
                  { value: 'paper', label: '论文' },
                  { value: 'webpage', label: '网页' },
                  { value: 'upload', label: '上传' },
                ]}
              />
              <Space size={4}>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>Top</Typography.Text>
                <Slider min={5} max={100} value={ui.topK} onChange={(v) => patchUI({ topK: v })} style={{ width: 100, margin: 0 }} />
                <Typography.Text style={{ minWidth: 28, fontFamily: 'monospace', fontWeight: 600, color: token.colorPrimary, fontSize: 12 }}>{ui.topK}</Typography.Text>
              </Space>
              <Space size={4}>
                <AppstoreOutlined style={{ color: ui.chunkMode ? token.colorPrimary : token.colorTextSecondary, fontSize: 14 }} />
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>展开全部片段</Typography.Text>
                <Switch size="small" checked={ui.chunkMode} onChange={(v) => patchUI({ chunkMode: v })} />
              </Space>
              {ui.sourceType === 'paper' && (
                <Space size={4}>
                  <FontSizeOutlined style={{ color: ui.titleOnly ? token.colorPrimary : token.colorTextSecondary, fontSize: 14 }} />
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>仅标题</Typography.Text>
                  <Switch size="small" checked={ui.titleOnly} onChange={(v) => patchUI({ titleOnly: v })} />
                </Space>
              )}
              {ui.sourceType === 'paper' && (
                <Space size={4}>
                  <StopOutlined style={{ color: ui.includeWithdrawn ? token.colorPrimary : token.colorTextSecondary, fontSize: 14 }} />
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>包含撤稿</Typography.Text>
                  <Switch size="small" checked={ui.includeWithdrawn} onChange={(v) => patchUI({ includeWithdrawn: v })} />
                </Space>
              )}
            </Space>
            {ui.sourceType === 'paper' && (
              <Button type="text" size="small" icon={<FilterOutlined />} onClick={() => setShowFilters((s) => !s)} style={{ color: showFilters ? token.colorPrimary : token.colorTextSecondary }}>
                {showFilters ? '收起筛选' : '展开筛选'}
                {(ui.venues.length + ui.years.length) > 0 && ` · ${ui.venues.length + ui.years.length}`}
              </Button>
            )}
          </Row>

          <PaperFilterPanel
            value={{ venues: ui.venues, years: ui.years }}
            onChange={(next: PaperFilterValue) => patchUI({ venues: next.venues, years: next.years })}
            expanded={showFilters && ui.sourceType === 'paper'}
          />
        </Space>
      </Card>

      <IndexCoveragePanel
        visible={ui.sourceType === 'paper'}
        selectedVenues={ui.venues}
        onSelectVenue={(venue) => patchUI({
          venues: ui.venues.includes(venue)
            ? ui.venues.filter((v) => v !== venue)
            : [...ui.venues, venue],
        })}
      />

      <PipelineVis metrics={search.data?.pipeline_metrics} searching={search.isPending} stage={search.stage} progress={search.progress} />

      {search.data && (
        <>
          <DistributionBar data={search.data} />

          <Row justify="space-between" align="middle">
            <Space>
              <Typography.Text style={{ fontSize: 14 }}>
                找到 <strong style={{ color: token.colorPrimary }}>{papers.length}</strong> 篇论文
                <Typography.Text type="secondary" style={{ fontSize: 12, marginLeft: 6 }}>· {totalChunks} 个片段</Typography.Text>
              </Typography.Text>
              {search.data.pipeline_metrics && (
                <Tag style={{ borderRadius: 4, background: token.colorPrimaryBg, color: token.colorPrimary, border: 'none', fontSize: 11 }}>
                  {search.data.pipeline_metrics.total_ms.toFixed(0)}ms
                </Tag>
              )}
            </Space>
            <Segmented
              size="small"
              value={ui.sortBy}
              onChange={(v) => patchUI({ sortBy: v as 'score' | 'year' | 'citations' })}
              options={[
                { value: 'score', label: '相关度' },
                { value: 'year', label: '年份' },
                { value: 'citations', label: '引用数' },
              ]}
            />
          </Row>

          {papers.map((p, i) => (
            <motion.div
              key={`${p.paper_id || p.doc_id}-${i}`}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: Math.min(i * 0.04, 0.4) }}
            >
              <PaperCard p={p} rank={i + 1} query={ui.query} />
            </motion.div>
          ))}
        </>
      )}

      {!search.isPending && !search.data && (
        <div style={{ padding: '60px 0' }}>
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="开始检索" />
        </div>
      )}
      </>
      )}
    </Space>
  );
}
