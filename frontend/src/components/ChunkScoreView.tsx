import { Children, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import Markdown from 'react-markdown';
import type { Components } from 'react-markdown';
import { displayMath, rehypePlugins, remarkPlugins } from '../lib/markdown';
import { Typography, Tag, Tooltip, Segmented, theme } from 'antd';
import type { SearchResult } from '../types';

const { Text } = Typography;

const LEVEL_LABEL: Record<number, string> = { 1: '摘要', 2: '段落', 3: '句子' };

// Absolute score → categorical badge color (quick read of match strength).
function scoreBadgeColor(s: number): string {
  if (s >= 0.8) return 'green';
  if (s >= 0.5) return 'blue';
  if (s >= 0.3) return 'gold';
  return 'default';
}

// Relative heat: normalized score (within the displayed set) → light→deep indigo.
// Relative so the "which parts match best" signal stays vivid even when the
// absolute reranker scores are compressed into a narrow band.
function heatBg(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  const c0 = [238, 242, 255]; // #eef2ff faint
  const c1 = [79, 70, 229];   // #4F46E5 indigo
  const ch = c0.map((a, i) => Math.round(a + (c1[i] - a) * x));
  return `rgb(${ch[0]}, ${ch[1]}, ${ch[2]})`;
}

function buildHighlightRegex(query?: string): RegExp | null {
  if (!query) return null;
  const words = query.split(/\s+/).map((w) => w.trim()).filter((w) => w.length > 1);
  if (!words.length) return null;
  const esc = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  return new RegExp(`(${esc.join('|')})`, 'gi');
}

// Wrap query-term matches in <mark>. Operates only on string children so it
// never corrupts nested markdown nodes (code/strong/links pass through).
function highlightString(text: string, regex: RegExp): ReactNode[] {
  const parts = text.split(regex);
  const test = new RegExp(`^(?:${regex.source})$`, 'i');
  return parts.map((p, i) =>
    test.test(p)
      ? <mark key={i} style={{ background: 'rgba(79,70,229,0.22)', padding: '0 2px', borderRadius: 3 }}>{p}</mark>
      : <span key={i}>{p}</span>,
  );
}

// react-markdown component overrides: compact spacing, themed code/tables,
// query highlight on text nodes, and image refs (PDF page assets that 404)
// collapsed to a tag instead of a broken <img>.
function mdComponents(regex: RegExp | null, borderColor: string, codeBg: string): Components {
  const hl = (children: ReactNode): ReactNode =>
    !regex
      ? children
      : Children.map(children, (child) =>
          typeof child === 'string' ? highlightString(child, regex) : child,
        );
  const cell = { border: `1px solid ${borderColor}`, padding: '2px 6px' } as const;
  return {
    p: ({ children }) => <p style={{ margin: '4px 0' }}>{hl(children)}</p>,
    li: ({ children }) => <li style={{ margin: '2px 0' }}>{hl(children)}</li>,
    h1: ({ children }) => <Text strong style={{ fontSize: 14, display: 'block', margin: '6px 0 2px' }}>{hl(children)}</Text>,
    h2: ({ children }) => <Text strong style={{ fontSize: 13.5, display: 'block', margin: '6px 0 2px' }}>{hl(children)}</Text>,
    h3: ({ children }) => <Text strong style={{ fontSize: 13, display: 'block', margin: '4px 0 2px' }}>{hl(children)}</Text>,
    h4: ({ children }) => <Text strong style={{ fontSize: 12.5, display: 'block', margin: '4px 0 2px' }}>{hl(children)}</Text>,
    blockquote: ({ children }) => (
      <blockquote style={{ borderLeft: `3px solid ${borderColor}`, margin: '4px 0', paddingLeft: 8, opacity: 0.85 }}>{children}</blockquote>
    ),
    code: ({ children, className }) => (
      <code className={className} style={{ background: codeBg, padding: '0 4px', borderRadius: 4, fontSize: 12 }}>{children}</code>
    ),
    table: ({ children }) => (
      <div style={{ overflowX: 'auto' }}><table style={{ borderCollapse: 'collapse', fontSize: 12, margin: '4px 0' }}>{children}</table></div>
    ),
    td: ({ children }) => <td style={cell}>{hl(children)}</td>,
    th: ({ children }) => <th style={cell}>{hl(children)}</th>,
    a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
    img: ({ alt }) => <Tag style={{ fontSize: 11 }}>🖼 {alt || '图'}</Tag>,
  };
}

function chunkDomId(c: SearchResult): string {
  return `chunk-${String(c.chunk_id)}`;
}

function chunkPos(c: SearchResult): number {
  const m = (c.metadata || {}) as Record<string, unknown>;
  // chunk_index (SQL document order, attached by the backend for in-paper
  // search) is preferred; start_idx is a fallback. Unknown position sorts last.
  if (typeof m.chunk_index === 'number') return m.chunk_index;
  if (typeof m.start_idx === 'number') return m.start_idx;
  return Number.MAX_SAFE_INTEGER;
}

function scrollToChunk(c: SearchResult) {
  const el = document.getElementById(chunkDomId(c));
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ─── document-order heatmap strip ───
function OverviewStrip({ chunks, norm }: { chunks: SearchResult[]; norm: (s: number) => number }) {
  const { token } = theme.useToken();
  const ordered = useMemo(
    () => [...chunks].sort((a, b) => chunkPos(a) - chunkPos(b)),
    [chunks],
  );
  return (
    <div style={{ marginBottom: 12 }}>
      <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>
        论文匹配热力图 · 按正文顺序排列，颜色越深与查询越相关（点击跳转到对应片段）
      </Text>
      <div style={{ display: 'flex', gap: 2, height: 30, alignItems: 'stretch' }}>
        {ordered.map((c, idx) => (
          <Tooltip key={chunkDomId(c)} title={`正文第 ${idx + 1}/${ordered.length} 段 · 分数 ${c.score.toFixed(4)}`}>
            <div
              onClick={() => scrollToChunk(c)}
              style={{
                flex: 1, minWidth: 3, borderRadius: 2, cursor: 'pointer',
                background: heatBg(norm(c.score)),
                border: `1px solid ${token.colorBorderSecondary}`,
              }}
            />
          </Tooltip>
        ))}
      </div>
    </div>
  );
}

// ─── one chunk: score border + badge + meta + markdown body ───
function ChunkRow({
  chunk, rank, regex, norm,
}: {
  chunk: SearchResult; rank: number;
  regex: RegExp | null; norm: (s: number) => number;
}) {
  const { token } = theme.useToken();
  const [expanded, setExpanded] = useState(false);
  const meta = (chunk.metadata || {}) as Record<string, unknown>;
  const level = meta.level as number | undefined;
  const sectionName = meta.section_name as string | undefined;
  const rerankScore = meta.reranker_score as number | undefined;
  const initialScore = meta.initial_score as number | undefined;
  const content = chunk.content || '';
  const longContent = content.length > 320;

  const components = useMemo(
    () => mdComponents(regex, token.colorBorderSecondary, token.colorFillTertiary),
    [regex, token.colorBorderSecondary, token.colorFillTertiary],
  );

  return (
    <div
      id={chunkDomId(chunk)}
      style={{
        borderLeft: `4px solid ${heatBg(norm(chunk.score))}`,
        background: token.colorFillQuaternary,
        borderRadius: 8,
        padding: '8px 12px',
        marginBottom: 8,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
        <Text type="secondary" style={{ fontSize: 11, fontFamily: 'monospace' }}>#{rank}</Text>
        <Tooltip title={
          <span>
            最终分: {chunk.score.toFixed(4)}
            {rerankScore != null && <><br />rerank: {rerankScore.toFixed(4)}</>}
            {initialScore != null && <><br />初召回: {initialScore.toFixed(4)}</>}
          </span>
        }>
          <Tag color={scoreBadgeColor(chunk.score)} style={{ margin: 0, fontFamily: 'monospace', fontWeight: 600 }}>
            {chunk.score.toFixed(4)}
          </Tag>
        </Tooltip>
        {level != null && (
          <Tag style={{ margin: 0, fontSize: 11 }}>L{level} · {LEVEL_LABEL[level] ?? '未知'}</Tag>
        )}
        {sectionName && !['title_summary', 'body'].includes(sectionName) && (
          <Tag style={{ margin: 0, fontSize: 11, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            § {sectionName}
          </Tag>
        )}
      </div>
      <div
        style={{
          fontSize: 12.5, lineHeight: 1.6, color: token.colorText,
          maxHeight: expanded || !longContent ? undefined : 150,
          overflow: expanded || !longContent ? undefined : 'hidden',
          position: 'relative',
        }}
      >
        <Markdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>{displayMath(content)}</Markdown>
        {!expanded && longContent && (
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0, height: 40,
            background: `linear-gradient(transparent, ${token.colorFillQuaternary})`,
          }} />
        )}
      </div>
      {longContent && (
        <Typography.Link style={{ fontSize: 11 }} onClick={() => setExpanded((v) => !v)}>
          {expanded ? '收起' : '展开全文'}
        </Typography.Link>
      )}
    </div>
  );
}

export interface ChunkScoreViewProps {
  chunks: SearchResult[];
  query?: string;
  /** Render the document-order heatmap strip above the chunk list. */
  showOverview?: boolean;
}

/**
 * Renders a paper's matched chunks as markdown, each annotated with a
 * relative score-heat (border + badge) so the match degree of different parts
 * of the paper is visually obvious. Reused by search result cards (a few
 * chunks) and the paper-detail in-paper search (all chunks + overview strip).
 */
export default function ChunkScoreView({ chunks, query, showOverview }: ChunkScoreViewProps) {
  const regex = useMemo(() => buildHighlightRegex(query), [query]);
  // Two views of the same matched chunks: by relevance (most-relevant first) or
  // by document position (read the paper top-to-bottom with scores annotated).
  // The heatmap strip above always stays in document order — it's the map.
  const [sortMode, setSortMode] = useState<'score' | 'position'>('score');
  const { lo, hi } = useMemo(() => {
    if (!chunks.length) return { lo: 0, hi: 1 };
    const s = chunks.map((c) => c.score);
    return { lo: Math.min(...s), hi: Math.max(...s) };
  }, [chunks]);
  const norm = useMemo(() => (s: number) => (hi > lo ? (s - lo) / (hi - lo) : 1), [lo, hi]);
  const displayChunks = useMemo(() => {
    const arr = [...chunks];
    arr.sort(sortMode === 'position'
      ? (a, b) => chunkPos(a) - chunkPos(b)
      : (a, b) => b.score - a.score);
    return arr;
  }, [chunks, sortMode]);

  if (!chunks.length) return null;

  return (
    <div>
      {showOverview && chunks.length > 1 && <OverviewStrip chunks={chunks} norm={norm} />}
      {chunks.length > 1 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
          <Segmented
            size="small"
            value={sortMode}
            onChange={(v) => setSortMode(v as 'score' | 'position')}
            options={[
              { value: 'score', label: '按相关度' },
              { value: 'position', label: '按正文顺序' },
            ]}
          />
        </div>
      )}
      {displayChunks.map((c, i) => (
        <ChunkRow key={chunkDomId(c)} chunk={c} rank={i + 1} regex={regex} norm={norm} />
      ))}
    </div>
  );
}

// Helper for callers that need the same heat scale (e.g. overview legends).
export { heatBg, scoreBadgeColor };
