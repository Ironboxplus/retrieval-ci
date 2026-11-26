/**
 * PaperFilterPanel — shared advanced-filter UI for paper selection.
 *
 * Lifted from Search.tsx so both the semantic search page and the documents
 * page (paper-mode) can reuse the same venue-group / year multi-select with
 * count badges, grey-out-empty entries, and clear/select-all controls.
 *
 * Stays controlled — parent owns `venues` and `years` state.
 */

import { useCallback, useMemo } from 'react';
import { Card, Tag, Typography, Row, Col, Space, Button, Tooltip, theme } from 'antd';
import { FilterOutlined } from '@ant-design/icons';
import { motion, AnimatePresence } from 'framer-motion';
import { useConferences, buildVenueGroups, type VenueGroup } from '../api/hooks/useConferences';
import { useDistribution } from '../api/hooks/useDashboard';

export const venueColors: Record<string, string> = {
  AAAI: '#4F46E5', ACL: '#059669', CVPR: '#db2777', ICCV: '#7c3aed',
  ICLR: '#ea580c', ICML: '#0891b2', IJCAI: '#2563eb', NeurIPS: '#dc2626',
  EMNLP: '#10b981', KDD: '#f97316', SIGIR: '#06b6d4', SIGMOD: '#8b5cf6',
  VLDB: '#a855f7', CCS: '#ef4444', 'ACM MM': '#f43f5e',
  'USENIX Security': '#0ea5e9', WWW: '#4338ca', CSUR: '#64748b',
};

const CATEGORY_LABEL_CN: Record<string, string> = {
  AI: 'AI', DB: '数据/检索', NW: '网络', SC: '网安', SE: '系统软件',
  DS: '体系结构', CT: '理论', CG: '多媒体', HI: '人机交互', MX: '交叉',
};

const FALLBACK_VENUE_GROUPS: VenueGroup[] = [
  { label: 'AI', sub: 'AI', venues: [
    { title: 'NeurIPS', ccf: 'A' }, { title: 'ICML', ccf: 'A' }, { title: 'ICLR', ccf: 'A' },
    { title: 'AAAI', ccf: 'A' }, { title: 'ACL', ccf: 'A' }, { title: 'CVPR', ccf: 'A' },
    { title: 'ICCV', ccf: 'A' }, { title: 'IJCAI', ccf: 'B' }, { title: 'EMNLP', ccf: 'B' },
  ] },
  { label: 'DB', sub: 'DB', venues: [
    { title: 'KDD', ccf: 'A' }, { title: 'SIGIR', ccf: 'A' },
    { title: 'SIGMOD', ccf: 'A' }, { title: 'VLDB', ccf: 'A' },
  ] },
  { label: 'SC', sub: 'SC', venues: [
    { title: 'CCS', ccf: 'A' }, { title: 'USENIX Security', ccf: 'A' },
  ] },
  { label: 'CG', sub: 'CG', venues: [{ title: 'ACM MM', ccf: 'A' }] },
];
const FALLBACK_YEARS = [2021, 2022, 2023, 2024, 2025, 2026];

export interface PaperFilterValue {
  venues: string[];
  years: number[];
}

export interface PaperFilterPanelProps {
  value: PaperFilterValue;
  onChange: (next: PaperFilterValue) => void;
  /** Optional collapsible wrapper — default `true` shows panel inline. */
  expanded?: boolean;
  /** Title shown in the panel header. Default 高级筛选. */
  title?: string;
  /** Show only the 年份 row (no venue rows, no venue counter). Default `false`. */
  yearsOnly?: boolean;
}

export function PaperFilterPanel({ value, onChange, expanded = true, title = '高级筛选', yearsOnly = false }: PaperFilterPanelProps) {
  const { token } = theme.useToken();
  const { data: confData } = useConferences();
  const { data: distData } = useDistribution();
  const venueGroups = useMemo(
    () => (confData ? buildVenueGroups(confData) : FALLBACK_VENUE_GROUPS),
    [confData],
  );
  const allYears = useMemo(() => {
    if (!confData) return FALLBACK_YEARS;
    const yrs = new Set<number>();
    for (const c of confData.conferences) for (const y of c.years) yrs.add(y);
    return [...yrs].sort();
  }, [confData]);

  const venueCounts = useMemo<Record<string, number> | undefined>(() => {
    if (!distData) return undefined;
    const m: Record<string, number> = {};
    for (const v of distData.venues) m[v.venue] = v.papers;
    return m;
  }, [distData]);
  const yearCounts = useMemo<Record<number, number> | undefined>(() => {
    if (!distData) return undefined;
    const m: Record<number, number> = {};
    for (const y of distData.years) m[y.year] = y.papers;
    return m;
  }, [distData]);

  const venueCount = useCallback((v: string) => (venueCounts ? (venueCounts[v] ?? 0) : undefined), [venueCounts]);
  const yearCount = useCallback((y: number) => (yearCounts ? (yearCounts[y] ?? 0) : undefined), [yearCounts]);

  const patch = useCallback(
    (delta: Partial<PaperFilterValue>) => onChange({ ...value, ...delta }),
    [onChange, value],
  );

  return (
    <AnimatePresence initial={false}>
      {expanded && (
        <motion.div
          key="paper-filter-panel"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
          style={{ overflow: 'hidden' }}
        >
          <Card
            size="small"
            styles={{ body: { padding: '12px 14px', background: token.colorFillQuaternary } }}
            style={{ borderRadius: 10 }}
          >
            <Row justify="space-between" align="middle" style={{ marginBottom: 10 }}>
              <Space size={6}>
                <FilterOutlined style={{ color: token.colorPrimary, fontSize: 13 }} />
                <Typography.Text strong style={{ fontSize: 12, color: token.colorText }}>{title}</Typography.Text>
                {!yearsOnly && value.venues.length > 0 && (
                  <Tag style={{ borderRadius: 10, background: token.colorPrimaryBg, color: token.colorPrimary, border: 'none', fontSize: 11, margin: 0 }}>
                    {value.venues.length} 会议
                  </Tag>
                )}
                {value.years.length > 0 && (
                  <Tag style={{ borderRadius: 10, background: token.colorSuccessBg, color: token.colorSuccess, border: 'none', fontSize: 11, margin: 0 }}>
                    {value.years.length} 年份
                  </Tag>
                )}
              </Space>
              {((yearsOnly ? 0 : value.venues.length) + value.years.length) > 0 && (
                <Button
                  type="link"
                  size="small"
                  style={{ fontSize: 11, padding: 0, height: 20 }}
                  onClick={() => patch(yearsOnly ? { years: [] } : { venues: [], years: [] })}
                >清空全部</Button>
              )}
            </Row>

            {/* Venues */}
            {!yearsOnly && (
            <div style={{ marginBottom: 12 }}>
              <Space orientation="vertical" size={6} style={{ width: '100%' }}>
                {venueGroups.map((g) => {
                  const selectable = g.venues
                    .filter((v) => {
                      const n = venueCount(v.title);
                      return n === undefined || n > 0;
                    })
                    .map((v) => v.title);
                  const allIn = selectable.length > 0 && selectable.every((v) => value.venues.includes(v));
                  const someIn = selectable.some((v) => value.venues.includes(v));
                  const groupAllEmpty = selectable.length === 0;
                  const cnLabel = CATEGORY_LABEL_CN[g.sub] ?? g.label;
                  return (
                    <Row key={g.sub} gutter={8} wrap={true} align="middle">
                      <Col flex="0 0 84px">
                          <Tag.CheckableTag
                            checked={allIn}
                            onChange={() => {
                              if (groupAllEmpty) return;
                              patch({
                                venues: allIn
                                  ? value.venues.filter((v) => !selectable.includes(v))
                                  : [...new Set([...value.venues, ...selectable])],
                              });
                            }}
                            style={{
                              borderRadius: 6,
                              fontSize: 11,
                              fontWeight: 600,
                              margin: 0,
                              padding: '2px 8px',
                              border: someIn && !allIn ? `1px dashed ${token.colorPrimary}` : undefined,
                              background: groupAllEmpty
                                ? token.colorFillQuaternary
                                : (allIn ? token.colorPrimary : (someIn ? token.colorPrimaryBg : token.colorFillSecondary)),
                              color: groupAllEmpty
                                ? token.colorTextDisabled
                                : (allIn ? '#fff' : (someIn ? token.colorPrimary : token.colorTextSecondary)),
                              display: 'inline-block',
                              textAlign: 'center',
                              minWidth: 76,
                              cursor: groupAllEmpty ? 'not-allowed' : 'pointer',
                              opacity: groupAllEmpty ? 0.55 : 1,
                            }}
                          >
                            {cnLabel}
                          </Tag.CheckableTag>
                      </Col>
                      <Col flex="auto" style={{ minWidth: 0 }}>
                        <Space size={[6, 4]} style={{ flexWrap: 'wrap' }}>
                          {g.venues.map((v) => {
                            const checked = value.venues.includes(v.title);
                            const dot = venueColors[v.title];
                            const cnt = venueCount(v.title);
                            const empty = cnt === 0;
                            const labelNode = (
                              <Tooltip title={
                                cnt === undefined ? v.title
                                  : empty ? `${v.title}：暂无已索引论文`
                                  : `${v.title}：${cnt.toLocaleString()} 篇已索引`
                              }>
                                <span>
                                  {v.title}
                                  {v.ccf !== 'A' && (
                                    <sup style={{ fontSize: 9, marginLeft: 2, opacity: 0.7 }}>{v.ccf}</sup>
                                  )}
                                </span>
                              </Tooltip>
                            );
                            if (empty) {
                              return (
                                <Tag
                                  key={v.title}
                                  style={{
                                    borderRadius: 4,
                                    fontSize: 11,
                                    margin: 0,
                                    padding: '2px 8px',
                                    borderLeft: dot ? `3px solid ${token.colorBorderSecondary}` : undefined,
                                    background: token.colorFillQuaternary,
                                    color: token.colorTextDisabled,
                                    cursor: 'not-allowed',
                                    textDecoration: 'line-through',
                                    textDecorationColor: token.colorTextDisabled,
                                  }}
                                >{labelNode}</Tag>
                              );
                            }
                            return (
                              <Tag.CheckableTag
                                key={v.title}
                                checked={checked}
                                onChange={(c) => patch({
                                  venues: c
                                    ? [...value.venues, v.title]
                                    : value.venues.filter((x) => x !== v.title),
                                })}
                                style={{
                                  borderRadius: 4,
                                  fontSize: 11,
                                  margin: 0,
                                  padding: '2px 8px',
                                  borderLeft: dot ? `3px solid ${checked ? '#fff' : dot}` : undefined,
                                }}
                              >{labelNode}</Tag.CheckableTag>
                            );
                          })}
                        </Space>
                      </Col>
                    </Row>
                  );
                })}
              </Space>
            </div>
            )}

            {/* Years — same leading row-header-chip + tag-row structure as
                each venue category row above, instead of a floating
                "全选/清空" pair with nothing anchoring the row. */}
            <div>
              <Row gutter={8} wrap align="middle">
                <Col flex="0 0 84px">
                  {(() => {
                    const yearsSelectable = allYears.filter((y) => {
                      const n = yearCount(y);
                      return n === undefined || n > 0;
                    });
                    const allIn = yearsSelectable.length > 0 && yearsSelectable.every((y) => value.years.includes(y));
                    const someIn = yearsSelectable.some((y) => value.years.includes(y));
                    return (
                      <Tag.CheckableTag
                        checked={allIn}
                        onChange={() => patch({
                          years: allIn ? value.years.filter((y) => !yearsSelectable.includes(y)) : [...new Set([...value.years, ...yearsSelectable])],
                        })}
                        style={{
                          borderRadius: 6,
                          fontSize: 11,
                          fontWeight: 600,
                          margin: 0,
                          padding: '2px 8px',
                          border: someIn && !allIn ? `1px dashed ${token.colorPrimary}` : undefined,
                          background: allIn ? token.colorPrimary : (someIn ? token.colorPrimaryBg : token.colorFillSecondary),
                          color: allIn ? '#fff' : (someIn ? token.colorPrimary : token.colorTextSecondary),
                          display: 'inline-block',
                          textAlign: 'center',
                          minWidth: 76,
                          cursor: 'pointer',
                        }}
                      >年份</Tag.CheckableTag>
                    );
                  })()}
                </Col>
                <Col flex="auto" style={{ minWidth: 0 }}>
              <Space size={[6, 4]} style={{ flexWrap: 'wrap' }}>
                {allYears.map((y) => {
                  const checked = value.years.includes(y);
                  const cnt = yearCount(y);
                  const empty = cnt === 0;
                  const labelNode = (
                    <Tooltip title={
                      cnt === undefined ? String(y)
                        : empty ? `${y}：暂无已索引论文`
                        : `${y}：${cnt.toLocaleString()} 篇`
                    }>
                      <span>{y}</span>
                    </Tooltip>
                  );
                  if (empty) {
                    return (
                      <Tag key={y} style={{
                        borderRadius: 4, fontSize: 11, margin: 0,
                        padding: '2px 10px', fontFamily: 'monospace',
                        background: token.colorFillQuaternary,
                        color: token.colorTextDisabled,
                        cursor: 'not-allowed',
                        textDecoration: 'line-through',
                        textDecorationColor: token.colorTextDisabled,
                      }}>{labelNode}</Tag>
                    );
                  }
                  return (
                    <Tag.CheckableTag
                      key={y}
                      checked={checked}
                      onChange={(c) => patch({
                        years: c
                          ? [...value.years, y].sort((a, b) => a - b)
                          : value.years.filter((x) => x !== y),
                      })}
                      style={{ borderRadius: 4, fontSize: 11, margin: 0, padding: '2px 10px', fontFamily: 'monospace' }}
                    >{labelNode}</Tag.CheckableTag>
                  );
                })}
              </Space>
                </Col>
                <Col flex="0 0 auto">
                  <Space size={4}>
                    <Button
                      type="link"
                      size="small"
                      style={{ fontSize: 11, padding: 0, height: 18 }}
                      onClick={() => patch({
                        years: allYears.filter((y) => {
                          const n = yearCount(y);
                          return n === undefined || n > 0;
                        }),
                      })}
                    >全选</Button>
                    <span style={{ color: token.colorBorderSecondary }}>·</span>
                    <Button
                      type="link"
                      size="small"
                      style={{ fontSize: 11, padding: 0, height: 18 }}
                      disabled={value.years.length === 0}
                      onClick={() => patch({ years: [] })}
                    >清空</Button>
                  </Space>
                </Col>
              </Row>
              {!yearsOnly && (
              <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1 border-t pt-2 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <span className="rounded bg-muted px-1.5 text-muted-foreground/60 line-through">CCS</span>
                  未入库，不可选
                </span>
                <span className="flex items-center gap-1.5">
                  <span>IJCAI<sup className="ml-0.5 text-[9px]">B</sup></span>
                  CCF 等级（未标注为 A）
                </span>
                <span>左侧色条 = 该会议在结果分布图中的颜色</span>
              </div>
              )}
            </div>
          </Card>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default PaperFilterPanel;
