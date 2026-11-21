/**
 * IndexCoveragePanel — "which conference is indexed through which year".
 *
 * Shown on the search page while the source filter is 论文, where the question
 * "is 2026 in there yet?" decides whether an empty result means "nothing
 * matched" or "never crawled". /api/papers/stats only carries global totals,
 * so this reads /api/papers/coverage.
 *
 * A venue with a gap (indexed 2022 and 2024 but not 2023) must not render as a
 * clean 2022–2024 span; the year strip marks absent years explicitly.
 */

import { useMemo, useState } from 'react';
import { Card, Col, Empty, Row, Skeleton, Space, Tag, Tooltip, Typography, theme } from 'antd';
import { DatabaseOutlined, WarningOutlined } from '@ant-design/icons';
import { AnimatePresence, motion } from 'framer-motion';
import { useIndexCoverage, type VenueCoverage } from '../api/hooks/useIndexCoverage';
import { venueColors } from './PaperFilterPanel';

const FALLBACK_COLOR = '#64748b';

export interface IndexCoveragePanelProps {
  /** Render nothing unless true — the panel is paper-only. */
  visible: boolean;
  /** Clicking a venue card hands the venue name back for filtering. */
  onSelectVenue?: (venue: string) => void;
  /** Venue names currently active in the parent's filter. */
  selectedVenues?: string[];
}

function YearStrip({ venue, color }: { venue: VenueCoverage; color: string }) {
  const { token } = theme.useToken();
  const max = Math.max(...venue.years.map((y) => y.count), 1);
  const missing = new Set(venue.missing_years);
  const span: number[] = [];
  for (let y = venue.min_year; y <= venue.max_year; y += 1) span.push(y);
  const byYear = new Map(venue.years.map((y) => [y.year, y.count]));

  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 34 }}>
      {span.map((year) => {
        const count = byYear.get(year) ?? 0;
        const absent = missing.has(year);
        return (
          <Tooltip
            key={year}
            title={absent ? `${year} 未索引` : `${year} · ${count.toLocaleString()} 篇`}
          >
            <motion.div
              initial={{ height: 2 }}
              animate={{ height: absent ? 4 : 6 + (count / max) * 24 }}
              transition={{ type: 'spring', stiffness: 260, damping: 26 }}
              style={{
                flex: 1,
                minWidth: 6,
                borderRadius: 2,
                background: absent ? token.colorErrorBorder : color,
                opacity: absent ? 1 : 0.35 + 0.65 * (count / max),
              }}
            />
          </Tooltip>
        );
      })}
    </div>
  );
}

function VenueCard({
  venue,
  index,
  selected,
  onSelect,
}: {
  venue: VenueCoverage;
  index: number;
  selected: boolean;
  onSelect?: (venue: string) => void;
}) {
  const { token } = theme.useToken();
  const color = venueColors[venue.venue] ?? FALLBACK_COLOR;
  const hasGap = venue.missing_years.length > 0;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      // Stagger by position so 18 cards cascade instead of popping at once.
      transition={{ delay: Math.min(index * 0.025, 0.4), duration: 0.28, ease: 'easeOut' }}
      whileHover={{ y: -3 }}
      style={{ height: '100%' }}
    >
      <Card
        size="small"
        hoverable={Boolean(onSelect)}
        onClick={onSelect ? () => onSelect(venue.venue) : undefined}
        style={{
          height: '100%',
          borderColor: selected ? color : undefined,
          borderWidth: selected ? 2 : 1,
          background: selected ? `${color}0d` : undefined,
        }}
        styles={{ body: { padding: '10px 12px' } }}
      >
        <Space orientation="vertical" size={6} style={{ width: '100%' }}>
          <Space size={6} style={{ justifyContent: 'space-between', width: '100%' }}>
            <Space size={6}>
              <span style={{
                width: 8, height: 8, borderRadius: 4, background: color, display: 'inline-block',
              }} />
              <Typography.Text strong style={{ fontSize: 13 }}>{venue.venue}</Typography.Text>
            </Space>
            {hasGap && (
              <Tooltip title={`缺年份：${venue.missing_years.join('、')}`}>
                <WarningOutlined style={{ color: token.colorWarning, fontSize: 12 }} />
              </Tooltip>
            )}
          </Space>

          <Space size={4} align="baseline">
            <Typography.Text style={{ fontFamily: 'monospace', fontSize: 15, fontWeight: 600 }}>
              {venue.min_year}
            </Typography.Text>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>→</Typography.Text>
            <Typography.Text style={{ fontFamily: 'monospace', fontSize: 15, fontWeight: 600, color }}>
              {venue.max_year}
            </Typography.Text>
          </Space>

          <YearStrip venue={venue} color={color} />
        </Space>
      </Card>
    </motion.div>
  );
}

export function IndexCoveragePanel({
  visible,
  onSelectVenue,
  selectedVenues = [],
}: IndexCoveragePanelProps) {
  const { token } = theme.useToken();
  const [expanded, setExpanded] = useState(false);
  const { data, isLoading, isError } = useIndexCoverage(visible);

  const selected = useMemo(() => new Set(selectedVenues), [selectedVenues]);
  // 18 venues is a lot of vertical space above the results; show the biggest
  // ones and let the rest open on demand.
  const COLLAPSED_COUNT = 8;
  const venues = data?.venues ?? [];
  const shown = expanded ? venues : venues.slice(0, COLLAPSED_COUNT);

  if (!visible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        transition={{ duration: 0.25, ease: 'easeOut' }}
        style={{ overflow: 'hidden' }}
      >
        <Card
          size="small"
          title={(
            <Space size={6}>
              <DatabaseOutlined style={{ color: token.colorPrimary }} />
              <Typography.Text strong style={{ fontSize: 13 }}>索引覆盖</Typography.Text>
            </Space>
          )}
          extra={venues.length > COLLAPSED_COUNT && (
            <Typography.Link style={{ fontSize: 12 }} onClick={() => setExpanded((v) => !v)}>
              {expanded ? '收起' : `展开全部 ${venues.length}`}
            </Typography.Link>
          )}
          styles={{ body: { padding: 12 } }}
        >
          {isLoading && <Skeleton active paragraph={{ rows: 2 }} />}
          {isError && (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={<Typography.Text type="secondary">覆盖信息读取失败</Typography.Text>}
            />
          )}
          {!isLoading && !isError && (
            <>
              <Row gutter={[10, 10]}>
                <AnimatePresence mode="popLayout">
                  {shown.map((venue, i) => (
                    <Col key={venue.venue} xs={12} sm={8} md={6} lg={4} xl={3}>
                      <VenueCard
                        venue={venue}
                        index={i}
                        selected={selected.has(venue.venue)}
                        onSelect={onSelectVenue}
                      />
                    </Col>
                  ))}
                </AnimatePresence>
              </Row>
              {Boolean(data?.unclassified) && (
                <Tag style={{ marginTop: 10 }} color="default">
                  另有 {data!.unclassified.toLocaleString()} 篇无会议/年份信息
                </Tag>
              )}
            </>
          )}
        </Card>
      </motion.div>
    </AnimatePresence>
  );
}

export default IndexCoveragePanel;
