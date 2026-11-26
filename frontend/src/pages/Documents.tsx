import { useMemo, useState } from 'react';
import {
  Table,
  Button,
  Space,
  Typography,
  Tag,
  Badge,
  Drawer,
  Descriptions,
  List,
  Row,
  Col,
  Segmented,
  Input,
  theme,
} from 'antd';
import type { BadgeProps } from 'antd';
import {
  ReloadOutlined,
  SearchOutlined,
  FilterOutlined,
  ClearOutlined,
  CheckCircleFilled,
  WarningFilled,
} from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import {
  useDocuments,
  useDocumentDetail,
  useDocumentIndexStatus,
} from '../api/hooks/useDocuments';
import { usePaperBrowse } from '../api/hooks/usePapers';
import type { PaperBrief } from '../types';
import PaperFilterPanel, { venueColors, type PaperFilterValue } from '../components/PaperFilterPanel';
import { statusColor, statusLabel } from '../constants';
import { parseCreatedAt, relativeTimeFrom } from './taskTime';

// 文档库 is a pure browse surface — identical on public and internal builds.
// Upload / edit / delete / URL-crawl-submit moved to Settings → 数据管理 (F1-e),
// behind ConsoleGate; this page only ever reads.
function DocumentIndexBadge({ docId }: { docId: number }) {
  const { data, isLoading } = useDocumentIndexStatus(docId);
  if (isLoading) return <Tag>索引状态加载中…</Tag>;
  if (!data) return null;
  const { sql_chunks_count: sql, qdrant_points_count: qdr, indexed } = data;
  if (indexed && qdr >= sql) {
    return <Tag color="success" icon={<CheckCircleFilled />}>已索引 {qdr}/{sql}</Tag>;
  }
  if (qdr > 0) {
    return (
      <Tag color="warning" icon={<WarningFilled />}>
        部分索引 {qdr}/{sql}
      </Tag>
    );
  }
  return (
    <Tag color="error" icon={<WarningFilled />}>
      未索引（SQL chunks {sql}，Qdrant 0）
    </Tag>
  );
}

type SourceFilter = 'all' | 'paper' | 'webpage' | 'upload';

export default function Documents() {
  const { token } = theme.useToken();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all');

  // Paper-mode advanced filter state (only used when sourceFilter === 'paper').
  // Distinct from `useDocuments` so the filter UI feels like Search, but the
  // results come from `/papers/browse` which returns the full SQL match set.
  const [paperFilter, setPaperFilter] = useState<PaperFilterValue>({ venues: [], years: [] });
  const [paperKw, setPaperKw] = useState('');
  const [paperKwCommitted, setPaperKwCommitted] = useState('');
  const [showFilters, setShowFilters] = useState(true);

  const isPaperMode = sourceFilter === 'paper';

  const { data, isLoading, isFetching } = useDocuments(
    page,
    pageSize,
    !isPaperMode && sourceFilter !== 'all' ? sourceFilter : undefined,
    { enabled: !isPaperMode },
  );

  const paperBrowse = usePaperBrowse(
    isPaperMode
      ? {
          venue: paperFilter.venues.length ? paperFilter.venues : undefined,
          year: paperFilter.years.length ? paperFilter.years : undefined,
          q: paperKwCommitted.trim() || undefined,
          page,
          page_size: pageSize,
        }
      : { page: 1, page_size: 1 },
  );

  const paperCriteriaActive = useMemo(
    () => paperFilter.venues.length + paperFilter.years.length + (paperKwCommitted.trim() ? 1 : 0),
    [paperFilter, paperKwCommitted],
  );

  const qc = useQueryClient();
  const [detailId, setDetailId] = useState<number>();
  const { data: detail } = useDocumentDetail(detailId);

  const sourceLabel: Record<string, { text: string; color: string }> = {
    paper: { text: '论文', color: 'blue' },
    webpage: { text: '网页', color: 'green' },
    upload: { text: '上传', color: 'orange' },
  };

  const columns = [
    {
      title: '标题',
      dataIndex: 'title',
      ellipsis: true,
      render: (t: string, r: { id: number; source_type: string; metadata?: Record<string, unknown> }) => {
        const paperId = r.metadata?.paper_id as string | undefined;
        if (r.source_type === 'paper' && paperId) {
          return <Link to={`/papers/${paperId}`}>{t || '-'}</Link>;
        }
        return <a onClick={() => setDetailId(r.id)}>{t || '-'}</a>;
      },
    },
    {
      title: '来源',
      dataIndex: 'source_type',
      width: 80,
      render: (v: string) => {
        const m = sourceLabel[v] ?? { text: v, color: 'default' };
        return <Tag color={m.color}>{m.text}</Tag>;
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 90,
      render: (s: string) => (
        <Badge status={(statusColor[s] as BadgeProps['status']) ?? 'default'} text={statusLabel[s] ?? s} />
      ),
    },
    { title: '片段', dataIndex: 'chunk_count', width: 70 },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 110,
      render: (v: string) => relativeTimeFrom(parseCreatedAt(v)),
    },
  ];

  // Paper-mode columns — drive the "筛选出所有匹配到的论文" table off /papers/browse.
  const paperColumns = [
    {
      title: '标题',
      dataIndex: 'title',
      ellipsis: true,
      render: (t: string, r: PaperBrief) => <Link to={`/papers/${r.paper_id}`}>{t || '-'}</Link>,
    },
    {
      title: '会议',
      dataIndex: 'venue',
      width: 130,
      render: (v: string) => <Tag color={venueColors[v] || 'blue'}>{v}</Tag>,
    },
    { title: '年份', dataIndex: 'year', width: 70 },
    {
      title: '作者',
      dataIndex: 'authors',
      ellipsis: true,
      width: 240,
      render: (a: string[]) => a?.join(', ') ?? '-',
    },
    {
      title: '级别',
      dataIndex: 'venue_tier',
      width: 80,
      render: (t: string) => (t ? <Tag color="gold">{t}</Tag> : '-'),
    },
  ];

  // Commit keyword → triggers a fresh /papers/browse fetch from page 1.
  const commitPaperKw = () => {
    setPaperKwCommitted(paperKw.trim());
    setPage(1);
  };

  const paperTotal = paperBrowse.data?.total ?? 0;

  return (
    <Space orientation="vertical" size="middle" style={{ width: '100%' }}>
      <Row justify="space-between" align="middle">
        <Segmented
          size="middle"
          value={sourceFilter}
          onChange={(v) => {
            setSourceFilter(v as SourceFilter);
            setPage(1);
          }}
          options={[
            { value: 'all', label: '全部' },
            { value: 'paper', label: '论文' },
            { value: 'webpage', label: '网页' },
            { value: 'upload', label: '上传' },
          ]}
        />
        <Button
          icon={<ReloadOutlined spin={isPaperMode ? paperBrowse.isFetching : isFetching} />}
          onClick={() => qc.invalidateQueries({ queryKey: [isPaperMode ? 'papers' : 'documents'] })}
        >刷新</Button>
      </Row>

      {isPaperMode && (
        <Space orientation="vertical" size="small" style={{ width: '100%' }}>
          <Row gutter={8} align="middle" wrap={false}>
            <Col flex="auto">
              <Input.Search
                allowClear
                size="middle"
                placeholder="关键词匹配标题 / 摘要"
                prefix={<SearchOutlined style={{ color: token.colorTextTertiary }} />}
                value={paperKw}
                onChange={(e) => setPaperKw(e.target.value)}
                onSearch={commitPaperKw}
                enterButton="检索"
              />
            </Col>
            <Col>
              <Button
                icon={<FilterOutlined />}
                onClick={() => setShowFilters((s) => !s)}
                type={showFilters ? 'primary' : 'default'}
                ghost={showFilters}
              >
                筛选{paperCriteriaActive > 0 ? ` · ${paperCriteriaActive}` : ''}
              </Button>
            </Col>
            {paperCriteriaActive > 0 && (
              <Col>
                <Button
                  icon={<ClearOutlined />}
                  onClick={() => {
                    setPaperFilter({ venues: [], years: [] });
                    setPaperKw('');
                    setPaperKwCommitted('');
                    setPage(1);
                  }}
                >
                  清空
                </Button>
              </Col>
            )}
          </Row>
          <PaperFilterPanel
            value={paperFilter}
            onChange={(next) => {
              setPaperFilter(next);
              setPage(1);
            }}
            expanded={showFilters}
            title="论文筛选"
          />
        </Space>
      )}

      {isPaperMode ? (
        <Table
          dataSource={paperBrowse.data?.results ?? []}
          columns={paperColumns}
          rowKey={(r) => r.paper_id || String(r.doc_id)}
          loading={paperBrowse.isLoading || paperBrowse.isFetching}
          size="small"
          pagination={{
            current: page,
            pageSize,
            total: paperTotal,
            showSizeChanger: true,
            pageSizeOptions: [20, 50, 100, 200],
            showTotal: (t) => `命中 ${t.toLocaleString()} 篇`,
            onChange: (p, ps) => {
              setPage(p);
              setPageSize(ps);
            },
          }}
        />
      ) : (
        <Table
          dataSource={data?.documents ?? []}
          columns={columns}
          rowKey="id"
          loading={isLoading}
          size="small"
          pagination={{
            current: page,
            pageSize,
            total: data?.total ?? 0,
            showSizeChanger: true,
            showTotal: (t) => `共 ${t}`,
            onChange: (p, ps) => {
              setPage(p);
              setPageSize(ps);
            },
          }}
        />
      )}

      <Drawer
        title="文档详情"
        open={!!detailId}
        onClose={() => setDetailId(undefined)}
        size="large"
      >
        {detail && (
          <Space orientation="vertical" style={{ width: '100%' }}>
            <DocumentIndexBadge docId={detail.id} />
            <Descriptions bordered size="small" column={1}>
              <Descriptions.Item label="ID">{detail.id}</Descriptions.Item>
              <Descriptions.Item label="标题">{detail.title}</Descriptions.Item>
              <Descriptions.Item label="来源">{detail.source}</Descriptions.Item>
              <Descriptions.Item label="类型">{detail.source_type}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Badge status={(statusColor[detail.status] as BadgeProps['status']) ?? 'default'} text={statusLabel[detail.status] ?? detail.status} />
              </Descriptions.Item>
              <Descriptions.Item label="片段数">{detail.chunk_count}</Descriptions.Item>
              <Descriptions.Item label="创建时间">{relativeTimeFrom(parseCreatedAt(detail.created_at))}</Descriptions.Item>
            </Descriptions>
            {detail.chunks && detail.chunks.length > 0 && (
              <List
                header={`${detail.chunks.length} 个片段`}
                size="small"
                dataSource={detail.chunks}
                renderItem={(c) => (
                  <List.Item>
                    <List.Item.Meta
                      title={<><Tag>L{c.level}</Tag> #{c.chunk_index} ({c.token_count} tokens)</>}
                      description={
                        <Typography.Paragraph ellipsis={{ rows: 2 }} style={{ margin: 0, fontSize: 12 }}>
                          {c.content_preview}
                        </Typography.Paragraph>
                      }
                    />
                  </List.Item>
                )}
              />
            )}
          </Space>
        )}
      </Drawer>
    </Space>
  );
}
