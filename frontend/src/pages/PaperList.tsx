import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Table,
  Select,
  Space,
  Tag,
  Typography,
  Card,
  Row,
  Col,
  Input,
  Button,
  Result,
  Switch,
} from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { useQueryClient } from '@tanstack/react-query';
import { usePaperBrowse, usePaperTags } from '../api/hooks/usePapers';
import { useAuthUI } from '../stores/authStore';
import type { PaperBrowseParams, PaperBrief } from '../types';
import { withdrawnMarks } from './withdrawnModel';

export default function PaperList() {
  const [filters, setFilters] = useState<PaperBrowseParams>({
    page: 1,
    page_size: 20,
  });

  const { data, isLoading, isFetching, isError, error } = usePaperBrowse(filters);
  const { data: tags } = usePaperTags();
  const qc = useQueryClient();
  const openLogin = useAuthUI((s) => s.openLogin);

  // 论文库走 /api/papers/browse，已纳入登录闸（W127）。匿名 → 401 → 提示登录而非空表。
  const status = (error as { response?: { status?: number } } | null)?.response?.status;
  if (isError && status === 401) {
    return (
      <Result
        status="403"
        title="请先登录"
        subTitle="论文库需登录后浏览，新用户登录即注册（浏览不扣费）。"
        extra={<Button type="primary" onClick={openLogin}>登录 / 注册</Button>}
      />
    );
  }

  const updateFilter = (patch: Partial<PaperBrowseParams>) =>
    setFilters((prev) => ({ ...prev, ...patch, page: 1 }));

  const columns = [
    {
      title: '标题',
      dataIndex: 'title',
      ellipsis: true,
      render: (t: string, r: PaperBrief) => {
        const marks = withdrawnMarks(r);
        return (
          <Space size={4}>
            <Link to={`/papers/${r.paper_id}`}>{t}</Link>
            {marks.withdrawn && <Tag color="red">已撤稿</Tag>}
            {marks.versionLabel && <Tag>{marks.versionLabel}</Tag>}
          </Space>
        );
      },
    },
    {
      title: '会议',
      dataIndex: 'venue',
      width: 100,
      render: (v: string) => <Tag color="blue">{v}</Tag>,
    },
    { title: '年份', dataIndex: 'year', width: 70 },
    {
      title: '作者',
      dataIndex: 'authors',
      ellipsis: true,
      width: 260,
      render: (a: string[]) => a?.join(', ') ?? '-',
    },
    {
      title: '级别',
      dataIndex: 'venue_tier',
      width: 80,
      render: (t: string) => t ? <Tag color="gold">{t}</Tag> : '-',
    },
  ];

  return (
    <Space orientation="vertical" size="middle" style={{ width: '100%' }}>
      <Row justify="space-between" align="middle">
        <Typography.Title level={3} style={{ margin: 0 }}>论文库</Typography.Title>
        <Button icon={<ReloadOutlined spin={isFetching} />} onClick={() => qc.invalidateQueries({ queryKey: ['papers'] })}>刷新</Button>
      </Row>

      <Card size="small">
        <Row gutter={[12, 12]} align="middle">
          <Col>
            <Select
              placeholder="会议"
              allowClear
              style={{ width: 160 }}
              value={filters.venue}
              onChange={(v) => updateFilter({ venue: v })}
              options={tags?.venues?.map((v) => ({
                value: v,
                label: `${v} (${tags.venue_counts?.[v] ?? 0})`,
              }))}
            />
          </Col>
          <Col>
            <Select
              placeholder="年份"
              allowClear
              style={{ width: 130 }}
              value={filters.year}
              onChange={(v) => updateFilter({ year: v })}
              options={tags?.years?.map((y) => ({
                value: y,
                label: `${y} (${tags.year_counts?.[String(y)] ?? 0})`,
              }))}
            />
          </Col>
          <Col>
            <Select
              placeholder="级别"
              allowClear
              style={{ width: 120 }}
              value={filters.venue_tier}
              onChange={(v) => updateFilter({ venue_tier: v })}
              options={tags?.venue_tiers?.map((t) => ({
                value: t,
                label: t,
              }))}
            />
          </Col>
          <Col>
            <Space size={4}>
              <Typography.Text type="secondary">包含撤稿</Typography.Text>
              <Switch
                size="small"
                checked={!!filters.include_withdrawn}
                onChange={(v) => updateFilter({ include_withdrawn: v || undefined })}
              />
            </Space>
          </Col>
          <Col flex="auto">
            <Input.Search
              placeholder="搜索标题..."
              allowClear
              onSearch={(v) => {
                if (v.trim()) window.location.href = `/search?q=${encodeURIComponent(v)}`;
              }}
            />
          </Col>
        </Row>
      </Card>

      <Table
        dataSource={data?.results ?? []}
        columns={columns}
        rowKey={(r) => r.paper_id || String(r.doc_id)}
        loading={isLoading}
        size="small"
        pagination={{
          current: filters.page,
          pageSize: filters.page_size,
          total: data?.total ?? 0,
          showSizeChanger: true,
          showTotal: (t) => `共 ${t} 篇论文`,
          onChange: (page, pageSize) =>
            setFilters((prev) => ({ ...prev, page, page_size: pageSize })),
        }}
      />
    </Space>
  );
}
