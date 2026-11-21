import { useState } from 'react';
import {
  Table, Button, Space, Typography, Upload, Modal, Form, Input, InputNumber,
  Switch, Tag, Badge, Popconfirm, message, Segmented, theme,
} from 'antd';
import type { BadgeProps } from 'antd';
import {
  UploadOutlined, DeleteOutlined, EditOutlined, DownloadOutlined,
  SendOutlined, GlobalOutlined,
} from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import type { UploadFile } from 'antd';

import {
  useDocuments, useUpdateDocument, useDeleteDocument, useUploadDocument,
} from '../../api/hooks/useDocuments';
import { useCrawlUrls } from '../../api/hooks/useCrawl';
import type { DocumentBrief } from '../../types';
import { statusColor, statusLabel } from '../../constants';
import { parseCreatedAt, relativeTimeFrom } from '../taskTime';

type SourceFilter = 'all' | 'paper' | 'webpage' | 'upload';

const sourceLabel: Record<string, { text: string; color: string }> = {
  paper: { text: '论文', color: 'blue' },
  webpage: { text: '网页', color: 'green' },
  upload: { text: '上传', color: 'orange' },
};

/**
 * Settings → 数据管理 (F1-e). The upload / edit / delete / URL-crawl-submit
 * controls that used to live inline on the public 文档库 browse page
 * (Documents.tsx) — that page is now a pure, read-only browse surface
 * identical on public and internal; every write action lives only here,
 * behind ConsoleGate.
 */
export default function DataManagementPanel() {
  const { token } = theme.useToken();
  const qc = useQueryClient();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all');

  const { data, isLoading, isFetching } = useDocuments(
    page, pageSize, sourceFilter !== 'all' ? sourceFilter : undefined,
  );

  const uploadMut = useUploadDocument();
  const updateMut = useUpdateDocument();
  const deleteMut = useDeleteDocument();
  const crawlMut = useCrawlUrls();

  const [uploadModal, setUploadModal] = useState(false);
  const [editModal, setEditModal] = useState(false);
  const [editDoc, setEditDoc] = useState<DocumentBrief | null>(null);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploadTitle, setUploadTitle] = useState('');
  const [editForm] = Form.useForm();

  const [crawlUrls, setCrawlUrls] = useState('');
  const [crawlUseProxy, setCrawlUseProxy] = useState(false);
  const [crawlAutoIndex, setCrawlAutoIndex] = useState(true);
  const [crawlMaxConcurrent, setCrawlMaxConcurrent] = useState(3);

  const handleUpload = () => {
    const file = fileList[0]?.originFileObj;
    if (!file) return message.warning('请选择文件');
    uploadMut.mutate(
      { file, title: uploadTitle || undefined, autoIndex: true },
      {
        onSuccess: () => {
          message.success('已上传');
          setUploadModal(false);
          setFileList([]);
          setUploadTitle('');
        },
        onError: (e) => message.error(String(e)),
      },
    );
  };

  const handleEdit = () => {
    if (!editDoc) return;
    editForm.validateFields().then((vals) => {
      updateMut.mutate(
        { docId: editDoc.id, title: vals.title },
        { onSuccess: () => { message.success('已更新'); setEditModal(false); }, onError: (e) => message.error(String(e)) },
      );
    });
  };

  const handleCrawl = () => {
    const list = crawlUrls.split('\n').map((u) => u.trim()).filter(Boolean);
    if (list.length === 0) return message.warning('至少输入一个 URL');
    crawlMut.mutate(
      { urls: list, use_proxy: crawlUseProxy, auto_index: crawlAutoIndex, max_concurrent: crawlMaxConcurrent },
      {
        onSuccess: (r) => {
          message.success(r.message || `已提交 ${list.length} 个 URL`);
          setCrawlUrls('');
          qc.invalidateQueries({ queryKey: ['documents'] });
        },
        onError: (e) => message.error(String(e)),
      },
    );
  };

  const columns = [
    { title: '标题', dataIndex: 'title', ellipsis: true },
    {
      title: '来源', dataIndex: 'source_type', width: 80,
      render: (v: string) => { const m = sourceLabel[v] ?? { text: v, color: 'default' }; return <Tag color={m.color}>{m.text}</Tag>; },
    },
    {
      title: '状态', dataIndex: 'status', width: 90,
      render: (s: string) => (
        <Badge status={(statusColor[s] as BadgeProps['status']) ?? 'default'} text={statusLabel[s] ?? s} />
      ),
    },
    { title: '片段', dataIndex: 'chunk_count', width: 70 },
    {
      title: '创建时间', dataIndex: 'created_at', width: 110,
      render: (v: string) => relativeTimeFrom(parseCreatedAt(v)),
    },
    {
      title: '操作', width: 120,
      render: (_: unknown, r: DocumentBrief) => (
        <Space size="small">
          <Button size="small" icon={<EditOutlined />} onClick={() => { setEditDoc(r); editForm.setFieldsValue({ title: r.title }); setEditModal(true); }} />
          {r.download_url && <Button size="small" icon={<DownloadOutlined />} href={r.download_url} />}
          <Popconfirm title="确定删除？" onConfirm={() => deleteMut.mutate(r.id, { onSuccess: () => message.success('已删除') })}>
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Space orientation="vertical" size="large" style={{ width: '100%' }}>
      <Space wrap>
        <Button type="primary" icon={<UploadOutlined />} onClick={() => setUploadModal(true)}>上传文档</Button>
        <Segmented
          value={sourceFilter}
          onChange={(v) => { setSourceFilter(v as SourceFilter); setPage(1); }}
          options={[
            { value: 'all', label: '全部' },
            { value: 'paper', label: '论文' },
            { value: 'webpage', label: '网页' },
            { value: 'upload', label: '上传' },
          ]}
        />
      </Space>

      <Table
        dataSource={data?.documents ?? []}
        columns={columns}
        rowKey="id"
        loading={isLoading || isFetching}
        size="small"
        pagination={{
          current: page, pageSize, total: data?.total ?? 0,
          showSizeChanger: true,
          onChange: (p, ps) => { setPage(p); setPageSize(ps); },
        }}
      />

      <div style={{ borderTop: `1px solid ${token.colorBorderSecondary}`, paddingTop: 16 }}>
        <Typography.Title level={5} style={{ marginTop: 0 }}>
          <GlobalOutlined style={{ marginRight: 8, color: token.colorSuccess }} />采集网页
        </Typography.Title>
        <Space orientation="vertical" size="middle" style={{ width: '100%', maxWidth: 640 }}>
          <Input.TextArea
            rows={5}
            placeholder={'https://example.com/article-1\nhttps://example.com/article-2'}
            value={crawlUrls}
            onChange={(e) => setCrawlUrls(e.target.value)}
          />
          <Space size={20} wrap>
            <Space size={6}><Typography.Text>使用代理</Typography.Text><Switch checked={crawlUseProxy} onChange={setCrawlUseProxy} /></Space>
            <Space size={6}><Typography.Text>自动索引</Typography.Text><Switch checked={crawlAutoIndex} onChange={setCrawlAutoIndex} /></Space>
            <Space size={6}>
              <Typography.Text>并发数</Typography.Text>
              <InputNumber min={1} max={10} size="small" value={crawlMaxConcurrent} onChange={(v) => setCrawlMaxConcurrent(v ?? 3)} />
            </Space>
          </Space>
          <Space>
            <Button type="primary" icon={<SendOutlined />} loading={crawlMut.isPending} onClick={handleCrawl}>提交采集</Button>
            <Link to="/tasks"><Button type="link">在任务页查看进度</Button></Link>
          </Space>
        </Space>
      </div>

      <Modal title="上传文档" open={uploadModal} onOk={handleUpload} onCancel={() => setUploadModal(false)} confirmLoading={uploadMut.isPending}>
        <Space orientation="vertical" style={{ width: '100%' }}>
          <Upload fileList={fileList} beforeUpload={() => false} onChange={({ fileList: fl }) => setFileList(fl.slice(-1))} maxCount={1}>
            <Button icon={<UploadOutlined />}>选择文件</Button>
          </Upload>
          <Input placeholder="标题（可选）" value={uploadTitle} onChange={(e) => setUploadTitle(e.target.value)} />
        </Space>
      </Modal>

      <Modal title="编辑文档" open={editModal} onOk={handleEdit} onCancel={() => setEditModal(false)} confirmLoading={updateMut.isPending}>
        <Form form={editForm} layout="vertical">
          <Form.Item name="title" label="标题" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
        </Form>
      </Modal>
    </Space>
  );
}
