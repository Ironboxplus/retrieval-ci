import { useState, useEffect, useMemo, useRef } from 'react';
import { useParams, Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Descriptions,
  Tag,
  Button,
  Space,
  Typography,
  Card,
  Spin,
  Collapse,
  message,
  Tooltip,
  Result,
  Input,
  Empty,
} from 'antd';
import {
  ArrowLeftOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
  FilePdfOutlined,
  LinkOutlined,
  CheckCircleFilled,
  WarningFilled,
  ThunderboltOutlined,
  CopyOutlined,
  ReadOutlined,
  SearchOutlined,
  DeploymentUnitOutlined,
  LockOutlined,
} from '@ant-design/icons';
import {
  usePaperDetail,
  usePaperIndexStatus,
  useTriggerPaperIndex,
  usePaperFulltext,
} from '../api/hooks/usePapers';
import { useSearchStream } from '../api/hooks/useSearchStream';
import { useAuthUI } from '../stores/authStore';
import { useMe } from '../api/hooks/useAuth';
import api from '../api/client';
import type { SearchRequest } from '../types';
import ChunkScoreView from '../components/ChunkScoreView';
import MarkdownDocument from '../components/MarkdownDocument';
import MarkdownSnippet from '../components/MarkdownSnippet';
import { groupPapersClientSide } from '../components/groupPapers';
import { PUBLIC_MODE } from '../constants';
import { copyText } from '../utils/clipboard';

// In-paper search: filter retrieval to this one paper, keep every matched chunk
// so the heatmap can show how each part of the paper matches the query (W87).
const IN_PAPER_TOP_K = 100;

export default function PaperDetail() {
  const { paperId } = useParams<{ paperId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: paper, isLoading, isError, error, refetch } = usePaperDetail(paperId);
  const { data: idx, isLoading: idxLoading, isError: idxIsError, refetch: refetchIdx } = usePaperIndexStatus(paperId);
  const triggerIdx = useTriggerPaperIndex();
  const [showPdf, setShowPdf] = useState(false);
  const [showFulltext, setShowFulltext] = useState(false);
  const fulltext = usePaperFulltext(showFulltext ? paperId : undefined);
  const openLogin = useAuthUI((s) => s.openLogin);
  const openAccount = useAuthUI((s) => s.openAccount);
  const { data: me } = useMe();
  const canAccessPdf = !PUBLIC_MODE || me?.tier === 'internal' || me?.tier === 'admin';
  const paysForMarkdown = PUBLIC_MODE && me?.tier === 'paying';
  const fulltextStats = useMemo(() => {
    const content = fulltext.data?.content ?? '';
    return {
      characters: content.length,
      lines: content ? content.split(/\r?\n/).length : 0,
    };
  }, [fulltext.data?.content]);

  // ─── In-paper search (reuses the default /api/search/stream pipeline) ───
  const urlQ = searchParams.get('q') ?? '';
  const inPaperSearch = useSearchStream();
  const [inPaperQuery, setInPaperQuery] = useState(urlQ);
  const lastFired = useRef<string>('');

  const fireInPaper = (q: string) => {
    const query = q.trim();
    if (!query || !paperId) return;
    const req: SearchRequest = {
      query, method: 'hybrid', top_k: IN_PAPER_TOP_K,
      dedup_by_paper: false, paper_ids: [paperId],
    };
    lastFired.current = `${paperId}::${query}`;
    if (!inPaperSearch.hydrate(req)) inPaperSearch.mutate(req);
  };

  const runInPaper = (q: string) => {
    const query = q.trim();
    if (!query || !paperId) return;
    setInPaperQuery(query);
    const next = new URLSearchParams(searchParams);
    next.set('tab', 'chunks');
    next.set('q', query);
    setSearchParams(next, { replace: true });
    fireInPaper(query);
  };

  // Auto-run when arriving via a deep link (search card → 在此论文中检索).
  useEffect(() => {
    if (!paperId || !urlQ.trim()) return;
    const key = `${paperId}::${urlQ}`;
    if (lastFired.current === key) return;
    setInPaperQuery(urlQ);
    fireInPaper(urlQ);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paperId, urlQ]);

  const inPaperResult = useMemo(() => {
    const d = inPaperSearch.data;
    if (!d) return undefined;
    const ps = (d.papers && d.papers.length) ? d.papers : groupPapersClientSide(d.results || []);
    return ps[0];
  }, [inPaperSearch.data]);
  const inPaperQueryEcho = inPaperSearch.data?.query ?? inPaperQuery;

  const goBack = () => {
    if (window.history.length > 1) navigate(-1);
    else navigate('/papers');
  };

  if (isLoading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;
  if (isError || !paper) {
    const detail = (error as { response?: { data?: { detail?: string }; status?: number }; message?: string } | null);
    const status = detail?.response?.status;
    const reason = detail?.response?.data?.detail ?? detail?.message ?? '论文未找到';
    // 401 → paper content is login-gated (metadata + full-text + PDF). Prompt login
    // rather than show a generic failure (W127).
    if (status === 401) {
      return (
        <Result
          status="403"
          title="请先登录"
          subTitle="论文详情需登录后查看，新用户登录即注册（浏览不扣费）。"
          extra={
            <Space>
              <Button type="primary" onClick={openLogin}>登录 / 注册</Button>
              <Button icon={<ArrowLeftOutlined />} onClick={goBack}>返回</Button>
            </Space>
          }
        />
      );
    }
    return (
      <Result
        status={status === 404 ? '404' : 'warning'}
        title={status === 404 ? '论文未找到' : '加载失败'}
        subTitle={`paper_id=${paperId} · ${reason}`}
        extra={
          <Space>
            <Button type="primary" icon={<ArrowLeftOutlined />} onClick={goBack}>返回</Button>
            <Link to="/papers"><Button>论文库</Button></Link>
            <Button onClick={() => refetch()}>重试</Button>
          </Space>
        }
      />
    );
  }

  const handleIndex = () => {
    if (!paperId) return;
    triggerIdx.mutate(paperId, {
      onSuccess: (r) => {
        message.success(`已触发：${r.chunks_embedded ?? 0} chunks embedded`);
        refetchIdx();
      },
      onError: (e: unknown) => {
        const detail = (e as { response?: { data?: { detail?: string } }; message?: string });
        message.error(detail.response?.data?.detail ?? detail.message ?? '索引触发失败');
      },
    });
  };

  const meta = (paper as unknown as Record<string, unknown>).metadata as Record<string, unknown> | undefined;
  const authors = (meta?.authors as string[]) ?? [];
  const venue = (meta?.venue as string) ?? (paper as unknown as Record<string, unknown>).source as string ?? '-';
  const year = meta?.year as number;
  const abstract = meta?.abstract as string;
  const paperType = meta?.paper_type as string;
  const sourceUrl = (paper as unknown as Record<string, unknown>).source as string;
  const chunks = (paper as unknown as Record<string, unknown>).chunks as Array<{
    id: number;
    content: string;
  }> | undefined;

  return (
    <Space orientation="vertical" size="large" style={{ width: '100%' }}>
      <Space>
        <Button icon={<ArrowLeftOutlined />} onClick={goBack}>返回</Button>
        <Link to="/papers"><Button type="text">论文库</Button></Link>
      </Space>

      <Space align="center" wrap>
        <Typography.Title level={4} style={{ margin: 0 }}>{paper.title}</Typography.Title>
        {idxLoading && !idx ? (
          <Tag>索引状态加载中…</Tag>
        ) : idxIsError ? (
          <Tag color="default">状态未知</Tag>
        ) : idx ? (
          idx.indexed && idx.qdrant_points_count >= idx.sql_chunks_count ? (
            <Tag color="success" icon={<CheckCircleFilled />}>已索引</Tag>
          ) : idx.qdrant_points_count > 0 ? (
            <Tag color="warning" icon={<WarningFilled />}>部分索引 {idx.qdrant_points_count}/{idx.sql_chunks_count}</Tag>
          ) : (
            <Tag color="error" icon={<WarningFilled />}>未索引</Tag>
          )
        ) : (
          <Tag color="default">无索引数据</Tag>
        )}
      </Space>

      {authors.length > 0 && (
        <Typography.Text type="secondary">{authors.join(', ')}</Typography.Text>
      )}

      <Descriptions bordered column={{ xs: 1, sm: 2 }} size="small">
        <Descriptions.Item label="Paper ID">{paper.paper_id}</Descriptions.Item>
        <Descriptions.Item label="会议"><Tag color="blue">{venue}</Tag></Descriptions.Item>
        {year && <Descriptions.Item label="年份">{year}</Descriptions.Item>}
        {paperType && paperType !== 'unknown' && (
          <Descriptions.Item label="类型"><Tag>{paperType}</Tag></Descriptions.Item>
        )}
        <Descriptions.Item label="Chunks">{(paper as unknown as Record<string, unknown>).chunk_count as number ?? 0}</Descriptions.Item>
        <Descriptions.Item label="Doc ID">{(paper as unknown as Record<string, unknown>).doc_id as number}</Descriptions.Item>
      </Descriptions>

      <Space wrap>
        {sourceUrl && (
          <a href={sourceUrl} target="_blank" rel="noreferrer"><Button icon={<LinkOutlined />}>来源页面</Button></a>
        )}
        {canAccessPdf && (
          <Button icon={showPdf ? <EyeInvisibleOutlined /> : <EyeOutlined />} onClick={() => setShowPdf((v) => !v)}>
            {showPdf ? '隐藏 PDF' : '预览 PDF'}
          </Button>
        )}
        {canAccessPdf && paperId && (
          <a href={`/api/papers/${encodeURIComponent(paperId)}/pdf`} download target="_blank" rel="noreferrer">
            <Button icon={<FilePdfOutlined />}>下载 PDF</Button>
          </a>
        )}
        {!canAccessPdf && PUBLIC_MODE && me && (
          <Tooltip title="PDF 原文件仅向 member/internal/admin 开放">
            <Button disabled icon={<LockOutlined />}>PDF 会员专享</Button>
          </Tooltip>
        )}
        <Button
          icon={<CopyOutlined />}
          onClick={() => {
            if (!paperId) return;
            api.get(`/papers/${paperId}/bibtex`).then((res) => {
              const text = (res.data as { bibtex: string }).bibtex;
              return copyText(text);
            }).then(() => {
              message.success('BibTeX 已复制到剪贴板');
            }).catch(() => message.error('BibTeX 获取失败'));
          }}
        >复制 BibTeX</Button>
        <Button icon={showFulltext ? <EyeInvisibleOutlined /> : <ReadOutlined />} onClick={() => setShowFulltext((v) => !v)}>
          {showFulltext ? '隐藏全文' : paysForMarkdown ? '阅读全文（付费）' : '阅读全文'}
        </Button>
        <Link to={`/match-doc?paper_id=${encodeURIComponent(paperId ?? '')}`}>
          <Button icon={<DeploymentUnitOutlined />}>找相近论文</Button>
        </Link>
        {idx && !idx.indexed && idx.sql_chunks_count > 0 && (
          <Button danger icon={<ThunderboltOutlined />} loading={triggerIdx.isPending} onClick={handleIndex}>重新索引</Button>
        )}
      </Space>

      {showPdf && canAccessPdf && (
        <iframe src={`/api/papers/${paperId}/pdf`} title="PDF Preview" style={{ width: '100%', height: '80vh', border: 'none' }} />
      )}

      {abstract && (
        <Card title="摘要" size="small">
          <div style={{ lineHeight: 1.7 }}><MarkdownSnippet source={abstract} /></div>
        </Card>
      )}

      {/* ─── In-paper search: score heatmap of matched parts (W87) ─── */}
      <Card
        size="small"
        title={<Space><SearchOutlined />在此论文中检索</Space>}
        extra={inPaperResult ? <Tag color="blue">{inPaperResult.chunk_count} 段命中{inPaperSearch.data?.pipeline_metrics ? ` · ${inPaperSearch.data.pipeline_metrics.total_ms.toFixed(0)}ms` : ''}</Tag> : null}
      >
        <Input.Search
          placeholder="在本论文全文中检索，回车触发——热力图展示各部分的匹配程度"
          enterButton={<><SearchOutlined /> 检索</>}
          size="middle"
          loading={inPaperSearch.isPending}
          value={inPaperQuery}
          onChange={(e) => setInPaperQuery(e.target.value)}
          onSearch={runInPaper}
          allowClear
        />
        {inPaperSearch.isPending && <Spin style={{ display: 'block', margin: '24px auto' }} />}
        {inPaperResult && inPaperResult.chunks.length > 0 && (
          <div style={{ marginTop: 14 }}>
            <ChunkScoreView chunks={inPaperResult.chunks} query={inPaperQueryEcho} showOverview />
          </div>
        )}
        {inPaperSearch.stage === 'complete' && (!inPaperResult || inPaperResult.chunks.length === 0) && (
          <Empty style={{ marginTop: 16 }} image={Empty.PRESENTED_IMAGE_SIMPLE} description="本论文中没有命中该查询的片段" />
        )}
        {inPaperSearch.stage === 'error' && (
          <Result status="warning" subTitle={inPaperSearch.error || '检索失败'} style={{ padding: '16px 0' }} />
        )}
      </Card>

      {showFulltext && (
        <Card
          title={<Space size={7}><ReadOutlined />Markdown 全文</Space>}
          size="small"
          styles={{ body: { padding: 0 } }}
          extra={fulltext.data ? (
            <Space size={5} wrap>
              <Tag>{(fulltext.data.size / 1024).toFixed(0)} KB</Tag>
              <Tag>{fulltextStats.characters.toLocaleString()} 字符</Tag>
              <Tag>{fulltextStats.lines.toLocaleString()} 行</Tag>
              <Button
                size="small"
                icon={<CopyOutlined />}
                onClick={() => copyText(fulltext.data.content)
                  .then(() => message.success('Markdown 源码已复制'))
                  .catch(() => message.error('复制失败'))}
              >
                复制 Markdown 源码
              </Button>
            </Space>
          ) : null}
        >
          {fulltext.isLoading && <Spin style={{ display: 'block', margin: '40px auto' }} />}
          {fulltext.isError && (
            (() => {
              const response = (fulltext.error as { response?: { status?: number; data?: { detail?: string } } })?.response;
              const insufficient = response?.status === 402;
              return (
                <Result
                  status="warning"
                  title={insufficient ? '余额不足' : '无法加载全文'}
                  subTitle={response?.data?.detail || '文件不存在或无法访问'}
                  extra={insufficient ? <Button type="primary" onClick={() => openAccount('topup')}>充值后阅读</Button> : undefined}
                />
              );
            })()
          )}
          {fulltext.data && (
            <MarkdownDocument source={fulltext.data.content} />
          )}
        </Card>
      )}

      {chunks && chunks.length > 0 && (
        <Card title={`内容片段 · 按正文顺序 (${chunks.length})`} size="small">
          <Collapse
            size="small"
            items={chunks.slice(0, 20).map((c, i) => ({
              key: c.id,
              label: `片段 #${i + 1}`,
              children: (
                <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                  <MarkdownSnippet source={c.content ?? ''} max={500} />
                </div>
              ),
            }))}
          />
          {chunks.length > 20 && (
            <Typography.Text type="secondary" style={{ marginTop: 8, display: 'block' }}>
              还有 {chunks.length - 20} 个片段未显示（用上方「在此论文中检索」按相关度定位）
            </Typography.Text>
          )}
        </Card>
      )}
    </Space>
  );
}
