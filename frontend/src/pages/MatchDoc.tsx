import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  Card,
  Input,
  Segmented,
  Button,
  Tag,
  Space,
  Typography,
  Empty,
  Collapse,
  InputNumber,
  Row,
  Col,
  Alert,
  Modal,
  message,
  theme,
} from 'antd';
import {
  FileSearchOutlined,
  ThunderboltOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons';
import { AnimatePresence, motion } from 'framer-motion';
import Markdown from 'react-markdown';
import { clipMarkdown, displayMath, rehypePlugins, remarkPlugins } from '../lib/markdown';
import axios from 'axios';
import { useMatchDoc } from '../api/hooks/useMatchDoc';
import { login } from '../api/hooks/useAuth';
import { useAuthUI } from '../stores/authStore';
import { heatBg, scoreBadgeColor } from '../components/ChunkScoreView';
import type { MatchAgg, MatchInputType, MatchDocRequest, PaperMatch } from '../types';

const { Title, Text, Paragraph } = Typography;

// Normalize a value within [min,max] → [0,1] for relative score-heat. When the
// set is degenerate (all equal) everything reads "warm" so the UI isn't blank.
function normFactory(values: number[]) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;
  return (v: number) => (span < 1e-9 ? 0.6 : (v - lo) / span);
}

function MatchedSpans({ spans }: { spans: PaperMatch['matched'] }) {
  const norm = useMemo(() => normFactory(spans.map((s) => s.sim)), [spans]);
  return (
    <Space orientation="vertical" size={8} style={{ width: '100%' }}>
      {spans.map((s) => (
        <div
          key={`${s.query_chunk_idx}-${s.paper_chunk.slice(0, 12)}`}
          style={{
            borderLeft: `4px solid ${heatBg(norm(s.sim))}`,
            background: 'rgba(127,127,127,0.05)',
            borderRadius: 6,
            padding: '8px 12px',
          }}
        >
          <Space size={6} style={{ marginBottom: 4 }}>
            <Tag color="default" style={{ margin: 0, fontFamily: 'monospace' }}>
              你的第 {s.query_chunk_idx + 1} 段
            </Tag>
            <Tag color={scoreBadgeColor(s.sim)} style={{ margin: 0, fontFamily: 'monospace', fontWeight: 600 }}>
              {s.sim.toFixed(4)}
            </Tag>
          </Space>
          <div style={{ fontSize: 13, lineHeight: 1.6, opacity: 0.9 }}>
            <Markdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins}>
              {displayMath(clipMarkdown(s.paper_chunk, 480))}
            </Markdown>
          </div>
        </div>
      ))}
    </Space>
  );
}

function PaperMatchCard({ paper, rank, heat }: { paper: PaperMatch; rank: number; heat: number }) {
  const { token } = theme.useToken();
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: Math.min(rank * 0.03, 0.4) }}
    >
      <Card
        size="small"
        styles={{ body: { padding: 16 } }}
        style={{ borderLeft: `4px solid ${heatBg(heat)}` }}
      >
        <Row gutter={12} wrap={false} align="top">
          <Col flex="none">
            <div
              style={{
                width: 30, height: 30, borderRadius: 8, display: 'flex',
                alignItems: 'center', justifyContent: 'center', fontWeight: 700,
                color: '#fff', background: 'linear-gradient(135deg,#818cf8,#4F46E5)',
              }}
            >
              {rank + 1}
            </div>
          </Col>
          <Col flex="auto">
            <Space orientation="vertical" size={4} style={{ width: '100%' }}>
              <Space wrap size={6}>
                <Link to={`/papers/${paper.paper_id}`} style={{ fontWeight: 600, fontSize: 15 }}>
                  {paper.title || paper.paper_id}
                </Link>
              </Space>
              <Space wrap size={6}>
                {paper.venue && <Tag color="geekblue">{paper.venue}</Tag>}
                {paper.year != null && <Tag>{paper.year}</Tag>}
                <Tag color="purple">覆盖 {paper.coverage} 段</Tag>
                <Tag color={scoreBadgeColor(paper.score)} style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                  {paper.score.toFixed(4)}
                </Tag>
                <Link to={`/papers/${paper.paper_id}`} style={{ fontSize: 12, color: token.colorTextSecondary }}>
                  论文详情 <ArrowRightOutlined style={{ fontSize: 10 }} />
                </Link>
              </Space>
              <Collapse
                ghost
                size="small"
                items={[{
                  key: 'spans',
                  label: <Text type="secondary" style={{ fontSize: 12 }}>查看 {paper.matched.length} 段命中明细</Text>,
                  children: <MatchedSpans spans={paper.matched} />,
                }]}
              />
            </Space>
          </Col>
        </Row>
      </Card>
    </motion.div>
  );
}

export default function MatchDoc() {
  const { token } = theme.useToken();
  const [mode, setMode] = useState<MatchInputType>('text');
  const [text, setText] = useState('');
  const [paperId, setPaperId] = useState('');
  const [agg, setAgg] = useState<MatchAgg>('coverage');
  const [topK, setTopK] = useState(20);
  const [perChunkK, setPerChunkK] = useState(50);

  const mut = useMatchDoc();
  const resp = mut.data;

  // Paid-tier gate (PUBLIC_MODE): match-doc is metered → gateway answers 401
  // (anon) / 402 (no credits). Prompt login / open topup, same UX as Search.
  const openAccount = useAuthUI((s) => s.openAccount);
  const authStatus = axios.isAxiosError(mut.error) ? mut.error.response?.status : undefined;
  const authPromptedRef = useRef(false);
  useEffect(() => {
    if (!mut.isError) { authPromptedRef.current = false; return; }
    if (authPromptedRef.current) return;
    if (authStatus === 401) {
      authPromptedRef.current = true;
      Modal.confirm({
        title: '请先登录',
        content: '文档匹配需登录账户后使用，新用户登录即注册。',
        okText: '去登录', cancelText: '取消', onOk: login,
      });
    } else if (authStatus === 402) {
      authPromptedRef.current = true;
      message.warning('余额不足，请充值后再匹配');
      openAccount('topup');
    }
  }, [mut.isError, authStatus, openAccount]);

  // Deep link from a paper detail page (关联搜索): /match-doc?paper_id=X → 自动跑一次邻居匹配。
  const [searchParams] = useSearchParams();
  const autoFired = useRef(false);
  useEffect(() => {
    if (autoFired.current) return;
    const pid = searchParams.get('paper_id');
    if (pid && pid.trim()) {
      autoFired.current = true;
      setMode('paper_id');
      setPaperId(pid.trim());
      mut.mutate({ input_type: 'paper_id', agg: 'coverage', top_k: 20, per_chunk_k: 50, paper_id: pid.trim() });
    }
  }, [searchParams, mut]);

  const heatNorm = useMemo(
    () => (resp ? normFactory(resp.papers.map((p) => p.score)) : () => 0.6),
    [resp],
  );

  const canSubmit =
    (mode === 'text' && text.trim().length > 0) ||
    (mode === 'paper_id' && paperId.trim().length > 0);

  const submit = () => {
    if (!canSubmit) return;
    const req: MatchDocRequest = {
      input_type: mode,
      agg,
      top_k: topK,
      per_chunk_k: perChunkK,
      ...(mode === 'text' ? { text } : {}),
      ...(mode === 'paper_id' ? { paper_id: paperId.trim() } : {}),
    };
    mut.mutate(req);
  };

  return (
    <Space orientation="vertical" size={16} style={{ width: '100%' }}>
      <Title level={3} style={{ marginBottom: 0 }}>
        <FileSearchOutlined style={{ color: token.colorPrimary, marginRight: 8 }} />
        文档匹配
      </Title>

      <Card size="small">
        <Space orientation="vertical" size={12} style={{ width: '100%' }}>
          <Segmented
            value={mode}
            onChange={(v) => setMode(v as MatchInputType)}
            options={[
              { label: '粘贴文本', value: 'text' },
              { label: '已入库论文', value: 'paper_id' },
            ]}
          />

          {mode === 'text' ? (
            <Input.TextArea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="粘贴一段摘要 / 段落 / 方法描述…… 越完整匹配越准。"
              autoSize={{ minRows: 6, maxRows: 16 }}
              showCount
              maxLength={20000}
            />
          ) : (
            <Input
              value={paperId}
              onChange={(e) => setPaperId(e.target.value)}
              onPressEnter={submit}
              placeholder="输入已入库论文的 paper_id（如 aaai-2022-20281）—— 找它的内容邻居（自动排除自身）"
              allowClear
            />
          )}

          <Row gutter={[16, 12]} align="middle">
            <Col>
              <Space size={6}>
                <Text type="secondary">打分</Text>
                <Segmented
                  size="small"
                  value={agg}
                  onChange={(v) => setAgg(v as MatchAgg)}
                  options={[
                    { label: '整篇重合', value: 'coverage' },
                    { label: '单段最强', value: 'maxsim' },
                  ]}
                />
              </Space>
            </Col>
            <Col>
              <Space size={6}>
                <Text type="secondary">返回论文</Text>
                <InputNumber size="small" min={1} max={100} value={topK} onChange={(v) => setTopK(v ?? 20)} />
              </Space>
            </Col>
            <Col>
              <Space size={6}>
                <Text type="secondary">每段召回</Text>
                <InputNumber size="small" min={1} max={200} value={perChunkK} onChange={(v) => setPerChunkK(v ?? 50)} />
              </Space>
            </Col>
            <Col flex="auto" style={{ textAlign: 'right' }}>
              <Button
                type="primary"
                icon={<ThunderboltOutlined />}
                loading={mut.isPending}
                disabled={!canSubmit}
                onClick={submit}
              >
                开始匹配
              </Button>
            </Col>
          </Row>
        </Space>
      </Card>

      {mut.isError && authStatus !== 401 && authStatus !== 402 && (
        <Alert
          type="error"
          showIcon
          message="匹配失败"
          description={(mut.error as any)?.response?.data?.detail || (mut.error as any)?.message || '未知错误'}
        />
      )}

      {resp && (
        <Card
          size="small"
          title={
            <Space wrap size={8}>
              <Text strong>{resp.papers.length}</Text>
              <Text type="secondary">篇相近论文</Text>
              {typeof resp.metrics?.query_chunks === 'number' && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  · 查询切 {String(resp.metrics.query_chunks)} 段
                </Text>
              )}
              {typeof resp.metrics?.total_ms === 'number' && (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  · {Number(resp.metrics.total_ms).toFixed(0)} ms
                </Text>
              )}
            </Space>
          }
        >
          {resp.papers.length === 0 ? (
            <Empty
              description={
                resp.query_chunks.length === 0
                  ? '输入无法切分成查询块（文本太短或为空 / 该论文没有可用正文块）—— 请粘贴更完整的内容。'
                  : '没找到相近论文（试试放宽筛选或换「单段最强」模式）。'
              }
            />
          ) : (
            <Space orientation="vertical" size={10} style={{ width: '100%' }}>
              <AnimatePresence>
                {resp.papers.map((p, i) => (
                  <PaperMatchCard key={p.paper_id || p.doc_id} paper={p} rank={i} heat={heatNorm(p.score)} />
                ))}
              </AnimatePresence>
            </Space>
          )}
        </Card>
      )}

      {!resp && !mut.isPending && !mut.isError && (
        <Card size="small">
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                输入文档后点「开始匹配」
              </Paragraph>
            }
          />
        </Card>
      )}
    </Space>
  );
}
