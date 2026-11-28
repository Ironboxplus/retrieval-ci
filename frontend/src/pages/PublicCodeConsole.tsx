import { useMemo, useState } from 'react';
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Drawer,
  Form,
  Input,
  InputNumber,
  Pagination,
  Progress,
  Row,
  Select,
  Space,
  Switch,
  Table,
  Tabs,
  Tag,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import type { TableColumnsType } from 'antd';
import {
  CloudDownloadOutlined,
  CodeOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  SyncOutlined,
} from '@ant-design/icons';
import axios from 'axios';

import {
  useCheckPublicCodeUpdates,
  useCreatePublicCodeSource,
  useIndexPublicCodeSources,
  usePublicCodeJobs,
  usePublicCodeSources,
  useSearchPublicCode,
  useUpdatePublicCodePolicy,
  type PublicCodeJob,
  type PublicCodeSearchHit,
  type PublicCodeSearchMode,
  type PublicCodeSource,
  type PublicCodeSourceCreateRequest,
  type PublicCodeUpdatePolicy,
} from '../api/hooks/usePublicCode';
import {
  filterPublicCodeSources,
  summarizeSourceStates,
  sourceStateKey,
  revisionGaps,
  maxSourceChunkCount,
  type SourceStateKey,
} from './publicCodeViewModel';
import MorphGrid from '../components/morph/MorphGrid';
import MorphCard from '../components/morph/MorphCard';

const { Text } = Typography;
const EMPTY_SOURCES: PublicCodeSource[] = [];
const EMPTY_JOBS: PublicCodeJob[] = [];
const SOURCE_PAGE_SIZE = 24;

const STATE_COLOR: Record<SourceStateKey, string> = {
  active: '#10b981', hasUpdate: '#f59e0b', unconfigured: '#94a3b8', checkFailed: '#ef4444',
};
const STATE_LABEL: Record<SourceStateKey, string> = {
  active: '已激活', hasUpdate: '有更新', unconfigured: '未索引', checkFailed: '检查失败',
};

interface SourceFormValues {
  source_id: string;
  domain: string;
  upstream_url: string;
  relative_path: string;
  default_ref: string;
  proxy_profile?: string;
  credential_profile?: string;
  language_overrides?: string;
  update_policy: PublicCodeUpdatePolicy;
  update_interval_minutes?: number;
  index_now: boolean;
}

function apiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((item) => item.msg ?? String(item)).join('; ');
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

function shortRevision(revision: string | null): string {
  return revision ? revision.slice(0, 10) : '无';
}

function formatTime(value: string | number | null): string {
  if (!value) return '-';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  return Number.isNaN(date.valueOf()) ? '-' : date.toLocaleString();
}

const JOB_STATUS_LABEL: Record<PublicCodeJob['status'], string> = {
  queued: '排队中', running: '运行中', completed: '已完成', failed: '失败',
};

function jobState(status: PublicCodeJob['status']) {
  const colors: Record<PublicCodeJob['status'], string> = {
    queued: 'default',
    running: 'processing',
    completed: 'success',
    failed: 'error',
  };
  return <Tag color={colors[status]}>{JOB_STATUS_LABEL[status]}</Tag>;
}

/** Two-color bar standing in for "retrieval → rerank": the caption sentence
 * is gone (design rule §1); the split itself communicates the two scores. */
function ScoreBar({ retrieval, reranker }: { retrieval: number; reranker: number | null }) {
  if (reranker === null) {
    return <Text strong>{retrieval.toFixed(4)}</Text>;
  }
  const total = Math.max(retrieval, reranker, 0.0001);
  return (
    <Space orientation="vertical" size={2} style={{ width: 140 }}>
      <div style={{ display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden', background: 'var(--ant-color-fill-tertiary)' }}>
        <div style={{ width: `${(retrieval / total) * 50}%`, background: '#94a3b8' }} />
        <div style={{ width: `${(reranker / total) * 50}%`, background: '#2563EB' }} />
      </div>
      <Text strong style={{ fontSize: 12 }}>{retrieval.toFixed(4)} → {reranker.toFixed(4)}</Text>
    </Space>
  );
}

/** Three dots on one line — 生效 ← 镜像 ← 上游, left to right — whose spacing
 * (not text) shows staleness (design addendum §1.1): all three cluster
 * together when the revisions agree; a dot separates from its left neighbor
 * the instant that pair diverges. No "active/mirror/remote" words. */
function RevisionDots({ source }: { source: PublicCodeSource }) {
  const { token } = theme.useToken();
  const gaps = revisionGaps(source);
  const GAP = 48;
  const DOT = 9;
  const xActive = 0;
  const xMirror = xActive + gaps.activeMirrorGap * GAP;
  const xRemote = xMirror + gaps.mirrorRemoteGap * GAP;
  const width = xRemote + DOT;
  return (
    <svg width={width} height={DOT} style={{ display: 'block' }}>
      <line x1={DOT / 2} y1={DOT / 2} x2={xRemote + DOT / 2} y2={DOT / 2} stroke={token.colorBorder} strokeWidth={1.5} />
      <circle cx={xActive + DOT / 2} cy={DOT / 2} r={DOT / 2} fill={token.colorPrimary} />
      <circle cx={xMirror + DOT / 2} cy={DOT / 2} r={DOT / 2} fill={token.colorTextSecondary} />
      <circle cx={xRemote + DOT / 2} cy={DOT / 2} r={DOT / 2} fill={gaps.mirrorRemoteGap ? token.colorWarning : token.colorTextTertiary} />
    </svg>
  );
}

/** Counts-by-status as ONE proportional segmented bar (design addendum §1.1)
 * — segment widths are the counts, color is the status; this replaces the
 * old 已配置/已索引/定时/检查异常 row of antd Statistic tiles, whose numbers
 * silently overlapped (a source can be both "indexed" and "scheduled"), with
 * a bar whose four segments are a true partition and always sum to the
 * total. Numbers sit beside the bar, not inside a titled tile. */
function SourceStateBar({ sources }: { sources: PublicCodeSource[] }) {
  const { token } = theme.useToken();
  const counts = summarizeSourceStates(sources);
  const total = Math.max(1, sources.length);
  const order: SourceStateKey[] = ['active', 'hasUpdate', 'unconfigured', 'checkFailed'];
  return (
    <div>
      <div style={{ display: 'flex', height: 14, borderRadius: 4, overflow: 'hidden' }}>
        {order.map((key) => counts[key] > 0 && (
          <div
            key={key}
            style={{ width: `${(counts[key] / total) * 100}%`, background: STATE_COLOR[key] }}
            title={`${STATE_LABEL[key]} ${counts[key]}`}
          />
        ))}
      </div>
      <div style={{ marginTop: 6, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        {order.map((key) => counts[key] > 0 && (
          <span key={key} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: token.colorTextSecondary }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: STATE_COLOR[key], display: 'inline-block' }} />
            {STATE_LABEL[key]} {counts[key]}
          </span>
        ))}
      </div>
    </div>
  );
}

/** 生效 / 上游 revisions side by side, each named; a differing upstream is
 * called out as 有更新 rather than encoded in dot spacing. */
function RevisionLine({ source }: { source: PublicCodeSource }) {
  const behind = !!source.last_remote_revision && source.last_remote_revision !== source.active_revision;
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-xs">
      <span className="text-muted-foreground">生效</span>
      <span className="truncate font-mono">{shortRevision(source.active_revision)}</span>
      <span className="text-muted-foreground">上游</span>
      <span className={behind ? 'truncate font-mono text-warning' : 'truncate font-mono'}>
        {shortRevision(source.last_remote_revision)}
        {source.last_remote_revision ? (behind ? ' · 有更新' : ' · 已同步') : ' · 未检查'}
      </span>
    </div>
  );
}

function SourceCompactCard({
  source, selected, onToggleSelect,
}: {
  source: PublicCodeSource;
  selected: boolean;
  onToggleSelect: () => void;
}) {
  const stateKey = sourceStateKey(source);
  return (
    <MorphCard>
      <div className="flex items-start gap-2">
        {/* Only the checkbox claims its click; the rest of the card expands it. */}
        <input
          type="checkbox"
          checked={selected}
          onClick={(e) => e.stopPropagation()}
          onChange={onToggleSelect}
          className="mt-1 cursor-pointer"
          aria-label={`选择 ${source.source_id}`}
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{source.source_id}</div>
          <div className="text-xs text-muted-foreground">{source.domain}</div>
        </div>
        <span
          className="shrink-0 rounded-md border px-1.5 py-0.5 text-[11px] font-medium"
          style={{ color: STATE_COLOR[stateKey], borderColor: `${STATE_COLOR[stateKey]}66`, background: `${STATE_COLOR[stateKey]}14` }}
        >
          {STATE_LABEL[stateKey]}
        </span>
      </div>
      <div className="mt-2.5 border-t pt-2">
        <RevisionLine source={source} />
        <div className="mt-1 flex justify-between text-xs">
          <span className="text-muted-foreground">分块</span>
          <span className="tabular-nums">{source.active_chunk_count.toLocaleString()}</span>
        </div>
      </div>
    </MorphCard>
  );
}

function SourceExpandedCard({
  source, jobs, siblingMaxChunks, onCheck, onIndex, onSavePolicy, checkPending, indexPending, policyPending,
}: {
  source: PublicCodeSource;
  jobs: PublicCodeJob[];
  /** `maxSourceChunkCount(sources)` — the shared scale this card's mini-bar
   * is drawn against, so card size compares against siblings instead of
   * stating a bare count. */
  siblingMaxChunks: number;
  onCheck: () => void;
  onIndex: () => void;
  onSavePolicy: (policy: PublicCodeUpdatePolicy, intervalMinutes: number | null) => void;
  checkPending: boolean;
  indexPending: boolean;
  policyPending: boolean;
}) {
  const { token } = theme.useToken();
  const sourceJobs = jobs.filter((j) => j.source_ids.includes(source.source_id)).slice(0, 8);
  const metaParts = [
    source.default_ref ? `ref: ${source.default_ref}` : null,
    source.proxy_profile ? `代理 ${source.proxy_profile}` : null,
    source.credential_profile ? `凭据 ${source.credential_profile}` : null,
  ].filter((p): p is string => p != null);
  return (
    // Extra right padding reserves the corner MorphGrid's own close button
    // occupies (position: absolute, top/right: 8 — outside this card's own
    // box) so this card's action buttons never sit under it.
    <div style={{ padding: '20px 44px 20px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <Text strong style={{ fontSize: 16 }} copyable>{source.source_id}</Text>
          <div><Tag color="blue">{source.domain}</Tag><Tag>{source.source_origin}</Tag></div>
        </div>
        <Space>
          <Tooltip title="检查上游 revision">
            <Button icon={<CloudDownloadOutlined />} loading={checkPending} onClick={onCheck} />
          </Tooltip>
          <Tooltip title="刷新镜像并索引候选 revision">
            <Button type="primary" icon={<SyncOutlined />} loading={indexPending} onClick={onIndex} />
          </Tooltip>
        </Space>
      </div>

      <div style={{ marginTop: 16 }}>
        <RevisionDots source={source} />
        <div style={{ marginTop: 6, display: 'grid', gap: 2, fontFamily: 'monospace', fontSize: 12 }}>
          <Text type="secondary">生效 {shortRevision(source.active_revision)}</Text>
          <Text type="secondary">镜像 {shortRevision(source.last_mirrored_revision)}</Text>
          <Text type={revisionGaps(source).mirrorRemoteGap ? 'warning' : 'secondary'}>上游 {shortRevision(source.last_remote_revision)}</Text>
        </div>
      </div>

      <div style={{ marginTop: 14 }}>
        {/* The bar's width already encodes size relative to sibling sources
            — a permanent "N 分块" caption under it just restates that as
            text. The number is still there, on hover, for the exact count. */}
        <Tooltip title={`${source.active_chunk_count.toLocaleString()} 分块`}>
          <div style={{ height: 6, borderRadius: 3, background: token.colorFillTertiary, overflow: 'hidden' }}>
            <div
              style={{
                height: '100%', borderRadius: 3, background: token.colorPrimary,
                width: `${siblingMaxChunks > 0 ? Math.max(2, (source.active_chunk_count / siblingMaxChunks) * 100) : 0}%`,
              }}
            />
          </div>
        </Tooltip>
      </div>

      <div style={{ marginTop: 14 }}>
        <Select
          size="small"
          value={source.update_policy === 'scheduled' ? String(source.update_interval_minutes ?? 1440) : 'manual'}
          style={{ width: 180 }}
          loading={policyPending}
          onChange={(value) => {
            if (value === 'manual') onSavePolicy('manual', null);
            else onSavePolicy('scheduled', Number(value));
          }}
          options={[
            { value: 'manual', label: '手动' },
            { value: '60', label: '定时 · 每小时' },
            { value: '1440', label: '定时 · 每天' },
            { value: '10080', label: '定时 · 每周' },
          ]}
        />
      </div>

      <div style={{ marginTop: 14, display: 'grid', gap: 2 }}>
        <Tooltip title={source.upstream_url}><Text ellipsis style={{ maxWidth: 420 }}>{source.upstream_url}</Text></Tooltip>
        {metaParts.length > 0 && (
          <Text type="secondary" style={{ fontSize: 12 }}>{metaParts.join(' · ')}</Text>
        )}
        <Text type={source.last_check_error ? 'danger' : 'secondary'} style={{ fontSize: 12 }}>
          {source.last_check_error || `上次检查 ${formatTime(source.last_checked_at)}`}
        </Text>
      </div>

      {sourceJobs.length > 0 && (
        <div style={{ marginTop: 16, borderTop: `1px solid ${token.colorBorderSecondary}`, paddingTop: 10, display: 'grid', gap: 6 }}>
          {sourceJobs.map((j) => (
            <div key={j.job_id} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                background: j.status === 'failed' ? token.colorError
                  : j.status === 'running' ? token.colorPrimary
                    : j.status === 'completed' ? token.colorSuccess : token.colorTextQuaternary,
              }} />
              <Text type="secondary" style={{ width: 90, flexShrink: 0 }}>{formatTime(j.created_at)}</Text>
              <Text style={{ flex: 1 }} ellipsis>{j.operation} · {j.phase}</Text>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function PublicCodeConsole() {
  const { message } = App.useApp();
  const [form] = Form.useForm<SourceFormValues>();
  const [sourceFilter, setSourceFilter] = useState('');
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [sourcePage, setSourcePage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchSourceIds, setSearchSourceIds] = useState<string[]>([]);
  const [searchDomains, setSearchDomains] = useState<string[]>([]);
  const [searchMode, setSearchMode] = useState<PublicCodeSearchMode>('hybrid');
  const [searchLanguage, setSearchLanguage] = useState<string>();
  const [searchTopK, setSearchTopK] = useState(20);
  const [searchRerank, setSearchRerank] = useState(false);

  const sourcesQuery = usePublicCodeSources();
  const jobsQuery = usePublicCodeJobs(50);
  const createSource = useCreatePublicCodeSource();
  const checkUpdates = useCheckPublicCodeUpdates();
  const indexSources = useIndexPublicCodeSources();
  const updatePolicy = useUpdatePublicCodePolicy();
  const search = useSearchPublicCode();

  const sources = sourcesQuery.data ?? EMPTY_SOURCES;
  const jobs = jobsQuery.data ?? EMPTY_JOBS;
  const maxChunks = useMemo(() => maxSourceChunkCount(sources), [sources]);
  const filteredSources = useMemo(
    () => filterPublicCodeSources(sources, sourceFilter),
    [sourceFilter, sources],
  );
  // Reset to page 1 whenever the filter text changes — adjusted during
  // render (React-endorsed pattern), not in an effect.
  const [lastSourceFilter, setLastSourceFilter] = useState(sourceFilter);
  if (sourceFilter !== lastSourceFilter) {
    setLastSourceFilter(sourceFilter);
    setSourcePage(1);
  }
  const pagedSources = useMemo(
    () => filteredSources.slice((sourcePage - 1) * SOURCE_PAGE_SIZE, sourcePage * SOURCE_PAGE_SIZE),
    [filteredSources, sourcePage],
  );
  const domains = useMemo(
    () => [...new Set(sources.map((source) => source.domain))].sort(),
    [sources],
  );

  const queueCheck = async (sourceIds = selectedSourceIds) => {
    try {
      const job = await checkUpdates.mutateAsync({ source_ids: sourceIds, domains: [] });
      message.success(`更新检查已入队: ${job.job_id}`);
    } catch (error) {
      message.error(apiError(error));
    }
  };

  const queueIndex = async (sourceIds = selectedSourceIds) => {
    try {
      const job = await indexSources.mutateAsync({ source_ids: sourceIds, domains: [] });
      message.success(`索引任务已入队: ${job.job_id}`);
    } catch (error) {
      message.error(apiError(error));
    }
  };

  const savePolicy = async (
    sourceId: string,
    updatePolicyValue: PublicCodeUpdatePolicy,
    updateIntervalMinutes: number | null,
  ) => {
    try {
      await updatePolicy.mutateAsync({
        sourceId,
        updatePolicy: updatePolicyValue,
        updateIntervalMinutes,
      });
      message.success(`${sourceId} 更新策略已保存`);
    } catch (error) {
      message.error(apiError(error));
    }
  };

  const submitSource = async (values: SourceFormValues) => {
    let languageOverrides: Record<string, string> = {};
    if (values.language_overrides?.trim()) {
      try {
        languageOverrides = JSON.parse(values.language_overrides) as Record<string, string>;
      } catch {
        message.error('语言覆盖必须是 JSON 对象，例如 {".cu":"cpp"}');
        return;
      }
    }
    const request: PublicCodeSourceCreateRequest = {
      ...values,
      proxy_profile: values.proxy_profile || null,
      credential_profile: values.credential_profile || null,
      language_overrides: languageOverrides,
      update_interval_minutes: values.update_policy === 'scheduled'
        ? values.update_interval_minutes
        : null,
    };
    try {
      const result = await createSource.mutateAsync(request);
      setDrawerOpen(false);
      form.resetFields();
      message.success(result.index_job
        ? `来源已注册，首次索引任务: ${result.index_job.job_id}`
        : '来源已注册');
    } catch (error) {
      message.error(apiError(error));
    }
  };

  const runSearch = async () => {
    if (!searchQuery.trim()) {
      message.warning('请输入检索内容');
      return;
    }
    try {
      await search.mutateAsync({
        query: searchQuery.trim(),
        source_ids: searchSourceIds,
        domains: searchDomains,
        mode: searchMode,
        language: searchLanguage || null,
        top_k: searchTopK,
        rerank: searchRerank,
      });
    } catch (error) {
      message.error(apiError(error));
    }
  };

  const jobColumns: TableColumnsType<PublicCodeJob> = [
    {
      title: '任务',
      dataIndex: 'job_id',
      width: 160,
      render: (jobId: string) => <Text code copyable>{jobId}</Text>,
    },
    { title: '操作类型', dataIndex: 'operation', width: 100 },
    {
      title: '状态',
      width: 120,
      render: (_, row) => <Space>{jobState(row.status)}<Text type="secondary">{row.phase}</Text></Space>,
    },
    {
      title: '进度',
      width: 220,
      render: (_, row) => (
        <Space orientation="vertical" size={0} style={{ width: '100%' }}>
          <Progress percent={Math.round(row.progress * 100)} size="small" status={row.status === 'failed' ? 'exception' : undefined} />
          <Text type="secondary">
            {row.completed_sources}/{row.total_sources} 源 · {row.files_done}/{row.files_total} 文件 · {row.chunks_done} 分块
          </Text>
        </Space>
      ),
    },
    {
      title: '当前',
      width: 260,
      render: (_, row) => (
        <Space orientation="vertical" size={0}>
          <Text>{row.current_source_id || '-'}</Text>
          <Tooltip title={row.current_path}><Text type="secondary" ellipsis style={{ maxWidth: 240 }}>{row.current_path || row.message || '-'}</Text></Tooltip>
        </Space>
      ),
    },
    {
      title: '错误',
      width: 260,
      render: (_, row) => {
        const errors = Object.entries(row.errors);
        return errors.length
          ? <Text type="danger">{errors.map(([sourceId, error]) => `${sourceId}: ${error}`).join('; ')}</Text>
          : <Text type="secondary">-</Text>;
      },
    },
    {
      title: '时间',
      width: 200,
      render: (_, row) => (
        <Space orientation="vertical" size={0}>
          <Text type="secondary">创建 {formatTime(row.created_at)}</Text>
          <Text type="secondary">结束 {formatTime(row.finished_at)}</Text>
        </Space>
      ),
    },
  ];

  const hitColumns: TableColumnsType<PublicCodeSearchHit> = [
    {
      title: '分数',
      dataIndex: 'score',
      width: 180,
      render: (_score: number, row) => <ScoreBar retrieval={row.retrieval_score} reranker={row.reranker_score} />,
    },
    {
      title: '来源',
      width: 180,
      render: (_, row) => <Space orientation="vertical" size={0}><Text strong>{row.source_id}</Text><Tag>{row.domain}</Tag></Space>,
    },
    {
      title: '路径',
      width: 430,
      render: (_, row) => (
        <Space orientation="vertical" size={0}>
          <Text code copyable>{row.path}:{row.start_line}-{row.end_line}</Text>
          <Text type="secondary">{row.label}</Text>
        </Space>
      ),
    },
    { title: '语言', dataIndex: 'language', width: 110 },
    {
      title: 'Revision',
      dataIndex: 'revision',
      width: 130,
      render: (revision: string) => <Text code>{shortRevision(revision)}</Text>,
    },
  ];

  const sourceTab = (
    <Space orientation="vertical" size={12} style={{ width: '100%' }}>
      <Space wrap style={{ justifyContent: 'space-between', width: '100%' }}>
        <Input
          allowClear
          prefix={<SearchOutlined />}
          placeholder="按来源 / 领域 / 路径 / URL / revision 过滤"
          value={sourceFilter}
          onChange={(event) => setSourceFilter(event.target.value)}
          style={{ width: 420 }}
        />
        <Space wrap>
          <Text type="secondary">已选 {selectedSourceIds.length}</Text>
          <Button
            icon={<CloudDownloadOutlined />}
            disabled={!selectedSourceIds.length}
            loading={checkUpdates.isPending}
            onClick={() => void queueCheck()}
          >检查更新</Button>
          <Button
            type="primary"
            icon={<SyncOutlined />}
            disabled={!selectedSourceIds.length}
            loading={indexSources.isPending}
            onClick={() => void queueIndex()}
          >索引所选</Button>
          <Button icon={<PlusOutlined />} onClick={() => setDrawerOpen(true)}>添加来源</Button>
          <Button icon={<ReloadOutlined />} onClick={() => void sourcesQuery.refetch()}>刷新</Button>
        </Space>
      </Space>
      <MorphGrid
        items={pagedSources}
        getKey={(s) => s.source_id}
        columns="repeat(auto-fill, minmax(200px, 1fr))"
        renderCompact={(s) => (
          <SourceCompactCard
            source={s}
            selected={selectedSourceIds.includes(s.source_id)}
            onToggleSelect={() => setSelectedSourceIds((cur) => (
              cur.includes(s.source_id) ? cur.filter((id) => id !== s.source_id) : [...cur, s.source_id]
            ))}
          />
        )}
        renderExpanded={(s) => (
          <SourceExpandedCard
            source={s}
            jobs={jobs}
            siblingMaxChunks={maxChunks}
            onCheck={() => void queueCheck([s.source_id])}
            onIndex={() => void queueIndex([s.source_id])}
            onSavePolicy={(policy, interval) => void savePolicy(s.source_id, policy, interval)}
            checkPending={checkUpdates.isPending}
            indexPending={indexSources.isPending}
            policyPending={updatePolicy.isPending}
          />
        )}
        emptyState={<div style={{ padding: '60px 0', textAlign: 'center' }}><Text type="secondary">没有匹配的来源</Text></div>}
      />
      {filteredSources.length > SOURCE_PAGE_SIZE && (
        <Pagination
          current={sourcePage}
          pageSize={SOURCE_PAGE_SIZE}
          total={filteredSources.length}
          onChange={setSourcePage}
          showSizeChanger={false}
        />
      )}
    </Space>
  );

  const searchTab = (
    <Space orientation="vertical" size={12} style={{ width: '100%' }}>
      <Card size="small">
        <Space orientation="vertical" size={10} style={{ width: '100%' }}>
          <Input.Search
            size="large"
            enterButton="检索"
            prefix={<CodeOutlined />}
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            onSearch={() => void runSearch()}
            loading={search.isPending}
            placeholder="描述实现、算法、符号或精确 token"
          />
          <Space wrap>
            <Select
              mode="multiple"
              allowClear
              maxTagCount="responsive"
              placeholder="全部领域"
              value={searchDomains}
              onChange={setSearchDomains}
              options={domains.map((domain) => ({ value: domain, label: domain }))}
              style={{ minWidth: 220 }}
            />
            <Select
              mode="multiple"
              allowClear
              maxTagCount="responsive"
              placeholder="所选领域下的全部来源"
              value={searchSourceIds}
              onChange={setSearchSourceIds}
              options={sources.map((source) => ({ value: source.source_id, label: source.source_id }))}
              style={{ minWidth: 280 }}
            />
            <Select<PublicCodeSearchMode>
              value={searchMode}
              onChange={setSearchMode}
              options={[
                { value: 'hybrid', label: '混合' },
                { value: 'dense', label: '语义' },
                { value: 'keyword', label: '关键词' },
              ]}
              style={{ width: 120 }}
            />
            <Input
              allowClear
              placeholder="语言"
              value={searchLanguage}
              onChange={(event) => setSearchLanguage(event.target.value || undefined)}
              style={{ width: 130 }}
            />
            <InputNumber min={1} max={50} value={searchTopK} onChange={(value) => setSearchTopK(value ?? 20)} addonBefore="Top K" />
            <Space><Switch checked={searchRerank} onChange={setSearchRerank} /><Text>重排</Text></Space>
          </Space>
        </Space>
      </Card>
      {search.error && <Alert type="error" showIcon message={apiError(search.error)} />}
      {search.data && (
        <Space orientation="vertical" size={4} style={{ width: '100%' }}>
          <Space size={6} wrap>
            <Tag color={search.data.effective_mode === search.data.requested_mode ? 'blue' : 'orange'}>
              {search.data.effective_mode}
            </Tag>
            <Tag>{search.data.candidate_count} 候选</Tag>
            {search.data.rerank_applied && <Tag color="purple">已重排 · {search.data.rerank_candidate_count}</Tag>}
          </Space>
          {search.data.notice && <Alert type="warning" showIcon message={search.data.notice} />}
        </Space>
      )}
      <Table
        rowKey={(row) => `${row.source_id}:${row.revision}:${row.path}:${row.start_line}`}
        size="small"
        loading={search.isPending}
        columns={hitColumns}
        dataSource={search.data?.hits ?? []}
        pagination={{ pageSize: 20 }}
        scroll={{ x: 1000 }}
        expandable={{
          expandedRowRender: (row) => (
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', overflowX: 'auto', fontSize: 12, lineHeight: 1.55 }}>
              {row.code || row.content}
            </pre>
          ),
        }}
        locale={{ emptyText: search.data ? '没有匹配的分块' : '在所选领域或来源中检索' }}
      />
    </Space>
  );

  const jobsTab = (
    <Space orientation="vertical" size={12} style={{ width: '100%' }}>
      <Space style={{ justifyContent: 'flex-end', width: '100%' }}>
        <Button icon={<ReloadOutlined />} onClick={() => void jobsQuery.refetch()}>刷新</Button>
      </Space>
      <Table
        rowKey="job_id"
        size="small"
        loading={jobsQuery.isLoading}
        columns={jobColumns}
        dataSource={jobs}
        pagination={{ pageSize: 25, showSizeChanger: true }}
        scroll={{ x: 1400 }}
      />
    </Space>
  );

  return (
    <Space orientation="vertical" size={16} style={{ width: '100%' }}>
      {sourcesQuery.error && <Alert type="error" showIcon message="来源列表加载失败" description={apiError(sourcesQuery.error)} />}

      <SourceStateBar sources={sources} />

      <Card styles={{ body: { paddingTop: 8 } }}>
        <Tabs
          items={[
            { key: 'sources', label: `来源 (${sources.length})`, children: sourceTab },
            { key: 'search', label: '领域 / 仓库检索', children: searchTab },
            { key: 'jobs', label: `任务 (${jobs.length})`, children: jobsTab },
          ]}
        />
      </Card>

      <Drawer
        title="添加代码来源"
        size={560}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        extra={<Button type="primary" loading={createSource.isPending} onClick={() => form.submit()}>注册</Button>}
      >
        <Form<SourceFormValues>
          form={form}
          layout="vertical"
          initialValues={{ default_ref: 'HEAD', update_policy: 'manual', index_now: true }}
          onFinish={(values) => void submitSource(values)}
        >
          <Row gutter={12}>
            <Col span={12}><Form.Item name="source_id" label="来源 ID" rules={[{ required: true }]}><Input placeholder="pion-webrtc" /></Form.Item></Col>
            <Col span={12}><Form.Item name="domain" label="领域" rules={[{ required: true }]}><Input placeholder="networking" /></Form.Item></Col>
          </Row>
          <Form.Item name="upstream_url" label="Git 上游地址" rules={[{ required: true }]}><Input placeholder="https://github.com/pion/webrtc.git" /></Form.Item>
          <Row gutter={12}>
            <Col span={16}><Form.Item name="relative_path" label="镜像相对路径" rules={[{ required: true }]}><Input placeholder="networking/pion-webrtc" /></Form.Item></Col>
            <Col span={8}><Form.Item name="default_ref" label="默认 ref" rules={[{ required: true }]}><Input /></Form.Item></Col>
          </Row>
          <Row gutter={12}>
            <Col span={12}><Form.Item name="proxy_profile" label="代理配置名"><Input placeholder="local-10808" /></Form.Item></Col>
            <Col span={12}><Form.Item name="credential_profile" label="凭据配置名"><Input placeholder="github-readonly" /></Form.Item></Col>
          </Row>
          <Form.Item name="language_overrides" label="语言覆盖 (JSON)">
            <Input.TextArea rows={3} placeholder={'{".cu":"cpp",".cuh":"cpp"}'} />
          </Form.Item>
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="update_policy" label="更新策略">
                <Select options={[{ value: 'manual', label: '手动' }, { value: 'scheduled', label: '定时' }]} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item noStyle shouldUpdate={(before, after) => before.update_policy !== after.update_policy}>
                {({ getFieldValue }) => getFieldValue('update_policy') === 'scheduled' ? (
                  <Form.Item name="update_interval_minutes" label="间隔（分钟）" rules={[{ required: true }]}>
                    <InputNumber min={1} max={43_200} style={{ width: '100%' }} />
                  </Form.Item>
                ) : null}
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="index_now" label="立即入队首次索引" valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Drawer>
    </Space>
  );
}
