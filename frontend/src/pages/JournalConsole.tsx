import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  Card, Tabs, Segmented, Select, Input, Tag, Drawer, Tree,
  Empty, Button, Popconfirm, Space, Typography, Spin, Badge, Alert,
  Collapse, Descriptions, App as AntApp, theme,
} from 'antd';
import {
  BookOutlined, DatabaseOutlined, DeleteOutlined, FileTextOutlined,
  CloudServerOutlined, FolderOutlined, ReloadOutlined, ThunderboltOutlined,
  SyncOutlined, BulbOutlined, CodeOutlined,
} from '@ant-design/icons';
import type { DataNode } from 'antd/es/tree';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { rehypePlugins, remarkPlugins } from '../lib/markdown';

import {
  useConsoleOverview, useConsoleUsers, useJournalSearch, useJournalEntries,
  useJournalEntry, useDeleteJournalEntry, useInventory,
  useJournalPending, useReindexPending,
  usePlaybookBullets, useDeletePlaybookBullet,
  type JournalHit,
} from '../api/hooks/useAdminConsole';
import {
  buildInventoryTree, baseName, maxProjectChunkCount,
  type InventoryTreeNode, type FileSel, type ProjectSel,
} from '../components/inventoryTree';
import { sourceKindLabel, clientLabel } from './journalLabels';

const { Text, Paragraph } = Typography;

const ALL = '__all__';

const fade = {
  hidden: { opacity: 0, y: 8 },
  show: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i * 0.03 } }),
};

// Entries vectorized vs not IS a true two-way partition (indexed ⊆ entries),
// so it becomes one proportional bar. files/chunks/hosts/users are
// different-unit totals with no shared axis to bar-chart against each
// other — that kind of heterogeneous summary line doesn't belong on the
// page-level overview at all (design addendum), so it's gone entirely
// rather than moved into a caption here.
function OverviewBar() {
  const { data } = useConsoleOverview();
  const j = data?.journal;
  const total = j?.entries ?? 0;
  const indexed = j?.indexed ?? 0;
  const pct = total > 0 ? indexed / total : 0;
  return (
    <motion.div variants={fade} custom={0} initial="hidden" animate="show" style={{ marginBottom: 16 }}>
      <div className="flex items-center gap-3 text-xs">
        <span className="text-muted-foreground">已向量化的工作记忆</span>
        <div className="h-2 max-w-80 flex-1 overflow-hidden rounded-full bg-muted">
          <div className="h-full rounded-full bg-primary" style={{ width: `${pct * 100}%` }} />
        </div>
        <span className="tabular-nums">
          {data ? <><span className="font-semibold">{indexed.toLocaleString()}</span> / {total.toLocaleString()} 条</> : '加载中…'}
        </span>
      </div>
    </motion.div>
  );
}

// Replaces antd's deprecated `List`/`List.Item`/`List.Item.Meta` (removed in
// the next antd major version) with a plain row — same visual shape
// (optional leading avatar, title, description, trailing actions,
// bottom-border separated), used as a bare `.map()` instead of a `<List>`
// wrapper below.
function ListRow({
  avatar, title, description, actions,
}: {
  avatar?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode[];
}) {
  const { token } = theme.useToken();
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12,
      padding: '12px 0', borderBottom: `1px solid ${token.colorSplit}`,
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, minWidth: 0, flex: 1 }}>
        {avatar}
        <div style={{ minWidth: 0, flex: 1 }}>
          <div>{title}</div>
          {description && <div style={{ marginTop: 4 }}>{description}</div>}
        </div>
      </div>
      {actions && actions.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>{actions}</div>
      )}
    </div>
  );
}

function UserSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data } = useConsoleUsers();
  const options = [
    { value: ALL, label: '全部用户' },
    ...(data?.users ?? []).map((u) => ({
      value: u.owner_user_id,
      label: `${u.owner_user_id} · ${u.index_files} 文件${u.has_journal ? ' · 记忆' : ''}`,
    })),
  ];
  return (
    <Select
      value={value}
      onChange={onChange}
      options={options}
      style={{ minWidth: 220 }}
      showSearch
      optionFilterProp="label"
    />
  );
}

function JournalTab({ owner }: { owner: string }) {
  const { message } = AntApp.useApp();
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'keyword' | 'hybrid'>('keyword');
  const [openId, setOpenId] = useState<string | null>(null);

  const ownerId = owner === ALL ? null : owner;
  const search = useJournalSearch();
  // Browse fetches the newest N across all projects, then groups by project below.
  // Keep N at the route max so a single busy project can't crowd every other project
  // out of the (recency-ordered) window — otherwise the page looks like only one
  // project has any memory. Older entries beyond N are reachable via search.
  const browse = useJournalEntries({ owner_user_id: ownerId, limit: 200 }, !query);
  const detail = useJournalEntry(openId);
  const del = useDeleteJournalEntry();

  const runSearch = (q: string) => {
    setQuery(q);
    if (q.trim()) {
      search.mutate({ query: q, mode, owner_user_id: ownerId, include_chunks: true, chunk_limit: 5, top_k: 30 });
    }
  };

  // Unify search hits and browse rows into one list shape.
  const rows: JournalHit[] = query.trim()
    ? (search.data?.hits ?? [])
    : (browse.data?.entries ?? []).map((e) => ({ entry: e, score: 0, best_chunk: e.chunks[0] ?? null, chunks: e.chunks }));

  const loading = query.trim() ? search.isPending : browse.isLoading;

  const onDelete = (entryId: string) =>
    del.mutate({ entryId, reason: 'admin-console' }, {
      onSuccess: () => {
        message.success('已删除');
        if (query.trim()) runSearch(query);
      },
      onError: () => message.error('删除失败'),
    });

  // Group browse-mode rows by project so it reads as per-project memory, not a flat log.
  const grouped = useMemo(() => {
    const m = new Map<string, JournalHit[]>();
    for (const r of rows) {
      const arr = m.get(r.entry.project_slug) ?? [];
      arr.push(r);
      m.set(r.entry.project_slug, arr);
    }
    return [...m.entries()];
  }, [rows]);

  const renderRow = (hit: JournalHit, i: number) => {
    const e = hit.entry;
    return (
      <motion.div key={e.entry_id} custom={i} variants={fade} initial="hidden" animate="show">
        <ListRow
          actions={[
            <Button key="v" type="link" onClick={() => setOpenId(e.entry_id)}>详情</Button>,
            <Popconfirm key="d" title="删除该记忆？" onConfirm={() => onDelete(e.entry_id)}>
              <Button type="link" danger icon={<DeleteOutlined />} />
            </Popconfirm>,
          ]}
          title={
            <Space size={6} wrap>
              <a onClick={() => setOpenId(e.entry_id)}>{e.title || e.entry_id}</a>
              <Tag color="blue">{sourceKindLabel(e.source_kind)}</Tag>
              <Tag>{clientLabel(e.client)}</Tag>
              {!e.indexed && <Tag color="orange">未向量化</Tag>}
              {query.trim() && <Text type="secondary">score {hit.score.toFixed(3)}</Text>}
            </Space>
          }
          description={
            <Text type="secondary" ellipsis={{ tooltip: true }}>
              {(hit.best_chunk?.content || e.summary || e.body || '').slice(0, 200)}
            </Text>
          }
        />
      </motion.div>
    );
  };

  const browseShown = browse.data?.entries.length ?? 0;
  const browseLabel = `共 ${browseShown} 条`;

  return (
    <>
      <Space wrap style={{ marginBottom: 16 }}>
        <Input.Search
          placeholder="关键词或语义检索工作记忆"
          allowClear
          enterButton
          style={{ width: 420 }}
          onSearch={runSearch}
        />
        <Segmented
          value={mode}
          onChange={(v) => setMode(v as 'keyword' | 'hybrid')}
          options={[
            { label: '关键词', value: 'keyword' },
            { label: '语义', value: 'hybrid' },
          ]}
        />
        {query.trim() && (
          <Button icon={<ReloadOutlined />} onClick={() => setQuery('')}>返回浏览</Button>
        )}
        <Text type="secondary">
          {loading ? '' : query.trim()
            ? `命中 ${search.data?.total ?? 0} 条（${mode === 'keyword' ? 'FTS5 关键词' : '语义'}）`
            : browseLabel}
        </Text>
      </Space>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div>
      ) : rows.length === 0 ? (
        <Empty description="没有工作记忆" />
      ) : query.trim() ? (
        // search: flat, ranked by relevance
        <div>{rows.map((r, i) => renderRow(r, i))}</div>
      ) : (
        // browse: grouped by project so it's manageable, not a flat log
        <Collapse
          defaultActiveKey={grouped.map(([p]) => p)}
          items={grouped.map(([project, items]) => ({
            key: project,
            label: <Space><FolderOutlined /><b>{project}</b><Tag>{items.length}</Tag></Space>,
            children: <div>{items.map((r, i) => renderRow(r, i))}</div>,
          }))}
        />
      )}

      <Drawer
        open={!!openId}
        onClose={() => setOpenId(null)}
        size={720}
        title={detail.data?.title || openId}
        destroyOnClose
      >
        {detail.isLoading ? (
          <Spin />
        ) : detail.data ? (
          <>
            <Space wrap style={{ marginBottom: 12 }}>
              <Tag color="blue">{sourceKindLabel(detail.data.source_kind)}</Tag>
              <Tag>{clientLabel(detail.data.client)}</Tag>
              <Tag color="purple">{detail.data.project_slug}</Tag>
              {detail.data.tags.map((t) => <Tag key={t} color="geekblue">{t}</Tag>)}
            </Space>
            <Paragraph type="secondary" style={{ fontSize: 12 }}>
              {detail.data.entry_id} · {detail.data.chunk_count} 块 · {detail.data.created_at?.slice(0, 19)}
            </Paragraph>
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins}>
                {detail.data.body || '(无正文)'}
              </ReactMarkdown>
            </div>
          </>
        ) : (
          <Empty />
        )}
      </Drawer>
    </>
  );
}

function QueueTab() {
  const { message } = AntApp.useApp();
  const pending = useJournalPending(true, 4000);
  const reindex = useReindexPending();
  const rows = pending.data?.entries ?? [];
  const total = pending.data?.total ?? 0;

  const onReindex = () =>
    reindex.mutate(undefined, {
      onSuccess: (r) =>
        message.success(`重建完成：成功 ${r.reindexed} · 失败 ${r.failed} · 剩余 ${r.remaining}`),
      onError: () => message.error('重建失败'),
    });

  return (
    <>
      <Alert
        type={total ? 'warning' : 'success'}
        showIcon
        style={{ marginBottom: 16 }}
        message={
          total
            ? `${total} 条记忆已写入 SQL 但缺少向量（多因写入时 embed 服务不可用）`
            : 'SQL 与向量索引一致，重建队列为空'
        }
        description="队列每 4 秒自动刷新。点“立即重建”会重新嵌入并写回 Qdrant，使两者一致。"
      />
      <Space wrap style={{ marginBottom: 16 }}>
        <Button
          type="primary"
          icon={<ThunderboltOutlined />}
          loading={reindex.isPending}
          disabled={!total}
          onClick={onReindex}
        >
          立即重建 {total ? `(${total})` : ''}
        </Button>
        <Button icon={<ReloadOutlined />} onClick={() => pending.refetch()}>刷新</Button>
        <Text type="secondary">
          <SyncOutlined spin={pending.isFetching} /> 实时监控中（每 4s）
        </Text>
      </Space>

      {pending.isLoading ? (
        <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div>
      ) : rows.length === 0 ? (
        <Empty description="重建队列为空——所有记忆均已向量化" />
      ) : (
        <div>
          {rows.map((e, i) => (
            <motion.div key={e.entry_id} custom={i} variants={fade} initial="hidden" animate="show">
              <ListRow
                avatar={<Badge status="warning" />}
                title={
                  <Space size={6} wrap>
                    <Text>{e.title || e.entry_id}</Text>
                    <Tag color="purple">{e.project_slug}</Tag>
                    <Tag>{sourceKindLabel(e.source_kind)}</Tag>
                    <Tag color="orange">{e.chunk_count} 块待嵌入</Tag>
                  </Space>
                }
                description={
                  e.index_error
                    ? <Text type="danger" ellipsis={{ tooltip: true }}>失败原因：{e.index_error}</Text>
                    : <Text type="secondary">{e.created_at?.slice(0, 19)}</Text>
                }
              />
            </motion.div>
          ))}
        </div>
      )}
    </>
  );
}

// A project row's chunk count as a proportional mini-bar (design addendum
// §1.1) instead of "N chunks" text — width is chunkCount / the largest
// project in the whole tree, so a glance across the (already-indented) tree
// shows which projects are big without reading any numbers.
// Exported so the code search project TreeSelect (components/code/CodeScopePanel,
// F2-c) draws the same proportional mini-bar instead of inventing its own.
export function ProjectSizeBar({ chunkCount, max }: { chunkCount: number; max: number }) {
  const pct = max > 0 ? Math.min(1, chunkCount / max) : 0;
  return (
    <span style={{ display: 'inline-block', width: 48, height: 6, borderRadius: 3, background: 'var(--ant-color-fill-tertiary)', verticalAlign: 'middle', overflow: 'hidden' }}>
      <span style={{ display: 'block', width: `${pct * 100}%`, height: '100%', background: 'var(--ant-color-primary)' }} />
    </span>
  );
}

// JSX rendering layer over the shared pure tree (components/inventoryTree.ts,
// F1 §2.2) — the node shaping (incl. the fileMap/projectMap leaf lookups) is
// now common with the code search project TreeSelect; only the antd
// icon/Tag markup stays page-local.
function toDataNodes(nodes: InventoryTreeNode[], projectMax: number): DataNode[] {
  return nodes.map((n) => {
    switch (n.type) {
      case 'user':
        return {
          key: n.key, selectable: false, icon: <FolderOutlined />,
          title: <span><b>{n.label}</b> <Text type="secondary">· {n.fileCount} 文件 / {n.chunkCount} 块</Text></span>,
          children: n.children ? toDataNodes(n.children, projectMax) : undefined,
        };
      case 'host':
        return {
          key: n.key, selectable: false,
          title: <span><CloudServerOutlined /> {n.label} <Text type="secondary">· {n.fileCount} 文件</Text></span>,
          children: n.children ? toDataNodes(n.children, projectMax) : undefined,
        };
      case 'project':
        return {
          key: n.key, selectable: n.selectable,
          title: (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
              <DatabaseOutlined /> {n.label}
              <ProjectSizeBar chunkCount={n.chunkCount ?? 0} max={projectMax} />
              <Text type="secondary" style={{ fontSize: 12 }}>{n.rootDir}</Text>
            </span>
          ),
          children: n.children ? toDataNodes(n.children, projectMax) : undefined,
        };
      case 'dir':
        return {
          key: n.key, selectable: false,
          title: <span><FolderOutlined /> {n.label} <Text type="secondary">· {n.fileCount} 文件 / {n.chunkCount} 块</Text></span>,
          children: n.children ? toDataNodes(n.children, projectMax) : undefined,
        };
      default: // file
        return {
          key: n.key, isLeaf: true,
          title: (
            <span>
              {n.isDoc ? <FileTextOutlined /> : <CodeOutlined />} {n.label}
              {n.language && <Tag style={{ marginLeft: 6 }}>{n.language}</Tag>}
              <Text type="secondary"> · {n.chunkCount} 块{n.isDoc ? ' · 可查看内容' : ''}</Text>
            </span>
          ),
        };
    }
  });
}

function FileDetailDrawer({ sel, onClose }: { sel: FileSel | null; onClose: () => void }) {
  const f = sel?.file;
  const isDoc = !!f && (f.kind === 'doc' || f.collection === 'journal_chunks');
  const basename = f ? baseName(f.rel_path) : '';
  // For journal-backed docs the content lives in a journal entry (title = basename).
  const entries = useJournalEntries(
    { owner_user_id: sel?.owner ?? null, project_slug: sel?.project ?? null, q: basename, limit: 20 },
    !!sel && isDoc,
  );
  const matched = entries.data?.entries.find((e) => e.title === basename) ?? entries.data?.entries[0];
  const detail = useJournalEntry(isDoc ? matched?.entry_id ?? null : null);

  return (
    <Drawer open={!!sel} onClose={onClose} size={720} title={basename} destroyOnClose>
      {!f ? <Empty /> : (
        <>
          <Descriptions size="small" column={1} bordered style={{ marginBottom: 16 }}>
            <Descriptions.Item label="路径">{f.rel_path}</Descriptions.Item>
            <Descriptions.Item label="用户 / 机器">{sel.owner} · {sel.host}</Descriptions.Item>
            <Descriptions.Item label="项目">{sel.project}</Descriptions.Item>
            <Descriptions.Item label="类型">
              <Tag color={isDoc ? 'blue' : 'geekblue'}>{f.kind}</Tag>
              {f.language && <Tag>{f.language}</Tag>}
            </Descriptions.Item>
            <Descriptions.Item label="块数 / collection">{f.chunk_count} · {f.collection || '—'}</Descriptions.Item>
            {f.indexed_at && <Descriptions.Item label="索引时间">{f.indexed_at.slice(0, 19)}</Descriptions.Item>}
          </Descriptions>

          {!isDoc ? (
            <Empty description="代码内容请用 MCP search_code 检索" />
          ) : detail.isLoading || entries.isLoading ? (
            <div style={{ textAlign: 'center', padding: 32 }}><Spin /></div>
          ) : detail.data?.body ? (
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins}>{detail.data.body}</ReactMarkdown>
            </div>
          ) : (
            <Empty description="未找到对应的 journal 内容（可能已删除或尚未向量化）" />
          )}
        </>
      )}
    </Drawer>
  );
}

function InventoryTab({ owner }: { owner: string }) {
  const ownerId = owner === ALL ? null : owner;
  const summary = useInventory({ owner_user_id: ownerId });
  const [project, setProject] = useState<ProjectSel | null>(null);
  // Reset the drilled-in project when the owner filter changes — adjusted
  // during render (React-endorsed pattern) instead of in an effect.
  const [lastOwner, setLastOwner] = useState(owner);
  if (owner !== lastOwner) {
    setLastOwner(owner);
    setProject(null);
  }
  const detail = useInventory({
    owner_user_id: project?.owner,
    host: project?.host,
    project: project?.project,
    detail: true,
  }, !!project);
  const activeData = project ? detail.data : summary.data;
  const { nodes: pureNodes, fileMap, projectMap } = useMemo(
    () => buildInventoryTree(activeData), [activeData],
  );
  const projectMax = useMemo(() => maxProjectChunkCount(pureNodes), [pureNodes]);
  const treeData = useMemo(() => toDataNodes(pureNodes, projectMax), [pureNodes, projectMax]);
  const [sel, setSel] = useState<FileSel | null>(null);
  const totals = summary.data?.totals ?? {};
  const isLoading = project ? detail.isLoading : summary.isLoading;

  return (
    <>
      <Space wrap style={{ marginBottom: 12 }}>
        {project && <Button size="small" onClick={() => setProject(null)}>返回项目汇总</Button>}
        <Text type="secondary">
          {totals.users ?? 0} 用户 · {totals.hosts ?? 0} 机器 · {totals.projects ?? 0} 项目 ·{' '}
          {totals.files ?? 0} 文件 · {totals.chunks ?? 0} 块 ·{' '}
          {project ? `项目 ${project.project} · 点击文件查看详情` : '点击项目后按需加载文件明细'}
        </Text>
      </Space>
      {isLoading ? (
        <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div>
      ) : treeData.length === 0 ? (
        <Empty description="暂无索引清单（客户端 index_code / journal_index_dir 后上报）" />
      ) : (
        <motion.div variants={fade} custom={0} initial="hidden" animate="show">
          <Tree
            showIcon
            treeData={treeData}
            defaultExpandedKeys={treeData.slice(0, 1).map((n) => n.key as string)}
            onSelect={(keys) => {
              const k = keys[0] as string;
              if (k && fileMap.has(k)) setSel(fileMap.get(k)!);
              else if (k && projectMap.has(k)) setProject(projectMap.get(k)!);
            }}
            style={{ background: 'transparent' }}
          />
        </motion.div>
      )}
      <FileDetailDrawer sel={sel} onClose={() => setSel(null)} />
    </>
  );
}

const SECTION_COLOR: Record<string, string> = {
  patterns: 'blue', best_practices: 'green', insights: 'geekblue', failure_mode: 'red',
};

function PlaybookTab({ owner }: { owner: string }) {
  const { message } = AntApp.useApp();
  const ownerId = owner === ALL ? null : owner;
  const { data, isLoading } = usePlaybookBullets({ owner_user_id: ownerId });
  const del = useDeletePlaybookBullet();
  const bullets = data?.bullets ?? [];

  // group by project so it reads as a curated playbook, not a flat log
  const byProject = useMemo(() => {
    const m = new Map<string, typeof bullets>();
    for (const b of bullets) {
      const arr = m.get(b.project_slug) ?? [];
      arr.push(b);
      m.set(b.project_slug, arr);
    }
    return [...m.entries()];
  }, [bullets]);

  const onDelete = (id: number, bulletOwner: string) =>
    del.mutate({ id, owner_user_id: bulletOwner }, {
      onSuccess: () => message.success('已删除'),
      onError: () => message.error('删除失败'),
    });

  if (isLoading) return <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div>;
  if (bullets.length === 0)
    return <Empty description="ACE playbook 为空 —— agent 用 ace_smart_reflect 沉淀经验后出现在这里" />;

  return (
    <Collapse
      defaultActiveKey={byProject.map(([p]) => p)}
      items={byProject.map(([project, items]) => ({
        key: project,
        label: <Space><BulbOutlined /><b>{project}</b><Tag>{items.length} bullets</Tag></Space>,
        children: (
          <div>
            {items.map((b, i) => (
              <motion.div key={b.id} custom={i} variants={fade} initial="hidden" animate="show">
                <ListRow
                  actions={[
                    <Popconfirm key="d" title="删除该 bullet？" onConfirm={() => onDelete(b.id, b.owner_user_id)}>
                      <Button type="link" danger icon={<DeleteOutlined />} />
                    </Popconfirm>,
                  ]}
                  title={
                    <Space size={6} wrap>
                      <Tag color={SECTION_COLOR[b.section] || 'default'}>{b.section}</Tag>
                      {b.domain && <Tag color="purple">{b.domain}</Tag>}
                      <Text type="secondary">+{b.helpful}/-{b.harmful}</Text>
                    </Space>
                  }
                  description={<Text>{b.content}</Text>}
                />
              </motion.div>
            ))}
          </div>
        ),
      }))}
    />
  );
}

export default function JournalConsole() {
  const [owner, setOwner] = useState<string>(ALL);
  const pending = useJournalPending(true, 4000); // drives the live tab badge

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', marginBottom: 16 }}>
        <Space>
          <Text type="secondary">用户</Text>
          <UserSelect value={owner} onChange={setOwner} />
        </Space>
      </div>

      <OverviewBar />

      <Card styles={{ body: { paddingTop: 8 } }}>
        <Tabs
          items={[
            { key: 'journal', label: <span><BookOutlined /> 工作记忆</span>, children: <JournalTab owner={owner} /> },
            { key: 'inventory', label: <span><DatabaseOutlined /> 索引清单</span>, children: <InventoryTab owner={owner} /> },
            { key: 'playbook', label: <span><BulbOutlined /> ACE 经验手册</span>, children: <PlaybookTab owner={owner} /> },
            {
              key: 'queue',
              label: (
                <Badge count={pending.data?.total ?? 0} size="small" offset={[8, -2]}>
                  <span><SyncOutlined /> 重建队列</span>
                </Badge>
              ),
              children: <QueueTab />,
            },
          ]}
        />
      </Card>
    </motion.div>
  );
}
