import { useState } from 'react';
import type { ReactNode } from 'react';
import {
  Modal, Tabs, Card, Row, Col, Button, Typography, Space, Table, Tag,
  Popconfirm, Input, Badge, App, theme,
} from 'antd';
import {
  ThunderboltOutlined, KeyOutlined, PlusOutlined, CopyOutlined, WalletOutlined,
  GoogleOutlined, LinkOutlined, DisconnectOutlined, CheckCircleFilled,
} from '@ant-design/icons';
import { useAuthUI } from '../stores/authStore';
import {
  useMe, useApiKeys, useCreateKey, useRevokeKey, useTopup,
  useIdentities, useUnlinkIdentity, useAuthProviders, bindIdentity,
} from '../api/hooks/useAuth';
import type { ApiKeyInfo, ApiKeyCreated, Identity } from '../types';
import { formatDateTime } from '../constants';
import { copyText } from '../utils/clipboard';

// How each login provider is shown in the binding list.
const PROVIDER_META: Record<string, { label: string; icon: ReactNode; color: string }> = {
  cdk: { label: '访问密钥', icon: <KeyOutlined />, color: '#4F46E5' },
  linuxdo: { label: 'linux.do', icon: <span style={{ fontWeight: 800 }}>L</span>, color: '#f0a020' },
  google: { label: 'Google', icon: <GoogleOutlined />, color: '#ea4335' },
};

// 1 metered call (search / match-doc) = this many credits. Mirrors the gateway's
// SEARCH_COST default; used only for the informational "≈ N 次" hints.
const CREDITS_PER_CALL = 1;

// Credit packs (credits / price RMB). The middle pack is flagged 最划算.
const PACKS = [
  { credits: 100, price: '5.00' },
  { credits: 500, price: '20.00', best: true },
  { credits: 2000, price: '68.00' },
];

const GRADIENT = 'linear-gradient(135deg, #6366f1 0%, #4F46E5 55%, #4338ca 100%)';

function BalanceHero({ balance }: { balance: number }) {
  const low = balance < 5 * CREDITS_PER_CALL;
  return (
    <div
      style={{
        background: GRADIENT,
        borderRadius: 14,
        padding: '18px 22px',
        color: '#fff',
        boxShadow: '0 8px 24px rgba(79,70,229,0.28)',
      }}
    >
      <Row align="middle" justify="space-between">
        <Col>
          <Space size={6} style={{ opacity: 0.85, fontSize: 12 }}>
            <WalletOutlined /> 当前余额
          </Space>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 2 }}>
            <span style={{ fontSize: 34, fontWeight: 800, lineHeight: 1 }}>{balance}</span>
            <span style={{ fontSize: 13, opacity: 0.85 }}>credits</span>
          </div>
          <div style={{ fontSize: 12, opacity: 0.8, marginTop: 6 }}>
            约可检索 <strong>{Math.floor(balance / CREDITS_PER_CALL)}</strong> 次 · 1 credit = 1 次检索/匹配
          </div>
        </Col>
        <Col>
          <ThunderboltOutlined style={{ fontSize: 48, opacity: 0.22 }} />
        </Col>
      </Row>
      {low && (
        <div style={{ marginTop: 10, fontSize: 12, background: 'rgba(255,255,255,0.16)', borderRadius: 8, padding: '6px 10px' }}>
          余额偏低，建议充值以免检索中断。
        </div>
      )}
    </div>
  );
}

function PackCard({ pack, loading, onPay }: {
  pack: typeof PACKS[number];
  loading: boolean;
  onPay: () => void;
}) {
  const { token } = theme.useToken();
  const unit = (Number(pack.price) / pack.credits).toFixed(3);
  const body = (
    <Card
      size="small"
      hoverable
      style={{
        borderRadius: 12,
        textAlign: 'center',
        borderColor: pack.best ? token.colorPrimary : token.colorBorderSecondary,
        borderWidth: pack.best ? 2 : 1,
        height: '100%',
      }}
      styles={{ body: { padding: '18px 8px' } }}
    >
      <div style={{ fontSize: 24, fontWeight: 800, color: token.colorPrimary }}>{pack.credits}</div>
      <Typography.Text type="secondary" style={{ fontSize: 12 }}>credits</Typography.Text>
      <div style={{ fontSize: 11, color: token.colorTextTertiary, marginTop: 2 }}>
        ¥{unit} / 次
      </div>
      <Button type={pack.best ? 'primary' : 'default'} block style={{ marginTop: 12 }} loading={loading} onClick={onPay}>
        ¥{pack.price}
      </Button>
    </Card>
  );
  return pack.best
    ? <Badge.Ribbon text="最划算" color="#4F46E5">{body}</Badge.Ribbon>
    : body;
}

function TopupPanel() {
  const { message } = App.useApp();
  const { data: me } = useMe();
  const topup = useTopup();
  const [picked, setPicked] = useState<number>(0);

  const onPay = (credits: number, price: string) => {
    setPicked(credits);
    topup.mutate(
      { credits, price },
      {
        onSuccess: (r) => {
          // Mock provider returns a placeholder pay_url; a real epay/ldc URL opens
          // the cashier. New tab so the SPA session is preserved.
          window.open(r.pay_url, '_blank', 'noopener');
          message.info('已创建订单，请在新窗口完成支付；支付成功后余额自动到账');
        },
        onError: () => message.error('下单失败，请稍后重试'),
        onSettled: () => setPicked(0),
      },
    );
  };

  return (
    <Space orientation="vertical" size="large" style={{ width: '100%' }}>
      <BalanceHero balance={me?.balance ?? 0} />
      <div>
        <Typography.Text strong style={{ fontSize: 13 }}>选择充值档位</Typography.Text>
        <Row gutter={12} style={{ marginTop: 10 }}>
          {PACKS.map((pk) => (
            <Col span={8} key={pk.credits}>
              <PackCard pack={pk} loading={topup.isPending && picked === pk.credits} onPay={() => onPay(pk.credits, pk.price)} />
            </Col>
          ))}
        </Row>
      </div>
      <Typography.Text type="secondary" style={{ fontSize: 11 }}>
        支付完成后余额由支付回调自动入账（幂等，重复回调不会重复计费）。
      </Typography.Text>
    </Space>
  );
}

function KeysPanel() {
  const { message } = App.useApp();
  const { data: keys = [], isLoading } = useApiKeys();
  const createKey = useCreateKey();
  const revokeKey = useRevokeKey();
  const [freshKey, setFreshKey] = useState<ApiKeyCreated | null>(null);

  const onCreate = () => {
    createKey.mutate(undefined, {
      onSuccess: (k) => setFreshKey(k),
      onError: () => message.error('创建失败'),
    });
  };

  const columns = [
    {
      title: '密钥',
      key: 'key',
      render: (_: unknown, r: ApiKeyInfo) => (
        <Typography.Text code style={{ fontSize: 12 }}>
          {r.prefix}…{r.last4}
        </Typography.Text>
      ),
    },
    {
      title: '状态',
      key: 'status',
      render: (_: unknown, r: ApiKeyInfo) =>
        r.revoked ? <Tag>已吊销</Tag> : <Tag color="success">有效</Tag>,
    },
    {
      title: '最近使用',
      dataIndex: 'last_used_at',
      key: 'last_used_at',
      render: (v: string | null) => (
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>{formatDateTime(v)}</Typography.Text>
      ),
    },
    {
      title: '',
      key: 'action',
      render: (_: unknown, r: ApiKeyInfo) =>
        r.revoked ? null : (
          <Popconfirm title="吊销后无法恢复，确认？" onConfirm={() => revokeKey.mutate(r.id)} okText="吊销" cancelText="取消">
            <Button size="small" danger type="link">吊销</Button>
          </Popconfirm>
        ),
    },
  ];

  return (
    <Space orientation="vertical" size="middle" style={{ width: '100%' }}>
      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
        <Typography.Text code>Authorization: Bearer sk-…</Typography.Text>
      </Typography.Text>

      {freshKey && (
        <Card size="small" style={{ borderRadius: 10, borderColor: '#f59e0b', background: 'rgba(245,158,11,0.06)' }}>
          <Typography.Text strong style={{ color: '#d97706' }}>
            请立即复制并妥善保存 —— 此密钥仅显示这一次
          </Typography.Text>
          <Space.Compact style={{ marginTop: 8, display: 'flex' }}>
            <Input readOnly value={freshKey.key} style={{ fontFamily: 'monospace', fontSize: 12 }} />
            <Button
              type="primary"
              icon={<CopyOutlined />}
              onClick={() => copyText(freshKey.key)
                .then(() => message.success('已复制'))
                .catch(() => message.error('复制失败'))}
            >
              复制
            </Button>
          </Space.Compact>
        </Card>
      )}

      <Button type="primary" icon={<PlusOutlined />} onClick={onCreate} loading={createKey.isPending}>
        新建 API 密钥
      </Button>

      <Table
        rowKey="id"
        size="small"
        loading={isLoading}
        columns={columns}
        dataSource={keys}
        pagination={false}
        locale={{ emptyText: '暂无密钥' }}
      />
    </Space>
  );
}

function BindingsPanel() {
  const { message } = App.useApp();
  const { data: idents = [], isLoading } = useIdentities();
  const { data: providers } = useAuthProviders();
  const unlink = useUnlinkIdentity();

  const bound = new Set(idents.map((i) => i.provider));
  const canUnlink = idents.length > 1;

  // Providers that are enabled server-side AND not yet bound → offer to bind.
  const bindable: ('linuxdo' | 'google')[] = [];
  if ((providers?.linuxdo ?? true) && !bound.has('linuxdo')) bindable.push('linuxdo');
  if ((providers?.google ?? false) && !bound.has('google')) bindable.push('google');

  const onUnlink = (id: number) =>
    unlink.mutate(id, {
      onSuccess: () => message.success('已解绑'),
      onError: () => message.error('无法解绑（至少保留一种登录方式）'),
    });

  return (
    <Space orientation="vertical" size="middle" style={{ width: '100%' }}>
      <Table
        rowKey="id"
        size="small"
        loading={isLoading}
        pagination={false}
        locale={{ emptyText: '暂无绑定' }}
        dataSource={idents}
        columns={[
          {
            title: '登录方式',
            key: 'provider',
            render: (_: unknown, r: Identity) => {
              const m = PROVIDER_META[r.provider] ?? { label: r.provider, icon: <LinkOutlined />, color: 'default' };
              return (
                <Space>
                  <span style={{ color: m.color }}>{m.icon}</span>
                  <span>{m.label}</span>
                  <CheckCircleFilled style={{ color: '#52c41a', fontSize: 12 }} />
                </Space>
              );
            },
          },
          {
            title: '',
            key: 'action',
            align: 'right' as const,
            render: (_: unknown, r: Identity) =>
              canUnlink ? (
                <Popconfirm
                  title="解绑后将无法用此方式登录，确认？"
                  onConfirm={() => onUnlink(r.id)}
                  okText="解绑"
                  cancelText="取消"
                >
                  <Button size="small" type="link" danger icon={<DisconnectOutlined />}>解绑</Button>
                </Popconfirm>
              ) : (
                <Tag>唯一登录方式</Tag>
              ),
          },
        ]}
      />

      {bindable.length > 0 && (
        <div>
          <Typography.Text strong style={{ fontSize: 13 }}>绑定新的登录方式</Typography.Text>
          <Space style={{ marginTop: 10 }} wrap>
            {bindable.includes('linuxdo') && (
              <Button icon={<span style={{ fontWeight: 800 }}>L</span>} onClick={() => bindIdentity('linuxdo')}
                style={{ borderColor: '#f0a020', color: '#d98800' }}>
                绑定 linux.do
              </Button>
            )}
            {bindable.includes('google') && (
              <Button icon={<GoogleOutlined />} onClick={() => bindIdentity('google')}>
                绑定 Google
              </Button>
            )}
          </Space>
        </div>
      )}
    </Space>
  );
}

export default function AccountModal() {
  const open = useAuthUI((s) => s.accountOpen);
  const tab = useAuthUI((s) => s.accountTab);
  const close = useAuthUI((s) => s.closeAccount);
  const openAccount = useAuthUI((s) => s.openAccount);

  return (
    <Modal
      open={open}
      onCancel={close}
      footer={null}
      width={560}
      title="账户"
      destroyOnHidden
    >
      <Tabs
        activeKey={tab}
        onChange={(k) => openAccount(k as 'topup' | 'keys' | 'bindings')}
        items={[
          { key: 'topup', label: <span><ThunderboltOutlined /> 充值</span>, children: <TopupPanel /> },
          { key: 'keys', label: <span><KeyOutlined /> API 密钥</span>, children: <KeysPanel /> },
          { key: 'bindings', label: <span><LinkOutlined /> 账号绑定</span>, children: <BindingsPanel /> },
        ]}
      />
    </Modal>
  );
}
