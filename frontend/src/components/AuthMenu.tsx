import { useEffect } from 'react';
import { Button, Dropdown, Space, Tag, Avatar, Spin, App, theme } from 'antd';
import type { MenuProps } from 'antd';
import {
  UserOutlined, LoginOutlined, LogoutOutlined, WalletOutlined,
  ThunderboltOutlined, KeyOutlined, LinkOutlined,
} from '@ant-design/icons';
import { useMe, useLogout } from '../api/hooks/useAuth';
import { useAuthUI } from '../stores/authStore';
import AccountModal from './AccountModal';
import LoginModal from './LoginModal';

// One-shot toast for the social-login redirect outcomes the gateway encodes in
// the query string (?bind=ok|conflict, ?login=trust_low), then scrub the URL.
function useAuthRedirectToast() {
  const { message } = App.useApp();
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    const bind = p.get('bind');
    const loginErr = p.get('login');
    if (!bind && !loginErr) return;
    if (bind === 'ok') message.success('绑定成功，已并入当前账户');
    else if (bind === 'conflict') message.error('该身份已绑定到其它账户，无法重复绑定');
    else if (loginErr === 'trust_low') message.error('linux.do 等级不足，无法登录');
    p.delete('bind'); p.delete('login');
    const qs = p.toString();
    window.history.replaceState({}, '', window.location.pathname + (qs ? `?${qs}` : ''));
  }, [message]);
}

// Header auth widget (PUBLIC_MODE only). Anonymous → 登录 button; logged in →
// avatar dropdown with balance + 充值/密钥/登出. Owns the AccountModal mount.
const tierLabel: Record<string, { text: string; color: string }> = {
  paying: { text: '付费', color: 'blue' },
  internal: { text: '内部', color: 'green' },
  admin: { text: '管理员', color: 'gold' },
};

export default function AuthMenu() {
  const { token } = theme.useToken();
  const { data: me, isLoading } = useMe();
  const logout = useLogout();
  const openAccount = useAuthUI((s) => s.openAccount);
  const loginOpen = useAuthUI((s) => s.loginOpen);
  const openLogin = useAuthUI((s) => s.openLogin);
  const closeLogin = useAuthUI((s) => s.closeLogin);
  useAuthRedirectToast();

  if (isLoading) {
    return <Spin size="small" />;
  }

  if (!me) {
    return (
      <>
        <Button type="primary" icon={<LoginOutlined />} onClick={openLogin}>
          登录 / 注册
        </Button>
        <LoginModal open={loginOpen} onClose={closeLogin} />
        <AccountModal />
      </>
    );
  }

  const tl = tierLabel[me.tier] ?? { text: me.tier, color: 'default' };
  // Balance pill warms to amber/red as it runs low (1 credit = 1 search).
  const balanceColor = me.balance <= 0 ? 'error' : me.balance < 5 ? 'warning' : 'default';
  const items: MenuProps['items'] = [
    {
      key: 'balance',
      disabled: true,
      label: (
        <Space>
          <WalletOutlined />
          <span>余额 <strong style={{ color: token.colorPrimary }}>{me.balance}</strong> credits</span>
        </Space>
      ),
    },
    { type: 'divider' },
    { key: 'topup', icon: <ThunderboltOutlined />, label: '充值' },
    { key: 'keys', icon: <KeyOutlined />, label: 'API 密钥' },
    { key: 'bindings', icon: <LinkOutlined />, label: '账号绑定' },
    { type: 'divider' },
    { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', danger: true },
  ];

  const onClick: MenuProps['onClick'] = ({ key }) => {
    if (key === 'topup') openAccount('topup');
    else if (key === 'keys') openAccount('keys');
    else if (key === 'bindings') openAccount('bindings');
    else if (key === 'logout') logout.mutate();
  };

  return (
    <>
      <Dropdown menu={{ items, onClick }} placement="bottomRight" trigger={['click']}>
        <Space size={6} style={{ cursor: 'pointer' }}>
          <Avatar size="small" icon={<UserOutlined />} style={{ background: token.colorPrimary }} />
          <span style={{ fontSize: 13, color: token.colorText }}>{me.username}</span>
          <Tag color={tl.color} style={{ margin: 0, fontSize: 11, lineHeight: '16px' }}>{tl.text}</Tag>
          <Tag color={balanceColor} style={{ margin: 0, fontSize: 11, lineHeight: '16px' }}>
            <WalletOutlined /> {me.balance}
          </Tag>
        </Space>
      </Dropdown>
      <LoginModal open={loginOpen} onClose={closeLogin} />
      <AccountModal />
    </>
  );
}
