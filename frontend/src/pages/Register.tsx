import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Input, Button, Typography, Space, App, theme } from 'antd';
import { UserOutlined, ArrowRightOutlined } from '@ant-design/icons';
import axios from 'axios';
import { useRegister } from '../api/hooks/useAuth';

/**
 * First-login username claim. The gateway's `/auth/callback` redirects a brand-new
 * OIDC subject here (a `pending` registration sits in the session). We POST the
 * chosen username to `/auth/register`; on success the session becomes a real
 * account and we land on Search.
 *
 * If there's no pending registration (user hit /register directly, or already
 * registered), the gateway returns 400 → we bounce home.
 */
export default function Register() {
  const { token } = theme.useToken();
  const { message } = App.useApp();
  const navigate = useNavigate();
  const register = useRegister();
  const [username, setUsername] = useState('');

  const onSubmit = () => {
    const name = username.trim();
    if (name.length < 2) {
      message.warning('用户名至少 2 个字符');
      return;
    }
    register.mutate(name, {
      onSuccess: () => {
        message.success('注册成功，欢迎使用');
        navigate('/', { replace: true });
      },
      onError: (e) => {
        const status = axios.isAxiosError(e) ? e.response?.status : undefined;
        const detail = axios.isAxiosError(e)
          ? (e.response?.data as { detail?: string } | undefined)?.detail
          : undefined;
        if (status === 409) {
          message.error(detail === 'username taken' ? '用户名已被占用' : '该账户已注册，请直接登录');
          if (detail !== 'username taken') navigate('/', { replace: true });
        } else if (status === 400) {
          message.info('没有待完成的注册，已为你跳转首页');
          navigate('/', { replace: true });
        } else {
          message.error('注册失败，请重试');
        }
      },
    });
  };

  return (
    <div
      style={{
        minHeight: 'calc(100vh - 120px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        style={{
          width: 440,
          maxWidth: '100%',
          borderRadius: 18,
          overflow: 'hidden',
          boxShadow: '0 20px 60px rgba(79,70,229,0.18)',
          background: token.colorBgContainer,
          border: `1px solid ${token.colorBorderSecondary}`,
        }}
      >
        {/* Branded gradient header */}
        <div
          style={{
            background: 'linear-gradient(135deg, #6366f1 0%, #4F46E5 55%, #4338ca 100%)',
            padding: '32px 32px 28px',
            textAlign: 'center',
            color: '#fff',
          }}
        >
          <div
            style={{
              width: 56, height: 56, borderRadius: 14, margin: '0 auto 14px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 28, fontWeight: 800,
              background: 'rgba(255,255,255,0.18)', backdropFilter: 'blur(4px)',
            }}
          >
            R
          </div>
          <Typography.Title level={3} style={{ color: '#fff', margin: 0 }}>欢迎加入 Retrieval</Typography.Title>
          <Typography.Text style={{ color: 'rgba(255,255,255,0.85)', fontSize: 13 }}>
            首次登录 —— 为你的账户设置一个用户名
          </Typography.Text>
        </div>

        {/* Form */}
        <div style={{ padding: 32 }}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Input
              size="large"
              prefix={<UserOutlined style={{ color: token.colorTextTertiary }} />}
              placeholder="设置用户名"
              value={username}
              maxLength={32}
              onChange={(e) => setUsername(e.target.value)}
              onPressEnter={onSubmit}
              autoFocus
            />
            <Button
              type="primary"
              size="large"
              block
              loading={register.isPending}
              onClick={onSubmit}
              icon={<ArrowRightOutlined />}
              iconPosition="end"
            >
              完成注册，开始检索
            </Button>
            <Typography.Text type="secondary" style={{ fontSize: 12, textAlign: 'center', display: 'block' }}>
              用户名将作为你的账户标识，注册后不可更改。
            </Typography.Text>
          </Space>
        </div>
      </div>
    </div>
  );
}
