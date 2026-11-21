import { Modal, Button, Typography, Space, theme } from 'antd';
import { KeyOutlined, GoogleOutlined } from '@ant-design/icons';
import { login, loginLinuxDo, loginGoogle, useAuthProviders } from '../api/hooks/useAuth';

/**
 * Login / register provider chooser. The gateway supports three login methods,
 * all converging on one account model (账号绑定: bind more later, shared balance):
 *   - 访问密钥 (CDK)  → cf-oidc Worker
 *   - linux.do Connect → connect.linux.do
 *   - Google           → accounts.google.com
 * Only methods the gateway has enabled (GET /auth/providers) are shown.
 * Each choice is a full-page redirect, so no in-app callback handling is needed.
 */
export default function LoginModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { token } = theme.useToken();
  const { data: providers } = useAuthProviders();

  const ldEnabled = providers?.linuxdo ?? true;
  const googleEnabled = providers?.google ?? false;

  const linuxdoIcon = (
    <span style={{ fontWeight: 800, fontSize: 16, lineHeight: 1 }}>L</span>
  );

  return (
    <Modal open={open} onCancel={onClose} footer={null} width={400} title="登录 / 注册" centered>
      <Space direction="vertical" size="middle" style={{ width: '100%', marginTop: 4 }}>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          选择一种方式登录；首次登录将自动创建账户。登录后可在「账户 → 账号绑定」里绑定其它登录方式，
          多种身份共享同一余额。
        </Typography.Text>

        {ldEnabled && (
          <Button
            block
            size="large"
            icon={linuxdoIcon}
            onClick={loginLinuxDo}
            style={{ height: 48, borderColor: '#f0a020', color: '#d98800' }}
          >
            使用 linux.do 登录
          </Button>
        )}

        {googleEnabled && (
          <Button block size="large" icon={<GoogleOutlined />} onClick={loginGoogle} style={{ height: 48 }}>
            使用 Google 登录
          </Button>
        )}

        <Button
          block
          size="large"
          type="primary"
          icon={<KeyOutlined />}
          onClick={login}
          style={{ height: 48, background: token.colorPrimary }}
        >
          使用访问密钥登录
        </Button>
      </Space>
    </Modal>
  );
}
