import { useEffect, type ReactNode } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider, App as AntApp, Spin, theme as antTheme } from 'antd';
import dayjs from 'dayjs';
import 'dayjs/locale/zh-cn';

import { useAppStore } from './stores/appStore';
import { PUBLIC_MODE } from './constants';
import AppLayout from './components/Layout';
import Search from './pages/Search';
import PaperList from './pages/PaperList';
import PaperDetail from './pages/PaperDetail';
import Documents from './pages/Documents';
import Tasks from './pages/Tasks';
import MatchDoc from './pages/MatchDoc';
import Register from './pages/Register';
import Settings from './pages/Settings';
import { useConsoleAccessState } from './api/hooks/useAdminConsole';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Toaster } from '@/components/ui/sonner';

dayjs.locale('zh-cn');

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

const FONT =
  "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Noto Sans SC', sans-serif";

/** Admin console route: rendered only for admins (internal build, or tier=admin
    on public). Non-admins are bounced to search — the backend enforces auth too. */
function ConsoleGate({ children }: { children: ReactNode }) {
  const access = useConsoleAccessState();
  if (access === 'loading') {
    return (
      <div style={{ minHeight: 240, display: 'grid', placeItems: 'center' }}>
        <Spin size="large" />
      </div>
    );
  }
  return access === 'allowed' ? children : <Navigate to="/search" replace />;
}

export default function App() {
  const darkMode = useAppStore((s) => s.darkMode);
  // shadcn/ui reads `.dark` on <html>; antd reads the same flag below.
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider
        theme={{
          algorithm: darkMode ? antTheme.darkAlgorithm : antTheme.defaultAlgorithm,
          // Monochrome foundation + ONE functional accent (used only for active/CTA/focus,
          // never decoration). Neutral canvas — no navy/purple mass. High contrast text.
          token: {
            colorPrimary: '#2563EB',
            colorInfo: '#2563EB',
            borderRadius: 8,
            fontFamily: FONT,
            colorBgLayout: darkMode ? '#09090b' : '#fafafa',
            colorBgContainer: darkMode ? '#18181b' : '#ffffff',
            colorBgElevated: darkMode ? '#18181b' : '#ffffff',
            colorTextHeading: darkMode ? '#fafafa' : '#18181b',
          },
          components: {
            Card: { borderRadiusLG: 12 },
            Button: { primaryShadow: 'none', fontWeight: 500 },
            Table: { borderRadius: 8 },
            Menu: {
              darkItemBg: 'transparent',
              darkSubMenuItemBg: 'transparent',
              activeBarBorderWidth: 0,
              horizontalItemSelectedColor: '#2563EB',
              itemColor: darkMode ? 'rgba(255,255,255,0.65)' : 'rgba(0,0,0,0.65)',
            },
          },
        }}
      >
        <AntApp>
          <TooltipProvider delayDuration={200}>
          <BrowserRouter>
            <Routes>
              <Route element={<AppLayout />}>
                {/* 首页 = 检索页 for both internal and public (D1) — Dashboard is
                    gone; its numbers now live in the status capsule. */}
                <Route index element={<Navigate to="/search" replace />} />
                <Route path="search" element={<Search />} />
                <Route path="match-doc" element={<MatchDoc />} />
                {/* F2-c: code search is now /search?mode=code — CodeSearch.tsx is
                    gone, this is just an old-link redirect. */}
                <Route path="code" element={<Navigate to="/search?mode=code" replace />} />
                <Route path="papers" element={<PaperList />} />
                <Route path="papers/:paperId" element={<PaperDetail />} />
                {/* 文档库 is the public browse surface (read-only in public mode —
                    Documents.tsx hides upload/edit/delete/crawl). Internal keeps full;
                    those controls moved into /settings?tab=data (F1-e). */}
                <Route path="documents" element={<Documents />} />
                {/* Settings: gated by ConsoleGate (internal build or tier=admin).
                    /api/admin/* is routed to the gateway (require_admin) on public too.
                    Folds in the former 数据管理/公共代码库/记忆 pages (F1-e). */}
                <Route path="settings" element={<ConsoleGate><Settings /></ConsoleGate>} />
                {/* Old direct links redirect into the matching settings tab. */}
                <Route path="journal" element={<Navigate to="/settings?tab=journal" replace />} />
                <Route path="public-code" element={<Navigate to="/settings?tab=public-code" replace />} />
                {!PUBLIC_MODE && <Route path="tasks" element={<Tasks />} />}
                {/* Public: first-OIDC-login username claim (gateway redirects here). */}
                {PUBLIC_MODE && <Route path="register" element={<Register />} />}
                {/* Any unknown path falls back to Search: public hides admin paths,
                    internal keeps old bookmarks (/pipeline, /dashboard, /analysis,
                    /crawl) from rendering a blank page. */}
                <Route path="*" element={<Navigate to="/search" replace />} />
                {/* Crawl history now lives in the task board. */}
                {!PUBLIC_MODE && <Route path="crawl" element={<Navigate to="/tasks" replace />} />}
              </Route>
            </Routes>
          </BrowserRouter>
            <Toaster position="top-center" richColors />
          </TooltipProvider>
        </AntApp>
      </ConfigProvider>
    </QueryClientProvider>
  );
}
