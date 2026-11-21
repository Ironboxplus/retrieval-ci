import { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { motion, useReducedMotion } from 'framer-motion';
import {
  Search, FileSearch, BookOpen, ListChecks, Settings, Menu, Sun, Moon, ChevronsLeft, ChevronsRight,
  type LucideIcon,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet';
import { useAppStore } from '../stores/appStore';
import { PUBLIC_MODE } from '../constants';
import { useConsoleAccess } from '../api/hooks/useAdminConsole';
import { useIsWide } from '../hooks/useIsWide';
import ErrorBoundary from './ErrorBoundary';
import AuthMenu from './AuthMenu';
import StatusBar from './StatusBar';
import { pageTitle } from './pageTitle';

// Pages whose backend endpoints are open at the public edge.
const PUBLIC_KEYS = new Set(['/search', '/documents', '/match-doc']);

interface NavItem { key: string; icon: LucideIcon; label: string }

const baseNavItems: NavItem[] = [
  { key: '/search', icon: Search, label: '检索' },
  { key: '/match-doc', icon: FileSearch, label: '文档匹配' },
  { key: '/documents', icon: BookOpen, label: '文档库' },
  { key: '/tasks', icon: ListChecks, label: '任务' },
];

const navItems = PUBLIC_MODE ? baseNavItems.filter((m) => PUBLIC_KEYS.has(m.key)) : baseNavItems;

function BrandMark({ collapsed }: { collapsed: boolean }) {
  const navigate = useNavigate();
  return (
    <button
      type="button"
      onClick={() => navigate('/search')}
      className="flex h-14 w-full items-center gap-2.5 px-5 text-left"
    >
      <span className="size-2.5 shrink-0 rounded-[3px] bg-primary ring-4 ring-primary/15" />
      {!collapsed && <span className="text-base font-semibold tracking-tight">Retrieval</span>}
    </button>
  );
}

function NavList({ items, selectedKey, collapsed, onNavigate }: {
  items: NavItem[];
  selectedKey: string;
  collapsed: boolean;
  onNavigate: (key: string) => void;
}) {
  return (
    <nav className="grid gap-1 px-2">
      {items.map(({ key, icon: Icon, label }) => {
        const active = key === selectedKey;
        return (
          <button
            key={key}
            type="button"
            title={collapsed ? label : undefined}
            onClick={() => onNavigate(key)}
            className={cn(
              'relative flex h-9 items-center gap-3 rounded-lg px-3 text-sm transition-colors',
              active
                ? 'bg-primary/10 font-medium text-primary'
                : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              collapsed && 'justify-center px-0',
            )}
          >
            <Icon className="size-4 shrink-0" />
            {!collapsed && label}
          </button>
        );
      })}
    </nav>
  );
}

export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const darkMode = useAppStore((s) => s.darkMode);
  const toggleDarkMode = useAppStore((s) => s.toggleDarkMode);
  const collapsed = useAppStore((s) => s.siderCollapsed);
  const toggleSider = useAppStore((s) => s.toggleSider);
  const consoleAccess = useConsoleAccess();
  const isWide = useIsWide();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const reduceMotion = useReducedMotion();

  const items: NavItem[] = consoleAccess ? [...navItems, { key: '/settings', icon: Settings, label: '设置' }] : navItems;
  const selectedKey =
    items.find((m) => location.pathname === m.key || location.pathname.startsWith(`${m.key}/`))?.key
    ?? (location.pathname.startsWith('/papers') ? '/search' : '');
  const { title, sub } = pageTitle(location.pathname, location.search);

  const onNavigate = (key: string) => {
    navigate(key);
    setDrawerOpen(false);
  };

  const themeToggle = (
    <Button variant="ghost" size="icon" onClick={toggleDarkMode} aria-label={darkMode ? '切换到浅色' : '切换到深色'}>
      {darkMode ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  );

  const topSegment = `/${location.pathname.split('/')[1] || ''}`;

  return (
    <div className="mx-auto flex h-svh w-full max-w-[1680px] gap-3 p-2 text-foreground md:p-3">
      {/* Content-height card, not a full-height rail: five links in a
          viewport-tall panel left an empty column under them. */}
      {isWide && (
        <aside
          className={cn(
            'flex shrink-0 flex-col self-start rounded-2xl border bg-background pb-2 shadow-sm transition-[width] duration-200',
            collapsed ? 'w-16' : 'w-56',
          )}
        >
          <BrandMark collapsed={collapsed} />
          <NavList items={items} selectedKey={selectedKey} collapsed={collapsed} onNavigate={onNavigate} />
          <div className={cn('mx-2 mt-2 flex border-t pt-2', collapsed ? 'justify-center' : 'justify-end')}>
            <Button variant="ghost" size="icon" onClick={toggleSider} aria-label={collapsed ? '展开侧栏' : '收起侧栏'}>
              {collapsed ? <ChevronsRight className="size-4" /> : <ChevronsLeft className="size-4" />}
            </Button>
          </div>
        </aside>
      )}

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <header className="flex h-14 shrink-0 items-center gap-3 rounded-2xl border bg-background px-3 shadow-sm md:px-5">
          {!isWide && (
            <Button variant="ghost" size="icon" onClick={() => setDrawerOpen(true)} aria-label="打开菜单">
              <Menu className="size-4" />
            </Button>
          )}
          <div className="flex min-w-0 items-baseline gap-2">
            <h1 className="shrink-0 text-base font-semibold">{title}</h1>
            {sub && <span className="hidden truncate text-sm text-muted-foreground sm:inline">/ {sub}</span>}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <StatusBar compact={!isWide} />
            {PUBLIC_MODE && <AuthMenu />}
            {themeToggle}
          </div>
        </header>

        {/* layoutScroll: framer-motion measures shared-layout cards (MorphGrid)
            relative to this scroll container instead of the window. */}
        <motion.main layoutScroll className="min-h-0 flex-1 overflow-y-auto rounded-2xl">
          <div className="pb-10">
            <ErrorBoundary>
              {/* Enter-only. An exit animation cannot work around <Outlet />: it
                  always renders the CURRENT route, so the exiting copy flashed
                  the new page before fading, and mode="wait" left the new page
                  at opacity 0 whenever that exit was interrupted (blank page). */}
              <motion.div
                key={topSegment}
                initial={{ opacity: 0, x: reduceMotion ? 0 : 12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: reduceMotion ? 0.1 : 0.18, ease: 'easeOut' }}
              >
                <Outlet />
              </motion.div>
            </ErrorBoundary>
          </div>
        </motion.main>
      </div>

      {!isWide && (
        <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
          <SheetContent side="left" className="m-2 h-[calc(100%-1rem)] w-60 rounded-2xl border bg-background p-0 shadow-lg">
            <SheetTitle className="sr-only">导航</SheetTitle>
            <BrandMark collapsed={false} />
            <NavList items={items} selectedKey={selectedKey} collapsed={false} onNavigate={onNavigate} />
          </SheetContent>
        </Sheet>
      )}
    </div>
  );
}
