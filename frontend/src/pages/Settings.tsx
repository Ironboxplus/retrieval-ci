import { useSearchParams } from 'react-router-dom';
import { Database, Server, BookMarked, type LucideIcon } from 'lucide-react';

import { cn } from '@/lib/utils';
import DataManagementPanel from './settings/DataManagementPanel';
import PublicCodeConsole from './PublicCodeConsole';
import JournalConsole from './JournalConsole';

type Tab = 'data' | 'public-code' | 'journal';
const VALID: Tab[] = ['data', 'public-code', 'journal'];

const SECTIONS: { key: Tab; label: string; icon: LucideIcon }[] = [
  { key: 'data', label: '数据管理', icon: Database },
  { key: 'public-code', label: '公共代码库', icon: Server },
  { key: 'journal', label: '记忆', icon: BookMarked },
];

/**
 * /settings — internal build or public admin only (ConsoleGate wraps the
 * route). A left section list synced to `?tab=`; folds in 数据管理,
 * 公共代码库 and 记忆.
 */
export default function Settings() {
  const [sp, setSp] = useSearchParams();
  const raw = sp.get('tab');
  const tab: Tab = VALID.includes(raw as Tab) ? (raw as Tab) : 'data';

  const setTab = (key: Tab) => {
    const next = new URLSearchParams(sp);
    next.set('tab', key);
    setSp(next, { replace: true });
  };

  return (
    <div className="grid gap-6 md:grid-cols-[180px_minmax(0,1fr)]">
      <nav className="flex gap-1 overflow-x-auto md:flex-col">
        {SECTIONS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            className={cn(
              'flex h-9 shrink-0 items-center gap-2.5 rounded-md px-3 text-sm transition-colors',
              tab === key ? 'bg-primary/10 font-medium text-primary' : 'text-muted-foreground hover:bg-accent hover:text-foreground',
            )}
          >
            <Icon className="size-4" />
            {label}
          </button>
        ))}
      </nav>
      <div className="min-w-0">
        {tab === 'data' && <DataManagementPanel />}
        {tab === 'public-code' && <PublicCodeConsole />}
        {tab === 'journal' && <JournalConsole />}
      </div>
    </div>
  );
}
