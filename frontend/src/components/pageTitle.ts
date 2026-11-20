/** Header title for the current route — the page's name plus the sub-view
 * the URL selects (search mode, settings tab), so the header states where
 * you are instead of sitting empty. */
const SETTINGS_TAB: Record<string, string> = { data: '数据管理', 'public-code': '公共代码库', journal: '记忆' };

export function pageTitle(pathname: string, search: string): { title: string; sub?: string } {
  const params = new URLSearchParams(search);
  if (pathname.startsWith('/search')) {
    return { title: '检索', sub: params.get('mode') === 'code' ? '代码' : '论文' };
  }
  if (pathname.startsWith('/papers/')) return { title: '检索', sub: '论文详情' };
  if (pathname.startsWith('/papers')) return { title: '论文' };
  if (pathname.startsWith('/match-doc')) return { title: '文档匹配' };
  if (pathname.startsWith('/documents')) return { title: '文档库' };
  if (pathname.startsWith('/tasks')) return { title: '任务' };
  if (pathname.startsWith('/settings')) {
    return { title: '设置', sub: SETTINGS_TAB[params.get('tab') ?? 'data'] ?? SETTINGS_TAB.data };
  }
  if (pathname.startsWith('/register')) return { title: '注册' };
  return { title: '' };
}
