import { describe, expect, it } from 'vitest';

import { pageTitle } from './pageTitle';

describe('pageTitle', () => {
  it('names the search mode', () => {
    expect(pageTitle('/search', '')).toEqual({ title: '检索', sub: '论文' });
    expect(pageTitle('/search', '?mode=code&q=x')).toEqual({ title: '检索', sub: '代码' });
  });
  it('names the settings tab, defaulting to data', () => {
    expect(pageTitle('/settings', '')).toEqual({ title: '设置', sub: '数据管理' });
    expect(pageTitle('/settings', '?tab=public-code')).toEqual({ title: '设置', sub: '公共代码库' });
    expect(pageTitle('/settings', '?tab=journal')).toEqual({ title: '设置', sub: '记忆' });
  });
  it('names the plain pages', () => {
    expect(pageTitle('/tasks', '').title).toBe('任务');
    expect(pageTitle('/documents', '').title).toBe('文档库');
    expect(pageTitle('/match-doc', '').title).toBe('文档匹配');
    expect(pageTitle('/papers/abc', '')).toEqual({ title: '检索', sub: '论文详情' });
  });
});
