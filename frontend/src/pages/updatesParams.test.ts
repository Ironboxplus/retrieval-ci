import { describe, expect, it } from 'vitest';
import type { RJSFSchema } from '@rjsf/utils';

import { localizedSchema, paramRows, paramsUiSchema, paramTitle } from './updatesParams';

const schema: RJSFSchema = {
  title: 'ArxivDeltaParams',
  description: 'What one run may be told',
  type: 'object',
  properties: {
    html_concurrency: { type: 'integer', title: 'Html Concurrency', description: 'Parallel HTML downloads', minimum: 1, maximum: 6, default: 3 },
    limit: { type: 'integer', title: 'Limit', description: 'Download at most this many', minimum: 0, default: 0 },
    note: { type: 'string', title: 'Note' },
  },
};

describe('paramTitle', () => {
  it('names the field with its unit, and spells out what 0 means', () => {
    expect(paramTitle('arxiv_delta', 'html_concurrency')).toBe('HTML 并发（路）');
    expect(paramTitle('arxiv_delta', 'limit')).toBe('单次下载上限（篇，0 = 不限）');
  });

  it('names the volume source parameters', () => {
    expect(paramTitle('arxiv_delta', 'source')).toBe('下载来源');
    expect(paramTitle('arxiv_delta', 'receive_deadline_hours')).toBe('等待 cfscan 运行（小时）');
    expect(paramTitle('arxiv_delta', 'volume')).toBe('115 卷');
  });

  it('keeps the schema title for a field it has no label for', () => {
    expect(paramTitle('arxiv_delta', 'note', 'Note')).toBe('Note');
    expect(paramTitle('other_kind', 'x')).toBe('x');
  });
});

describe('paramRows', () => {
  it('shows each stored parameter with name and unit', () => {
    expect(paramRows('arxiv_delta', { html_concurrency: 3, no_html_retry_days: 3, limit: 0 })).toEqual([
      { name: 'HTML 并发', value: '3 路' },
      { name: '缺 HTML 重试窗口', value: '3 天' },
      { name: '单次下载上限', value: '不限' },
    ]);
    expect(paramRows('arxiv_delta', { limit: 50 })).toEqual([{ name: '单次下载上限', value: '50 篇' }]);
  });

  it('shows the volume source parameters with name and unit', () => {
    expect(paramRows('arxiv_delta', { source: 'volume', receive_deadline_hours: 6, volume: '/cfs-arxiv' })).toEqual([
      { name: '下载来源', value: 'volume' },
      { name: '等待 cfscan 运行', value: '6 小时' },
      { name: '115 卷', value: '/cfs-arxiv' },
    ]);
  });

  it('shows an unknown parameter by its key', () => {
    expect(paramRows('other_kind', { x: 1 })).toEqual([{ name: 'x', value: '1' }]);
  });
});

describe('form schemas', () => {
  it('drops English titles and descriptions but keeps the constraints', () => {
    const out = localizedSchema(schema);
    expect(out.title).toBeUndefined();
    expect(out.description).toBeUndefined();
    expect(out.properties?.html_concurrency).toEqual({ type: 'integer', minimum: 1, maximum: 6, default: 3 });
    expect(schema.title).toBe('ArxivDeltaParams');
  });

  it('titles every field in the uiSchema and renders no submit button', () => {
    const ui = paramsUiSchema('arxiv_delta', schema);
    expect(ui.html_concurrency['ui:title']).toBe('HTML 并发（路）');
    expect(ui.note['ui:title']).toBe('Note');
    expect(ui['ui:submitButtonOptions']).toEqual({ norender: true });
  });
});
