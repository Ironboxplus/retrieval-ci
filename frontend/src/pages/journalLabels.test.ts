import { describe, expect, it } from 'vitest';

import { sourceKindLabel, clientLabel } from './journalLabels';

describe('sourceKindLabel', () => {
  it('translates known entry-kind values to Chinese', () => {
    expect(sourceKindLabel('memory')).toBe('记忆');
    expect(sourceKindLabel('session_summary')).toBe('会话摘要');
    expect(sourceKindLabel('handoff')).toBe('交接');
    expect(sourceKindLabel('manual_note')).toBe('手动笔记');
    expect(sourceKindLabel('doc')).toBe('文档');
  });

  it('falls back to the raw value for an unrecognized kind, never throws', () => {
    expect(sourceKindLabel('future_kind')).toBe('future_kind');
  });
});

describe('clientLabel', () => {
  it('gives a display-cased label for known clients, never a bare lowercase tag', () => {
    expect(clientLabel('mcp')).toBe('MCP');
    expect(clientLabel('http')).toBe('HTTP');
    expect(clientLabel('codex')).toBe('Codex');
    expect(clientLabel('claude')).toBe('Claude');
  });

  it('falls back to the raw value for an unrecognized client, never throws', () => {
    expect(clientLabel('future_client')).toBe('future_client');
  });
});
