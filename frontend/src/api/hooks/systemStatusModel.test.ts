import { describe, expect, it } from 'vitest';

import { deriveHealthLevel, healthReasons, LATENCY_NORMAL_MS } from './systemStatusModel';

const BASE = {
  qdrantReachable: true,
  vectorDelta: 0,
  optimizerStatus: 'ok',
  diskFreeRatio: 0.5,
  latencyMs: null,
};

describe('deriveHealthLevel', () => {
  it('is ok when everything checks out', () => {
    expect(deriveHealthLevel(BASE)).toBe('ok');
  });

  it('is down when Qdrant is unreachable, regardless of other signals', () => {
    expect(deriveHealthLevel({ ...BASE, qdrantReachable: false })).toBe('down');
    expect(deriveHealthLevel({
      ...BASE, qdrantReachable: false, vectorDelta: 0, optimizerStatus: 'ok', diskFreeRatio: 0.9,
    })).toBe('down');
  });

  it('is warn when vector_delta is nonzero', () => {
    expect(deriveHealthLevel({ ...BASE, vectorDelta: 1337 })).toBe('warn');
    expect(deriveHealthLevel({ ...BASE, vectorDelta: -5 })).toBe('warn');
  });

  it('is ok when vector_delta is exactly zero or unknown (null)', () => {
    expect(deriveHealthLevel({ ...BASE, vectorDelta: 0 })).toBe('ok');
    expect(deriveHealthLevel({ ...BASE, vectorDelta: null })).toBe('ok');
  });

  it('is warn when optimizer status is not "ok"', () => {
    expect(deriveHealthLevel({ ...BASE, optimizerStatus: 'error' })).toBe('warn');
    expect(deriveHealthLevel({ ...BASE, optimizerStatus: { error: 'stuck' } })).toBe('warn');
  });

  it('treats a null/undefined optimizer status as no signal (not a warn)', () => {
    expect(deriveHealthLevel({ ...BASE, optimizerStatus: null })).toBe('ok');
    expect(deriveHealthLevel({ ...BASE, optimizerStatus: undefined })).toBe('ok');
  });

  it('is warn when disk free ratio is under 10%', () => {
    expect(deriveHealthLevel({ ...BASE, diskFreeRatio: 0.09 })).toBe('warn');
    expect(deriveHealthLevel({ ...BASE, diskFreeRatio: 0 })).toBe('warn');
  });

  it('is ok at exactly 10% free and above', () => {
    expect(deriveHealthLevel({ ...BASE, diskFreeRatio: 0.10 })).toBe('ok');
  });

  it('is ok when disk free ratio is unknown (null)', () => {
    expect(deriveHealthLevel({ ...BASE, diskFreeRatio: null })).toBe('ok');
  });

  it('combines multiple warn signals and still reports warn (not worse)', () => {
    expect(deriveHealthLevel({
      qdrantReachable: true, vectorDelta: 10, optimizerStatus: 'error', diskFreeRatio: 0.01, latencyMs: null,
    })).toBe('warn');
  });

  it('is warn when latency exceeds the fixed normal threshold, regardless of recent history', () => {
    expect(deriveHealthLevel({ ...BASE, latencyMs: LATENCY_NORMAL_MS + 1 })).toBe('warn');
  });

  it('is ok at exactly the threshold and below', () => {
    expect(deriveHealthLevel({ ...BASE, latencyMs: LATENCY_NORMAL_MS })).toBe('ok');
    expect(deriveHealthLevel({ ...BASE, latencyMs: 1 })).toBe('ok');
  });

  it('treats unknown latency (null) as no signal, not a warn', () => {
    expect(deriveHealthLevel({ ...BASE, latencyMs: null })).toBe('ok');
  });

  it('exposes the threshold as a fixed constant, not derived from samples', () => {
    expect(LATENCY_NORMAL_MS).toBe(100);
  });
});

describe('healthReasons', () => {
  it('is empty when every rule passes', () => {
    expect(healthReasons(BASE)).toEqual([]);
  });
  it('names the unreachable store alone', () => {
    expect(healthReasons({ ...BASE, qdrantReachable: false, vectorDelta: 5 })).toEqual(['Qdrant 不可达']);
  });
  it('names each firing rule with its value', () => {
    const reasons = healthReasons({
      ...BASE, vectorDelta: 1337, diskFreeRatio: 0.05, latencyMs: LATENCY_NORMAL_MS + 59, optimizerStatus: 'yellow',
    });
    expect(reasons).toEqual([
      '向量比分块多 1,337',
      'Qdrant 优化器异常',
      '磁盘可用不足 10%（5.0%）',
      `检索延迟 ${LATENCY_NORMAL_MS + 59} ms，超过 ${LATENCY_NORMAL_MS} ms`,
    ]);
  });
  it('says fewer when vectors lag behind chunks', () => {
    expect(healthReasons({ ...BASE, vectorDelta: -12 })).toEqual(['向量比分块少 12']);
  });
});
