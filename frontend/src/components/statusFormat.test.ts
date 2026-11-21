import { describe, expect, it } from 'vitest';

import { fmtBytes, fmtCount, fmtPercent, fmtSigned, fmtUptime } from './statusFormat';

describe('statusFormat', () => {
  it('formats counts with separators and a dash for unknown', () => {
    expect(fmtCount(14014330)).toBe('14,014,330');
    expect(fmtCount(null)).toBe('—');
  });
  it('formats bytes with binary units', () => {
    expect(fmtBytes(629 * 1024 ** 3)).toBe('629 GB');
    expect(fmtBytes(1.5 * 1024 ** 2)).toBe('1.5 MB');
    expect(fmtBytes(12)).toBe('12 B');
  });
  it('formats uptime in the two largest units', () => {
    expect(fmtUptime(15000)).toBe('4 小时 10 分');
    expect(fmtUptime(200000)).toBe('2 天 7 小时');
    expect(fmtUptime(90)).toBe('1 分');
  });
  it('signs deltas and formats ratios', () => {
    expect(fmtSigned(1337)).toBe('+1,337');
    expect(fmtSigned(-5)).toBe('-5');
    expect(fmtPercent(0.8123)).toBe('81.2%');
    expect(fmtPercent(null)).toBe('—');
  });
});
