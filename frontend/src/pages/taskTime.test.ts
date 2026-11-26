import { describe, expect, it } from 'vitest';

import { parseCreatedAt, parseTimestamp, instantMs, relativeTimeFrom, formatDuration, formatDateTime, formatClock } from './taskTime';

// Task timestamps carry an explicit offset; `parseCreatedAt` still reads the
// naive, UTC-written strings other endpoints send.
describe('parseCreatedAt', () => {
  it('treats a naive (no-zone) string as UTC', () => {
    const d = parseCreatedAt('2026-09-20T10:00:00');
    expect(d.toISOString()).toBe('2026-09-20T10:00:00.000Z');
  });

  it('treats a naive string with a space separator as UTC too', () => {
    const d = parseCreatedAt('2026-09-20 10:00:00');
    expect(d.toISOString()).toBe('2026-09-20T10:00:00.000Z');
  });

  it('respects an explicit Z suffix instead of double-applying UTC', () => {
    const d = parseCreatedAt('2026-09-20T10:00:00Z');
    expect(d.toISOString()).toBe('2026-09-20T10:00:00.000Z');
  });

  it('respects an explicit numeric offset', () => {
    const d = parseCreatedAt('2026-09-20T18:00:00+08:00');
    expect(d.toISOString()).toBe('2026-09-20T10:00:00.000Z');
  });
});

describe('parseTimestamp / instantMs', () => {
  it('reads the offset', () => {
    expect(parseTimestamp('2026-09-20T18:00:00+08:00').toISOString()).toBe('2026-09-20T10:00:00.000Z');
    expect(instantMs('2026-09-20T10:00:00Z')).toBe(Date.UTC(2026, 8, 20, 10));
  });

  it('orders by instant, not by string', () => {
    // 10:00+08:00 is 02:00Z, earlier than 05:00Z although it sorts later as text.
    expect(instantMs('2026-09-20T10:00:00+08:00')).toBeLessThan(instantMs('2026-09-20T05:00:00Z'));
  });

  it('sorts a missing or invalid value before everything', () => {
    expect(instantMs(undefined)).toBe(0);
    expect(instantMs('nope')).toBe(0);
  });
});

describe('relativeTimeFrom', () => {
  const now = new Date('2026-09-20T12:00:00.000Z');

  it('renders seconds-scale gaps as "刚刚"', () => {
    expect(relativeTimeFrom(new Date('2026-09-20T11:59:50.000Z'), now)).toBe('刚刚');
  });

  it('renders minute-scale gaps', () => {
    expect(relativeTimeFrom(new Date('2026-09-20T11:55:00.000Z'), now)).toBe('5 分钟前');
  });

  it('renders hour-scale gaps', () => {
    expect(relativeTimeFrom(new Date('2026-09-20T09:00:00.000Z'), now)).toBe('3 小时前');
  });

  it('renders day-scale gaps', () => {
    expect(relativeTimeFrom(new Date('2026-09-17T12:00:00.000Z'), now)).toBe('3 天前');
  });

  it('clamps a future timestamp (clock skew) to "刚刚" rather than a negative gap', () => {
    expect(relativeTimeFrom(new Date('2026-09-20T12:05:00.000Z'), now)).toBe('刚刚');
  });
});

describe('formatDuration', () => {
  it('renders seconds, minutes and hours with units', () => {
    expect(formatDuration(51)).toBe('51 秒');
    expect(formatDuration(120)).toBe('2 分');
    expect(formatDuration(125)).toBe('2 分 5 秒');
    expect(formatDuration(3725)).toBe('1 小时 2 分');
    expect(formatDuration(7200)).toBe('2 小时');
  });
  it('returns null for unknown or invalid input', () => {
    expect(formatDuration(null)).toBeNull();
    expect(formatDuration(undefined)).toBeNull();
    expect(formatDuration(-1)).toBeNull();
  });
});

describe('formatDateTime', () => {
  it('pads to YYYY-MM-DD HH:mm in local time', () => {
    expect(formatDateTime(new Date(2026, 8, 3, 7, 5))).toBe('2026-09-03 07:05');
  });
});

describe('formatClock', () => {
  it('pads to HH:mm:ss in local time', () => {
    expect(formatClock(new Date(2026, 8, 3, 7, 5, 9))).toBe('07:05:09');
  });
});
