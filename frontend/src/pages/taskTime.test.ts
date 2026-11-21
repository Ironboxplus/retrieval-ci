import { describe, expect, it } from 'vitest';

import { parseCreatedAt, parseStartedAt, relativeTimeFrom, formatDuration, formatDateTime } from './taskTime';

// Production fact (F1-f, confirmed on .88): `created_at` is UTC with no zone
// suffix; `started_at` is local time, also with no zone suffix. The same
// naive wall-clock string must be interpreted differently depending on
// which field it came from, or every task looks like it started 8h before
// it was created.
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

describe('parseStartedAt', () => {
  it('treats a naive string as local wall-clock time', () => {
    const d = parseStartedAt('2026-09-20T10:00:00');
    // Constructing with the same local-time fields must match exactly —
    // this is timezone-independent because both sides use the local ctor.
    expect(d.getTime()).toBe(new Date(2026, 8, 20, 10, 0, 0).getTime());
  });

  it('respects an explicit Z suffix (already unambiguous)', () => {
    const d = parseStartedAt('2026-09-20T10:00:00Z');
    expect(d.toISOString()).toBe('2026-09-20T10:00:00.000Z');
  });
});

describe('created_at vs started_at: the documented 8h production drift', () => {
  it('the same naive string parses to different instants for the two fields', () => {
    // Only meaningful off UTC — skip in a UTC test runner (offset 0 makes
    // parseCreatedAt and parseStartedAt legitimately agree).
    if (new Date().getTimezoneOffset() === 0) return;
    const raw = '2026-09-20T10:00:00';
    expect(parseCreatedAt(raw).getTime()).not.toBe(parseStartedAt(raw).getTime());
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
