import { describe, expect, it } from 'vitest';

import {
  countColumns, draftFromTrigger, elapsedSeconds, formatInZone, formatIntervalSeconds, formatMetric, formatTrigger, originLabel,
  relativeFromNow, resultCounts, secondsBetween, tickRow, triggerFromDraft,
  type Tick, type Trigger,
} from './updatesModel';

describe('formatTrigger', () => {
  it('daily names the clock time and zone', () => {
    expect(formatTrigger({ type: 'daily', time: '03:00', timezone: 'Asia/Shanghai' })).toBe('每日 03:00（Asia/Shanghai）');
  });

  it('weekly lists weekdays in week order whatever order they were stored in', () => {
    expect(formatTrigger({ type: 'weekly', days: ['wed', 'mon'], time: '09:30', timezone: 'UTC' }))
      .toBe('每周一、三 09:30（UTC）');
    expect(formatTrigger({ type: 'weekly', days: ['sun', 'sat'], time: '00:05', timezone: 'UTC' }))
      .toBe('每周六、日 00:05（UTC）');
  });

  it('interval picks the largest exact unit', () => {
    expect(formatTrigger({ type: 'interval', seconds: 30, timezone: 'UTC' })).toBe('每 30 秒');
    expect(formatTrigger({ type: 'interval', seconds: 1800, timezone: 'UTC' })).toBe('每 30 分钟');
    expect(formatTrigger({ type: 'interval', seconds: 7200, timezone: 'UTC' })).toBe('每 2 小时');
    expect(formatIntervalSeconds(90)).toBe('90 秒');
  });
});

describe('trigger draft', () => {
  const roundTrip = (t: Trigger) => triggerFromDraft(draftFromTrigger(t));

  it('round-trips every shape', () => {
    const shapes: Trigger[] = [
      { type: 'daily', time: '03:00', timezone: 'Asia/Shanghai' },
      { type: 'weekly', days: ['mon', 'fri'], time: '21:15', timezone: 'UTC' },
      { type: 'interval', seconds: 30, timezone: 'UTC' },
      { type: 'interval', seconds: 600, timezone: 'UTC' },
      { type: 'interval', seconds: 10800, timezone: 'UTC' },
    ];
    for (const shape of shapes) expect(roundTrip(shape)).toEqual(shape);
  });

  it('interval units convert to seconds', () => {
    const draft = draftFromTrigger({ type: 'interval', seconds: 60, timezone: 'UTC' });
    expect(triggerFromDraft({ ...draft, amount: 2, unit: 'hours' })).toEqual({ type: 'interval', seconds: 7200, timezone: 'UTC' });
  });

  it('is null while incomplete or invalid', () => {
    const daily = draftFromTrigger({ type: 'daily', time: '03:00', timezone: 'UTC' });
    expect(triggerFromDraft({ ...daily, time: '3:00' })).toBeNull();
    expect(triggerFromDraft({ ...daily, time: '24:00' })).toBeNull();
    expect(triggerFromDraft({ ...daily, timezone: ' ' })).toBeNull();
    expect(triggerFromDraft({ ...daily, type: 'weekly', days: [] })).toBeNull();
    expect(triggerFromDraft({ ...daily, type: 'interval', amount: 0 })).toBeNull();
  });

  it('keeps fields of the other shapes when the type is switched back and forth', () => {
    const draft = draftFromTrigger({ type: 'weekly', days: ['tue'], time: '08:00', timezone: 'UTC' });
    const asDaily = triggerFromDraft({ ...draft, type: 'daily' });
    expect(asDaily).toEqual({ type: 'daily', time: '08:00', timezone: 'UTC' });
    expect(triggerFromDraft({ ...draft, type: 'weekly' })).toEqual({ type: 'weekly', days: ['tue'], time: '08:00', timezone: 'UTC' });
  });
});

describe('times', () => {
  it('formats an instant on the wall clock of the given zone', () => {
    expect(formatInZone('2026-11-02T19:00:00+00:00', 'Asia/Shanghai')).toBe('2026-11-03 03:00');
    expect(formatInZone('2026-11-02T19:00:00+00:00', 'UTC')).toBe('2026-11-02 19:00');
    expect(formatInZone('not a date', 'UTC')).toBe('—');
  });

  it('reads midnight as 00, not 24', () => {
    expect(formatInZone('2026-11-02T16:00:00Z', 'Asia/Shanghai')).toBe('2026-11-03 00:00');
  });

  it('relative time points both ways', () => {
    const now = new Date('2026-11-03T00:00:00Z');
    expect(relativeFromNow('2026-11-03T00:00:30Z', now)).toBe('30 秒后');
    expect(relativeFromNow('2026-11-02T21:00:00Z', now)).toBe('3 小时前');
    expect(relativeFromNow('2026-11-05T00:00:00Z', now)).toBe('2 天后');
    expect(relativeFromNow('2026-11-02T23:59:58Z', now)).toBe('现在');
  });

  it('seconds between needs both ends', () => {
    expect(secondsBetween('2026-11-03T00:00:00Z', '2026-11-03T00:12:00Z')).toBe(720);
    expect(secondsBetween('2026-11-03T00:00:00Z', null)).toBeNull();
  });
});

describe('elapsedSeconds', () => {
  it('counts from the start while the task runs', () => {
    expect(elapsedSeconds({ status: 'running', duration_seconds: 5 }, 1_000_000, 1_090_000)).toBe(90);
  });

  it('uses the server figure once finished', () => {
    expect(elapsedSeconds({ status: 'completed', duration_seconds: 720 }, 1_000_000, 9_999_999)).toBe(720);
    expect(elapsedSeconds({ status: 'running', duration_seconds: 42 }, null, 9_999_999)).toBe(42);
    expect(elapsedSeconds(null, null, 0)).toBeNull();
  });
});

describe('formatMetric', () => {
  it('keeps name-less value and unit together with a thousands separator', () => {
    expect(formatMetric({ name: '语料', value: 133128, unit: '篇' })).toBe('133,128 篇');
    expect(formatMetric({ name: 'x', value: '2026-09-23', unit: '' })).toBe('2026-09-23');
  });

  it('shows a dash for a gauge with no reading', () => {
    expect(formatMetric({ name: 'OAI 水位', value: null, unit: '' })).toBe('—');
  });
});

describe('resultCounts', () => {
  it('names each present count, keeps zeros, skips absent keys', () => {
    expect(resultCounts({ new: 1021, replaced: 0, html_missing: 7, convert_rejected: 2, convert_errors: 1, junk: 'x' })).toEqual([
      { key: 'new', label: '新论文', value: 1021 },
      { key: 'replaced', label: '新版本', value: 0 },
      { key: 'html_missing', label: '缺 HTML', value: 7 },
      { key: 'convert_rejected', label: '转换拒绝', value: 2 },
      { key: 'convert_errors', label: '转换出错', value: 1 },
    ]);
    expect(resultCounts(null)).toEqual([]);
  });
});

describe('tickRow', () => {
  const base: Tick = {
    id: 9, schedule_id: 2, scheduled_for: '2026-11-02T19:00:00+00:00', fired_at: '2026-11-02T19:00:01+00:00',
    trigger: 'cron', status: 'enqueued', task_id: 403, attempt: 0,
  };

  it('maps a finished run to counts and duration', () => {
    const row = tickRow({
      ...base,
      task: {
        id: 403, status: 'completed', started_at: '2026-11-02T19:00:01+00:00', completed_at: '2026-11-02T19:12:01+00:00',
        result_json: {
          new: 1021, replaced: 12, withdrawn: 3, html_missing: 40, convert_rejected: 5,
          download_errors: 2, preindex_failed: 1, convert_errors: 4,
        },
      },
    }, 'arxiv_delta');
    expect(row).toMatchObject({ origin: '定时', taskStatus: 'completed', tickLabel: null, durationSeconds: 720 });
    expect(row.counts).toEqual({ new: 1021, replaced: 12, withdrawn: 3, html_missing: 40, convert_rejected: 5, failed: 7 });
  });

  it('a run without result keeps its counts empty rather than zero', () => {
    const row = tickRow({ ...base, task: { id: 403, status: 'running', started_at: base.fired_at, completed_at: null, result_json: null } }, 'arxiv_delta');
    expect(row).toMatchObject({ taskStatus: 'running', durationSeconds: null });
    expect(row.counts.new).toBeNull();
    expect(row.counts.failed).toBeNull();
  });

  it('a skipped tick has no task and says why', () => {
    const row = tickRow({ ...base, status: 'skipped', reason: 'already running #403', task_id: null, task: null });
    expect(row).toMatchObject({ taskStatus: null, tickLabel: '已跳过', reason: 'already running #403', taskId: null });
  });

  it('labels manual and numbered retry origins', () => {
    expect(tickRow({ ...base, trigger: 'manual' }).origin).toBe('手动');
    expect(tickRow({ ...base, trigger: 'retry', attempt: 2 }).origin).toBe('重试 2');
  });

  it('falls back to the fire time when the slot is unknown', () => {
    expect(tickRow({ ...base, scheduled_for: null }).time).toBe(base.fired_at);
  });
});

describe('countColumns', () => {
  it('arXiv delta names its counts, other kinds have none', () => {
    expect(countColumns('arxiv_delta').map((c) => c.label)).toEqual(['新论文', '新版本', '撤稿', '缺 HTML', '转换拒绝', '失败']);
    expect(countColumns('public_code_updates')).toEqual([]);
    expect(tickRow({
      id: 1, schedule_id: 1, scheduled_for: null, fired_at: null, trigger: 'cron', status: 'enqueued', task_id: 5, attempt: 0,
      task: { id: 5, status: 'completed', result_json: { new: 9 } },
    }, 'public_code_updates').counts).toEqual({});
  });
});

describe('originLabel', () => {
  it('reads params.origin.trigger', () => {
    expect(originLabel({ schedule_id: 2, trigger: 'manual' })).toBe('手动');
    expect(originLabel(undefined)).toBeNull();
  });
});
