import { describe, expect, it } from 'vitest';

import { taskActionErrorDetail } from './useTasks';

describe('taskActionErrorDetail', () => {
  it('uses the string detail of a 409 response', () => {
    const err = {
      message: 'Request failed with status code 409',
      response: { status: 409, data: { detail: 'Journal 索引已开始写入 Qdrant，无法取消' } },
    };
    expect(taskActionErrorDetail(err)).toBe('Journal 索引已开始写入 Qdrant，无法取消');
  });

  it('falls back to the error message when detail is an object', () => {
    const err = {
      message: 'Request failed with status code 422',
      response: { status: 422, data: { detail: [{ loc: ['body'], msg: 'bad' }] } },
    };
    expect(taskActionErrorDetail(err)).toBe('Request failed with status code 422');
  });

  it('uses the error message for a network error without response', () => {
    expect(taskActionErrorDetail(new Error('Network Error'))).toBe('Network Error');
  });

  it('returns a generic text when nothing usable is present', () => {
    expect(taskActionErrorDetail(undefined)).toBe('未知错误');
  });
});
