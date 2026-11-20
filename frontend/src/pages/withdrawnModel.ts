// 撤稿论文在前端的取值与参数：命中卡片、论文列表、论文详情和检索工具栏共用。
// 后端字段：`withdrawn`（bool，来自 SQL，权威）、`source_version`（正文所属版本，如 v2，
// 会议论文为空）、`withdrawn_at`（仅详情）。检索默认排除撤稿，`include_withdrawn=true` 才带回。

export type WithdrawnSource = {
  withdrawn?: unknown;
  source_version?: unknown;
  withdrawn_at?: unknown;
};

export type WithdrawnMarks = {
  withdrawn: boolean;
  /** 正文所属版本，如 v2；无版本信息（会议论文）时为 null。 */
  version: string | null;
  /** 标签文案「正文版本 v2」；无版本时为 null。 */
  versionLabel: string | null;
  /** 仅撤稿论文有：撤稿标记时间（原样，UTC ISO）。 */
  withdrawnAt: string | null;
};

const text = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null;

export function withdrawnMarks(source: WithdrawnSource | null | undefined): WithdrawnMarks {
  const version = text(source?.source_version);
  const withdrawn = source?.withdrawn === true;
  return {
    withdrawn,
    version,
    versionLabel: version ? `正文版本 ${version}` : null,
    withdrawnAt: withdrawn ? text(source?.withdrawn_at) : null,
  };
}

// ── 检索页：开关进 URL（与 titleOnly 同形：缺省即关，'1' 才开）──
export const INCLUDE_WITHDRAWN_PARAM = 'withdrawn';

export function readIncludeWithdrawn(sp: URLSearchParams): boolean {
  return sp.get(INCLUDE_WITHDRAWN_PARAM) === '1';
}

export function writeIncludeWithdrawn(sp: URLSearchParams, includeWithdrawn: boolean): void {
  if (includeWithdrawn) sp.set(INCLUDE_WITHDRAWN_PARAM, '1');
}

/**
 * 请求里的 include_withdrawn。只有「论文」来源才发（撤稿只存在于 arXiv 论文，上传文档/
 * 网页没有这个字段）；关闭时发 undefined 而不是 false，保持请求体与缓存键干净。
 */
export function includeWithdrawnField(sourceType: string, includeWithdrawn: boolean): true | undefined {
  return sourceType === 'paper' && includeWithdrawn ? true : undefined;
}
