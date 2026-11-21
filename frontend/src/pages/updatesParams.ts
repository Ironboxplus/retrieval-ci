/**
 * Chinese names and units for a job kind's parameters. The form itself is
 * rendered by react-jsonschema-form from the kind's `params_schema`
 * (GET /api/schedules/kinds); this only supplies what a Pydantic schema
 * cannot: the Chinese label, the unit and the meaning of 0.
 */
import type { RJSFSchema, UiSchema } from '@rjsf/utils';

interface ParamLabel {
  label: string;
  unit?: string;
  /** What a value of 0 means, shown instead of the number. */
  zero?: string;
}

const PARAM_LABELS: Record<string, Record<string, ParamLabel>> = {
  arxiv_delta: {
    html_concurrency: { label: 'HTML 并发', unit: '路' },
    no_html_retry_days: { label: '缺 HTML 重试窗口', unit: '天' },
    limit: { label: '单次下载上限', unit: '篇', zero: '不限' },
    source: { label: '下载来源' },
    receive_deadline_hours: { label: '等待 cfscan 运行', unit: '小时' },
    volume: { label: '115 卷' },
  },
};

const labelOf = (kind: string, field: string): ParamLabel | undefined => PARAM_LABELS[kind]?.[field];

/** The form's field title: name with its unit, e.g. "HTML 并发（路）". */
export function paramTitle(kind: string, field: string, fallback: string = field): string {
  const label = labelOf(kind, field);
  if (!label) return fallback;
  const hint = label.zero ? `${label.unit ?? ''}，0 = ${label.zero}` : label.unit;
  return hint ? `${label.label}（${hint}）` : label.label;
}

/** One stored parameter as shown on a schedule card. */
export function paramRows(kind: string, params: Record<string, unknown>): { name: string; value: string }[] {
  return Object.entries(params).map(([field, value]) => {
    const label = labelOf(kind, field);
    if (!label) return { name: field, value: String(value) };
    if (value === 0 && label.zero) return { name: label.label, value: label.zero };
    return { name: label.label, value: label.unit ? `${value} ${label.unit}` : String(value) };
  });
}

/** The schema without Pydantic's English titles/descriptions (the Chinese ones come
 * from the uiSchema) — descriptions would otherwise print under every field. */
const withoutText = <T extends object>(schema: T): T =>
  Object.fromEntries(Object.entries(schema).filter(([key]) => key !== 'title' && key !== 'description')) as T;

export function localizedSchema(schema: RJSFSchema): RJSFSchema {
  const properties = Object.fromEntries(
    Object.entries(schema.properties ?? {}).map(([field, prop]) => [field, typeof prop === 'object' ? withoutText(prop) : prop]),
  );
  return { ...withoutText(schema), properties };
}

export function paramsUiSchema(kind: string, schema: RJSFSchema): UiSchema {
  const ui: UiSchema = { 'ui:submitButtonOptions': { norender: true } };
  for (const [field, prop] of Object.entries(schema.properties ?? {})) {
    const fallback = typeof prop === 'object' && prop.title ? prop.title : field;
    ui[field] = { 'ui:title': paramTitle(kind, field, fallback) };
  }
  return ui;
}
