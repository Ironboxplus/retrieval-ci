import { useMemo, useState } from 'react';
import Form from '@rjsf/antd';
import validator from '@rjsf/validator-ajv8';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import Segmented from '../common/Segmented';
import {
  apiErrorText, usePatchSchedule, useTriggerPreview,
  type Schedule, type ScheduleKind,
} from '../../api/hooks/useSchedules';
import {
  INTERVAL_UNIT_LABEL, TRIGGER_TYPE_LABEL, WEEKDAYS, draftFromTrigger, formatInZone, relativeFromNow, triggerFromDraft,
  type IntervalUnit, type Trigger, type TriggerDraft, type Weekday,
} from '../../pages/updatesModel';
import { localizedSchema, paramsUiSchema } from '../../pages/updatesParams';

const WEEKDAY_NAME: Record<Weekday, string> = { mon: '一', tue: '二', wed: '三', thu: '四', fri: '五', sat: '六', sun: '日' };
const ZONES = [...new Set(['UTC', ...Intl.supportedValuesOf('timeZone')])];

function Field({ label, htmlFor, children, className }: { label: string; htmlFor: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn('grid gap-1.5', className)}>
      <Label htmlFor={htmlFor}>{label}</Label>
      {children}
    </div>
  );
}

function TriggerForm({ draft, onChange }: { draft: TriggerDraft; onChange: (next: TriggerDraft) => void }) {
  const set = (patch: Partial<TriggerDraft>) => onChange({ ...draft, ...patch });
  return (
    <div className="grid gap-3">
      <Segmented
        ariaLabel="触发方式"
        value={draft.type}
        options={(Object.keys(TRIGGER_TYPE_LABEL) as Trigger['type'][]).map((type) => ({ value: type, label: TRIGGER_TYPE_LABEL[type] }))}
        onChange={(type) => set({ type })}
      />
      <div className="grid grid-cols-2 gap-3">
        {draft.type !== 'interval' ? (
          <Field label="时间" htmlFor="trigger-time">
            <Input id="trigger-time" name="time" type="time" value={draft.time} onChange={(e) => set({ time: e.target.value })} />
          </Field>
        ) : (
          <Field label="间隔" htmlFor="trigger-amount">
            <div className="flex gap-2">
              <Input
                id="trigger-amount" name="amount"
                type="number" min={1} value={Number.isFinite(draft.amount) ? draft.amount : ''}
                onChange={(e) => set({ amount: e.target.valueAsNumber })}
              />
              <Segmented
                ariaLabel="间隔单位"
                value={draft.unit}
                options={(Object.keys(INTERVAL_UNIT_LABEL) as IntervalUnit[]).map((unit) => ({ value: unit, label: INTERVAL_UNIT_LABEL[unit] }))}
                onChange={(unit) => set({ unit })}
              />
            </div>
          </Field>
        )}
        <Field label="时区" htmlFor="trigger-timezone">
          <Input id="trigger-timezone" name="timezone" list="schedule-zones" value={draft.timezone} onChange={(e) => set({ timezone: e.target.value })} />
          <datalist id="schedule-zones">{ZONES.map((z) => <option key={z} value={z} />)}</datalist>
        </Field>
      </div>
      {draft.type === 'weekly' && (
        <Field label="星期" htmlFor="trigger-days">
          <div id="trigger-days" className="flex gap-1.5">
            {WEEKDAYS.map((day) => {
              const on = draft.days.includes(day);
              return (
                <Button
                  key={day} type="button" size="sm" variant={on ? 'default' : 'outline'} aria-pressed={on}
                  onClick={() => set({ days: on ? draft.days.filter((d) => d !== day) : [...draft.days, day] })}
                >
                  {WEEKDAY_NAME[day]}
                </Button>
              );
            })}
          </div>
        </Field>
      )}
    </div>
  );
}

/** Edit a schedule's title, trigger (with the server's preview of the next five
 * runs) and parameters (a form generated from the kind's JSON Schema). */
export default function ScheduleEditDialog({ schedule, kind, onClose }: {
  schedule: Schedule;
  kind: ScheduleKind | undefined;
  onClose: () => void;
}) {
  const patch = usePatchSchedule();
  const [title, setTitle] = useState(schedule.title);
  const [draft, setDraft] = useState(() => draftFromTrigger(schedule.trigger_json));
  const [params, setParams] = useState<Record<string, unknown>>(schedule.params_json);

  const trigger = triggerFromDraft(draft);
  const preview = useTriggerPreview(trigger);

  const schema = useMemo(() => (kind ? localizedSchema(kind.params_schema) : null), [kind]);
  const uiSchema = useMemo(() => (kind ? paramsUiSchema(kind.name, kind.params_schema) : {}), [kind]);
  const hasParams = schema != null && Object.keys(schema.properties ?? {}).length > 0;
  const paramsValid = !hasParams || validator.validateFormData(params, schema).errors.length === 0;

  const valid = title.trim() !== '' && trigger != null && preview.isSuccess && !preview.isPlaceholderData && paramsValid;

  const save = () => {
    if (!trigger) return;
    patch.mutate(
      { id: schedule.id, patch: { title: title.trim(), trigger, ...(hasParams ? { params } : {}) } },
      { onSuccess: () => { toast.success('已保存'); onClose(); } },
    );
  };

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[90svh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader><DialogTitle>编辑计划</DialogTitle></DialogHeader>

        <div className="grid gap-5">
          <Field label="标题" htmlFor="schedule-title"><Input id="schedule-title" name="title" value={title} onChange={(e) => setTitle(e.target.value)} /></Field>

          <div className="grid gap-3">
            <h4 className="text-sm font-semibold">触发</h4>
            <TriggerForm draft={draft} onChange={setDraft} />
            <div className="rounded-lg bg-muted/50 p-3">
              <div className="mb-1 text-[11px] text-muted-foreground">下 5 次运行</div>
              {preview.isError ? (
                <p className="text-xs text-destructive">{apiErrorText(preview.error)}</p>
              ) : trigger == null ? (
                <p className="text-xs text-muted-foreground">—</p>
              ) : (
                <ol className={cn('grid gap-0.5 text-sm tabular-nums', preview.isPlaceholderData && 'opacity-50')}>
                  {(preview.data ?? []).map((t) => (
                    <li key={t}>
                      {formatInZone(t, trigger.timezone)}
                      <span className="ml-2 text-xs text-muted-foreground">{relativeFromNow(t)}</span>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </div>

          {hasParams && schema && (
            <div className="grid gap-3">
              <h4 className="text-sm font-semibold">参数</h4>
              <Form
                tagName="div"
                // The default prefix makes the root field id="root", which index.css's `#root { min-height: 100vh }` stretches.
                idPrefix="schedule-params"
                schema={schema}
                uiSchema={uiSchema}
                formData={params}
                validator={validator}
                liveValidate
                showErrorList={false}
                onChange={(e) => setParams((e.formData ?? {}) as Record<string, unknown>)}
              >
                <></>
              </Form>
            </div>
          )}
        </div>

        {patch.isError && <p className="text-xs text-destructive">{apiErrorText(patch.error)}</p>}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>取消</Button>
          <Button disabled={!valid || patch.isPending} onClick={save}>{patch.isPending ? '保存中…' : '保存'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
