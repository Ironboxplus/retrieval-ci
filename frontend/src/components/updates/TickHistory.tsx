import { Fragment, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { StatusBadge } from '../TaskParts';
import { ReplayRun } from './RunPanels';
import RunPapers from './RunPapers';
import { apiErrorText, useScheduleTicks } from '../../api/hooks/useSchedules';
import { formatDuration } from '../../pages/taskTime';
import { countColumns, formatCount, formatInZone, hasRunPapers, relativeFromNow, tickRow } from '../../pages/updatesModel';

const FIXED_COLUMNS = 5;

/** Run history of one schedule; a finished run opens into its stages, events and
 * (for arXiv delta runs) the papers it touched. */
export default function TickHistory({ scheduleId, kind, timezone }: {
  scheduleId: number;
  kind: string;
  timezone: string;
}) {
  const ticks = useScheduleTicks(scheduleId);
  const columns = useMemo(() => countColumns(kind), [kind]);
  const rows = useMemo(() => (ticks.data?.pages ?? []).flat().map((t) => tickRow(t, kind)), [ticks.data, kind]);
  const width = FIXED_COLUMNS + columns.length;
  const [openId, setOpenId] = useState<number | null>(null);

  return (
    <div className="grid gap-2">
      <h4 className="text-sm font-semibold">运行历史</h4>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-8" />
            <TableHead>时间</TableHead>
            <TableHead>触发</TableHead>
            <TableHead>状态</TableHead>
            <TableHead>用时</TableHead>
            {columns.map((c) => (
              <TableHead key={c.key} className="text-right">{c.label}<span className="ml-0.5 text-[10px] font-normal text-muted-foreground">篇</span></TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            // A task that still runs is shown live above, not replayed.
            const expandable = row.taskId != null && row.taskStatus !== 'running' && row.taskStatus !== 'pending';
            const open = expandable && openId === row.id;
            return (
              <Fragment key={row.id}>
                <TableRow
                  className={cn(expandable && 'cursor-pointer')}
                  onClick={expandable ? () => setOpenId(open ? null : row.id) : undefined}
                >
                  <TableCell>
                    {expandable && (open ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {row.time ? (
                      <>
                        {formatInZone(row.time, timezone)}
                        <span className="ml-2 text-xs text-muted-foreground">{relativeFromNow(row.time)}</span>
                      </>
                    ) : '—'}
                  </TableCell>
                  <TableCell>{row.origin}</TableCell>
                  <TableCell>
                    {row.taskStatus ? (
                      <StatusBadge status={row.taskStatus} />
                    ) : (
                      <Badge variant="outline" className="text-muted-foreground" title={row.reason ?? undefined}>{row.tickLabel}</Badge>
                    )}
                  </TableCell>
                  <TableCell className="tabular-nums">{formatDuration(row.durationSeconds) ?? '—'}</TableCell>
                  {columns.map((c) => (
                    <TableCell key={c.key} className="text-right tabular-nums">{formatCount(row.counts[c.key])}</TableCell>
                  ))}
                </TableRow>
                {open && row.taskId != null && (
                  <TableRow className="hover:bg-transparent">
                    <TableCell colSpan={width} className="bg-muted/20 p-4 whitespace-normal">
                      <div className="grid gap-6">
                        <ReplayRun taskId={row.taskId} />
                        {hasRunPapers(kind) && <RunPapers taskId={row.taskId} />}
                      </div>
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            );
          })}
          {rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={width} className="py-5 text-center text-xs text-muted-foreground">
                {ticks.isError ? apiErrorText(ticks.error) : ticks.isLoading ? '加载中…' : '暂无记录'}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {ticks.hasNextPage && (
        <Button variant="ghost" size="sm" className="text-muted-foreground" disabled={ticks.isFetchingNextPage} onClick={() => ticks.fetchNextPage()}>
          {ticks.isFetchingNextPage ? '加载中…' : '显示更早的'}
        </Button>
      )}
    </div>
  );
}
