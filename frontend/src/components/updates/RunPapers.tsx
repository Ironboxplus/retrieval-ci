import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronLeft, ChevronRight } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { apiErrorText, useRunPapers } from '../../api/hooks/useSchedules';
import { PAPER_KINDS, PAPER_KIND_LABEL, formatCount, type PaperKind } from '../../pages/updatesModel';

const PAGE = 20;

/** The papers one delta run touched, one tab per kind with its count. */
export default function RunPapers({ taskId }: { taskId: number }) {
  const [kind, setKind] = useState<PaperKind>('new');
  const [offset, setOffset] = useState(0);
  const papers = useRunPapers(taskId, kind, offset, PAGE);
  const page = papers.data;
  const total = page?.total ?? 0;

  return (
    <div className="grid gap-3">
      <h4 className="text-sm font-semibold">本轮论文</h4>
      <Tabs value={kind} onValueChange={(next) => { setKind(next as PaperKind); setOffset(0); }}>
        <TabsList>
          {PAPER_KINDS.map((k) => (
            <TabsTrigger key={k} value={k}>
              {PAPER_KIND_LABEL[k]}
              <span className="ml-1.5 tabular-nums text-muted-foreground">{page ? formatCount(page.counts[k] ?? 0) : '—'}</span>
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {papers.isError ? (
        <p className="text-xs text-destructive">{apiErrorText(papers.error)}</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-40">arXiv ID</TableHead>
              <TableHead>标题</TableHead>
              <TableHead className="w-24">版本</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(page?.papers ?? []).map((p) => (
              <TableRow key={p.paper_id}>
                <TableCell className="tabular-nums">{p.arxiv_id ?? p.paper_id}</TableCell>
                <TableCell className="max-w-0 truncate" title={p.title ?? undefined}>
                  <Link to={`/papers/${p.paper_id}`} className="hover:underline">{p.title || '—'}</Link>
                </TableCell>
                <TableCell className="tabular-nums">
                  {p.source_version ?? '—'}
                  {p.withdrawn && <Badge variant="outline" className="ml-1.5 border-destructive/40 text-destructive">已撤稿</Badge>}
                </TableCell>
              </TableRow>
            ))}
            {page && page.papers.length === 0 && (
              <TableRow><TableCell colSpan={3} className="py-5 text-center text-xs text-muted-foreground">没有{PAPER_KIND_LABEL[kind]}的论文</TableCell></TableRow>
            )}
            {!page && (
              <TableRow><TableCell colSpan={3} className="py-5 text-center text-xs text-muted-foreground">加载中…</TableCell></TableRow>
            )}
          </TableBody>
        </Table>
      )}

      {total > PAGE && (
        <div className="flex items-center justify-end gap-2 text-xs tabular-nums text-muted-foreground">
          <span>{formatCount(offset + 1)}–{formatCount(Math.min(offset + PAGE, total))} / {formatCount(total)} 篇</span>
          <Button variant="outline" size="icon-sm" aria-label="上一页" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            <ChevronLeft className="size-4" />
          </Button>
          <Button variant="outline" size="icon-sm" aria-label="下一页" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>
            <ChevronRight className="size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
