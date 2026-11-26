import { useState } from 'react';
import { Globe } from 'lucide-react';
import { motion } from 'framer-motion';
import NumberedCodeSnippet from './NumberedCodeSnippet';

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import CopyButton from '../common/CopyButton';
import type { FederatedCodeHit } from '../../api/hooks/useCodeSearch';
import { originLabel } from '../../pages/codeSearchModel';

const fade = {
  hidden: { opacity: 0, y: 8 },
  show: (i: number) => ({ opacity: 1, y: 0, transition: { delay: Math.min(i * 0.03, 0.4) } }),
};

const COLLAPSE_LINES = 32;  // snippets longer than this start collapsed with an expander

function ScoreBar({ score, max }: { score: number; max: number }) {
  const ratio = max > 0 ? score / max : 0;
  const hue = 120 * ratio;  // red→green by relative rank
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          相关度
          <span className="h-1.5 w-14 overflow-hidden rounded-full bg-muted">
            <span className="block h-full" style={{ width: `${Math.max(ratio, 0.04) * 100}%`, background: `hsl(${hue},70%,45%)` }} />
          </span>
          <span className="font-medium tabular-nums text-foreground">{score.toFixed(3)}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent>本次结果最高分的 {Math.round(ratio * 100)}%</TooltipContent>
    </Tooltip>
  );
}

/** Federated code hit (private Code KB + curated Public Code). An origin badge
 * (project name for private, source + short revision for public), the cited
 * path:range, a relative score bar with its value, then the code. */
export default function CodeHitCard({
  hit, index, maxScore,
}: {
  hit: FederatedCodeHit;
  index: number;
  maxScore: number;
}) {
  // Authoritative line count comes from the cited range (backend strips trailing
  // blanks), not a newline count of hit.code (which over-counts a trailing newline).
  const total = Math.max(hit.end_line - hit.start_line + 1, 1);
  const codeLines = hit.code.replace(/\r\n/g, '\n').replace(/\n+$/, '').split('\n');
  const long = codeLines.length > COLLAPSE_LINES;
  const [expanded, setExpanded] = useState(false);
  const shown = long && !expanded ? codeLines.slice(0, COLLAPSE_LINES).join('\n') : codeLines.join('\n');
  const dir = hit.path.includes('/') ? hit.path.slice(0, hit.path.lastIndexOf('/') + 1) : '';
  const base = hit.path.slice(dir.length);
  const isPrivate = hit.origin === 'private';

  return (
    <motion.div variants={fade} custom={index} initial="hidden" animate="show">
      <article className={cn('mb-3 rounded-xl border border-l-[3px] bg-card p-3.5', isPrivate ? 'border-l-primary' : 'border-l-cyan-600')}>
        <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1.5">
          <span className="text-xs font-semibold tabular-nums text-muted-foreground">#{index + 1}</span>
          <Badge
            variant="outline"
            className={cn(
              'max-w-[240px] truncate',
              isPrivate ? 'border-primary/30 bg-primary/10 text-primary' : 'border-cyan-600/30 bg-cyan-600/10 text-cyan-700 dark:text-cyan-400',
            )}
            title={originLabel(hit)}
          >
            {!isPrivate && <Globe className="size-3" />}
            {originLabel(hit)}
          </Badge>
          <span className="flex min-w-0 items-center font-mono text-[13px]">
            {dir && <span className="truncate text-muted-foreground">{dir}</span>}
            <span className="font-semibold">{base}</span>
            <span className="text-muted-foreground">:{hit.start_line}-{hit.end_line}</span>
            <CopyButton text={`${hit.path}:${hit.start_line}`} label="复制 路径:行号" className="ml-0.5" />
          </span>
          {hit.label && <Badge variant="secondary" className="bg-violet-500/10 text-violet-700 dark:text-violet-300">{hit.label}</Badge>}
          {hit.language && <Badge variant="secondary">{hit.language}</Badge>}
          <span className="text-xs text-muted-foreground">{total} 行</span>
          <span className="ml-auto flex items-center gap-2">
            <ScoreBar score={hit.score} max={maxScore} />
            <CopyButton text={hit.code} label="复制代码" />
          </span>
        </div>
        <NumberedCodeSnippet code={shown} startLine={hit.start_line} />
        {long && (
          <button type="button" className="mt-1 text-sm text-primary hover:underline" onClick={() => setExpanded((v) => !v)}>
            {expanded ? '收起' : `展开全部 ${total} 行`}
          </button>
        )}
      </article>
    </motion.div>
  );
}
