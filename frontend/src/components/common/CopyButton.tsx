import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { copyText } from '../../utils/clipboard';

/** Icon button that copies `text` and flips to a check for 1.5 s. */
export default function CopyButton({ text, label, className }: { text: string; label: string; className?: string }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await copyText(text);
      setDone(true);
      setTimeout(() => setDone(false), 1500);
    } catch {
      toast.error('复制失败');
    }
  };
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={copy}
          aria-label={label}
          className={cn('inline-flex size-6 shrink-0 items-center justify-center rounded text-muted-foreground hover:bg-accent hover:text-foreground', className)}
        >
          {done ? <Check className="size-3.5 text-success" /> : <Copy className="size-3.5" />}
        </button>
      </TooltipTrigger>
      <TooltipContent>{done ? '已复制' : label}</TooltipContent>
    </Tooltip>
  );
}
