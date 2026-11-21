import { useRef } from 'react';
import { Loader2, Search, X } from 'lucide-react';

import { Button } from '@/components/ui/button';

/**
 * Large query input with a clear button and a 检索 submit button. Enter
 * submits unless an IME composition is in progress (Chinese input confirms
 * candidates with Enter).
 */
export default function SearchBox({
  value, onChange, onSearch, loading, placeholder,
}: {
  value: string;
  onChange: (next: string) => void;
  onSearch: (q: string) => void;
  loading?: boolean;
  placeholder?: string;
}) {
  const composing = useRef(false);
  return (
    <form
      className="flex min-w-0 flex-1"
      onSubmit={(e) => {
        e.preventDefault();
        if (!composing.current) onSearch(value);
      }}
    >
      <div className="relative min-w-0 flex-1">
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onCompositionStart={() => { composing.current = true; }}
          onCompositionEnd={() => { composing.current = false; }}
          placeholder={placeholder}
          className="h-10 w-full rounded-l-lg border border-r-0 border-input bg-background pl-3 pr-8 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/30 dark:bg-input/30"
        />
        {value && (
          <button
            type="button"
            aria-label="清空"
            onClick={() => onChange('')}
            className="absolute right-2 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>
      <Button type="submit" className="h-10 rounded-l-none px-5">
        {loading ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}
        检索
      </Button>
    </form>
  );
}
