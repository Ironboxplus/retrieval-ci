import { cn } from '@/lib/utils';

export interface SegmentedOption<T extends string> { value: T; label: string }

/** Pill-style single-choice switch (the shadcn "tabs list" look without tab panels). */
export default function Segmented<T extends string>({
  value, options, onChange, size = 'sm', ariaLabel, className,
}: {
  value: T;
  options: SegmentedOption<T>[];
  onChange: (next: T) => void;
  size?: 'sm' | 'lg';
  ariaLabel?: string;
  className?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn('inline-flex shrink-0 items-center rounded-lg bg-muted p-0.5', size === 'lg' ? 'h-10 p-1' : 'h-8', className)}
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            'rounded-md font-medium whitespace-nowrap transition-colors',
            size === 'lg' ? 'h-8 px-3.5 text-sm' : 'h-7 px-2.5 text-xs',
            value === o.value ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
