import type { CSSProperties, ReactNode } from 'react';

import { cn } from '@/lib/utils';

/**
 * Shared visual frame for MorphGrid's compact/expanded card content — one
 * surface (border, radius, padding) for every morph card in the app. The
 * morph animation lives in MorphGrid; this component has no motion.
 */
export default function MorphCard({
  children, style, bodyStyle, className,
}: {
  children: ReactNode;
  style?: CSSProperties;
  bodyStyle?: CSSProperties;
  className?: string;
}) {
  return (
    <div className={cn('overflow-hidden rounded-xl border bg-card text-card-foreground shadow-xs', className)} style={style}>
      <div className="p-3.5" style={bodyStyle}>{children}</div>
    </div>
  );
}
