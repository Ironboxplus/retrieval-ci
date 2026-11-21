import { useEffect, useRef, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

import { cn } from '@/lib/utils';
import type { TaskEvent } from '../../types';
import { EventRow } from '../TaskParts';

const ROW_ESTIMATE = 24;
const AT_BOTTOM_PX = 24;

/** A task's events as a virtualised list that follows the newest one until the
 * reader scrolls up; scrolling back to the bottom resumes it. */
export default function EventList({ events, className }: { events: TaskEvent[]; className?: string }) {
  const parentRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  const virtualizer = useVirtualizer({
    count: events.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_ESTIMATE,
    overscan: 12,
    getItemKey: (index) => events[index].id,
  });

  useEffect(() => {
    if (follow && events.length > 0) virtualizer.scrollToIndex(events.length - 1, { align: 'end' });
  }, [follow, events.length, virtualizer]);

  return (
    <div
      ref={parentRef}
      className={cn('h-72 overflow-auto rounded-md bg-muted/50 px-2 py-1', className)}
      // Only the reader pauses following: a scroll event alone cannot tell the
      // reader from the list re-measuring rows under its own scroll-to-end.
      onWheel={(e) => e.deltaY < 0 && setFollow(false)}
      onTouchMove={() => setFollow(false)}
      onKeyDown={() => setFollow(false)}
      onPointerDown={(e) => e.target === e.currentTarget && setFollow(false)}
      onScroll={(e) => {
        const el = e.currentTarget;
        if (el.scrollHeight - el.scrollTop - el.clientHeight <= AT_BOTTOM_PX) setFollow(true);
      }}
      tabIndex={0}
    >
      {events.length === 0 ? (
        <div className="py-6 text-center text-xs text-muted-foreground">暂无事件</div>
      ) : (
        <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
          {virtualizer.getVirtualItems().map((item) => (
            <div
              key={item.key}
              ref={virtualizer.measureElement}
              data-index={item.index}
              className="absolute inset-x-0 top-0"
              style={{ transform: `translateY(${item.start}px)` }}
            >
              <EventRow event={events[item.index]} animated={false} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
