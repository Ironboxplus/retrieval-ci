import { useEffect, useState } from 'react';

/**
 * True when the viewport is at least `breakpointPx` wide (default 992,
 * antd's own `lg` breakpoint — Layout.tsx's sider-vs-mobile-drawer split
 * already used this same threshold before this hook was pulled out for
 * Tasks.tsx's board to reuse). Tracks live via matchMedia so resizing or
 * rotating the device updates without a reload.
 */
export function useIsWide(breakpointPx = 992): boolean {
  const [wide, setWide] = useState(
    () => typeof window !== 'undefined' && window.innerWidth >= breakpointPx,
  );
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${breakpointPx}px)`);
    const onChange = () => setWide(mq.matches);
    onChange();
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [breakpointPx]);
  return wide;
}
