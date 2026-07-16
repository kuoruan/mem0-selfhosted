"use client";

import { useEffect, useEffectEvent, useRef } from "react";

interface InfiniteScrollSentinelProps {
  /** When false, the sentinel renders nothing. */
  hasMore: boolean;
  /** Whether a page is currently loading. Drives the label and re-subscription. */
  isLoading: boolean;
  /** Called when the sentinel scrolls into view and more pages remain. */
  onLoadMore: () => void;
  /** Viewport padding so the load fires before the sentinel is fully visible. */
  rootMargin?: string;
  /** Text shown when idle (more available, not loading). */
  idleLabel?: string;
  /** Text shown while a page is loading. */
  loadingLabel?: string;
  className?: string;
}

/**
 * Drop-in sentinel for infinite-scroll lists. Place it after the list/table;
 * an IntersectionObserver fires ``onLoadMore`` when it scrolls into view.
 * ``onLoadMore`` is an effect event, so the observer re-subscribes only on
 * ``hasMore`` / ``isLoading`` / ``rootMargin`` — a short first page that leaves
 * the sentinel in view auto-continues to the next page on the load transition.
 */
export default function InfiniteScrollSentinel({
  hasMore,
  isLoading,
  onLoadMore,
  rootMargin = "200px",
  idleLabel = "Scroll for more",
  loadingLabel = "Loading more…",
  className,
}: InfiniteScrollSentinelProps) {
  const ref = useRef<HTMLDivElement>(null);
  const loadMoreEvent = useEffectEvent(onLoadMore);

  useEffect(() => {
    if (!hasMore) return;
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) loadMoreEvent();
      },
      { rootMargin, threshold: 0 },
    );
    io.observe(el);
    return () => io.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasMore, isLoading, rootMargin]);

  if (!hasMore) return null;
  return (
    <div
      ref={ref}
      className={
        className ??
        "flex justify-center py-2 text-xs text-onSurface-default-tertiary"
      }
    >
      {isLoading ? loadingLabel : idleLabel}
    </div>
  );
}
