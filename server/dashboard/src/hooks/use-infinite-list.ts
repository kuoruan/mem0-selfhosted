import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DependencyList,
} from "react";

/** A single page from a paginated endpoint — only the fields the hook needs. */
export interface InfinitePage<T> {
  results: T[];
  next: string | null;
}

export interface UseInfiniteListResult<T> {
  items: T[];
  isLoading: boolean;
  hasMore: boolean;
  loadMore: () => void;
}

interface UseInfiniteListOptions<T> {
  fetchPage: (page: number) => Promise<InfinitePage<T>>;
  /** Gate fetching; when false, items reset to empty and no request fires. */
  enabled?: boolean;
  /** Optional error callback (e.g. toast). On error the pager stops. */
  onError?: (error: unknown) => void;
  /** Reset the list (clear items, return to page 1) when these change.
   *  Omit to load once on mount. */
  deps?: DependencyList;
}

const EMPTY_DEPS: DependencyList = [];

/**
 * Infinite-scroll list loader (react-query ``useInfiniteQuery`` semantics;
 * distinct from ``useApiQuery``'s single-page refetch). ``fetchPage`` /
 * ``onError`` are read through refs so inline closures won't retrigger the
 * mount fetch.
 *
 * ``loadMore`` guards read ``page`` / ``isLoading`` / ``hasMore`` from refs
 * (not state) so rapid scroll events can't observe stale values and fire
 * duplicate fetches; ``pageRef`` advances before the async resolves so a
 * concurrent guard backs off.
 */
export function useInfiniteList<T>({
  fetchPage,
  enabled = true,
  onError,
  deps = EMPTY_DEPS,
}: UseInfiniteListOptions<T>): UseInfiniteListResult<T> {
  const [items, setItems] = useState<T[]>([]);
  const [hasMore, setHasMore] = useState(enabled);
  const [isLoading, setLoading] = useState(enabled);

  const fetchPageRef = useRef(fetchPage);
  fetchPageRef.current = fetchPage;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  // Refs mirror state for synchronous guard reads. Initialized to `enabled`
  // so a sentinel callback firing before the mount fetch still sees "loading"
  // and backs off — prevents page 2 being claimed before page 1 starts.
  const pageRef = useRef(1);
  const isLoadingRef = useRef(enabled);
  const hasMoreRef = useRef(enabled);

  const run = useCallback(async (targetPage: number) => {
    isLoadingRef.current = true;
    setLoading(true);
    try {
      const data = await fetchPageRef.current(targetPage);
      const newItems = data.results ?? [];
      setItems((prev) =>
        targetPage === 1 ? newItems : [...prev, ...newItems],
      );
      pageRef.current = targetPage;
      const next = data.next != null;
      hasMoreRef.current = next;
      setHasMore(next);
    } catch (err) {
      hasMoreRef.current = false;
      setHasMore(false);
      onErrorRef.current?.(err);
    } finally {
      isLoadingRef.current = false;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      setItems([]);
      hasMoreRef.current = false;
      setHasMore(false);
      return;
    }
    pageRef.current = 1;
    hasMoreRef.current = true;
    setHasMore(true);
    void run(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, run, ...deps]);

  const loadMore = useCallback(() => {
    if (!enabled || isLoadingRef.current || !hasMoreRef.current) return;
    // Claim the page before the async resolves so a concurrent guard backs off.
    const next = pageRef.current + 1;
    pageRef.current = next;
    void run(next);
  }, [enabled, run]);

  return { items, isLoading, hasMore, loadMore };
}
