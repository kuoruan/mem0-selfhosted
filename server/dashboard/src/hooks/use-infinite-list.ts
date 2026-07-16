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
 * concurrent guard backs off. A monotonic ``queryIdRef`` discards results from
 * fetches superseded by a deps/enabled change, so a stale page can't append
 * onto the reset list.
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
  // Monotonic session id. Bumped on every fetch and on every reset; a fetch
  // whose id no longer matches the current session is stale (superseded by a
  // deps/enabled change or a newer page) and its result is discarded so it
  // can't append stale items onto a reset list.
  const queryIdRef = useRef(0);

  const run = useCallback(async (targetPage: number) => {
    const id = ++queryIdRef.current;
    isLoadingRef.current = true;
    setLoading(true);
    try {
      const data = await fetchPageRef.current(targetPage);
      if (id !== queryIdRef.current) return;
      const newItems = data.results ?? [];
      setItems((prev) =>
        targetPage === 1 ? newItems : [...prev, ...newItems],
      );
      pageRef.current = targetPage;
      const next = data.next != null;
      hasMoreRef.current = next;
      setHasMore(next);
    } catch (err) {
      if (id !== queryIdRef.current) return;
      hasMoreRef.current = false;
      setHasMore(false);
      onErrorRef.current?.(err);
    } finally {
      // Only the owning session clears the loading flag; a stale fetch leaves
      // it for the newer session that already set it.
      if (id === queryIdRef.current) {
        isLoadingRef.current = false;
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    // Invalidate any in-flight fetch from the previous deps/enabled state.
    queryIdRef.current += 1;
    if (!enabled) {
      setItems([]);
      hasMoreRef.current = false;
      setHasMore(false);
      isLoadingRef.current = false;
      setLoading(false);
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
