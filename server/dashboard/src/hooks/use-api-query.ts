import {
  useCallback,
  useEffect,
  useEffectEvent,
  useState,
  type DependencyList,
} from "react";
import { toast } from "@/components/ui/use-toast";
import { getErrorMessage } from "@/lib/error-message";

// Stable default so callers that omit `deps` don't refetch every render.
const EMPTY_DEPS: DependencyList = [];

interface UseApiQueryOptions<T> {
  enabled?: boolean;
  errorToast?: string;
  initialData?: T;
  /** Refetch when these change (react-query style). Omit to fetch once on mount. */
  deps?: DependencyList;
}

interface UseApiQueryResult<T> {
  data: T | undefined;
  isLoading: boolean;
  error: string;
  refetch: () => Promise<void>;
}

export function useApiQuery<T>(
  fetcher: () => Promise<T>,
  options: UseApiQueryOptions<T> = {},
): UseApiQueryResult<T> {
  const {
    enabled = true,
    errorToast,
    initialData,
    deps = EMPTY_DEPS,
  } = options;
  const [data, setData] = useState<T | undefined>(initialData);
  const [isLoading, setIsLoading] = useState(enabled);
  const [error, setError] = useState("");

  // Effect events: `fetcher` / `errorToast` can change without retriggering
  // the mount fetch (and without becoming effect dependencies).
  const fetchEvent = useEffectEvent(() => fetcher());
  const handleError = useEffectEvent((err: unknown) => {
    const message = getErrorMessage(err, errorToast || "Request failed");
    setError(message);
    if (errorToast) {
      toast({
        title: errorToast,
        description: message,
        variant: "destructive",
      });
    }
  });

  const run = useCallback(async () => {
    setIsLoading(true);
    setError("");
    try {
      setData(await fetchEvent());
    } catch (err) {
      handleError(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (enabled) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, run, ...deps]);

  return { data, isLoading, error, refetch: run };
}
