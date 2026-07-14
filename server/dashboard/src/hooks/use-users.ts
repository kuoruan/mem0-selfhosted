import { useCallback, useEffect, useState } from "react";
import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { USER_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import type { User } from "@/types/api";

const PAGE_SIZE = 50;

interface PaginatedUsers {
  count: number;
  next: string | null;
  previous: string | null;
  results: User[];
}

export interface UseUsersResult {
  users: User[];
  isLoading: boolean;
  hasMore: boolean;
  loadMore: () => void;
}

/**
 * Paginated loader for the dashboard user list. ``/users`` is admin-gated on
 * the server, so this only fetches when ``isAdmin`` is true — non-admins get
 * an empty list and never trigger a request. Loads page 1 on mount and
 * fetches the next page on demand via ``loadMore`` (e.g. when a picker scrolls
 * near the bottom of the list).
 */
export function useUsers(isAdmin: boolean): UseUsersResult {
  const [users, setUsers] = useState<User[]>([]);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [isLoading, setLoading] = useState(false);

  const fetchPage = useCallback(async (targetPage: number) => {
    setLoading(true);
    try {
      const res = await api.get<PaginatedUsers>(USER_ENDPOINTS.BASE, {
        params: { page: targetPage, page_size: PAGE_SIZE },
      });
      const items = res.data?.results ?? [];
      setUsers((prev) => (targetPage === 1 ? items : [...prev, ...items]));
      setHasMore(res.data?.next != null);
    } catch (error) {
      toast({
        title: "Failed to load users",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      // Stop retrying the pager on error so the spinner doesn't loop forever.
      setHasMore(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isAdmin) {
      setUsers([]);
      setHasMore(false);
      return;
    }
    setPage(1);
    void fetchPage(1);
  }, [isAdmin, fetchPage]);

  const loadMore = useCallback(() => {
    if (!isAdmin || isLoading || !hasMore) return;
    const next = page + 1;
    setPage(next);
    void fetchPage(next);
  }, [isAdmin, isLoading, hasMore, page, fetchPage]);

  return { users, isLoading, hasMore, loadMore };
}
