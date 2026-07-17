import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { USER_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import { EMPTY_PAGE_RESULTS } from "@/utils/helpers";
import type { User, UserListResponse } from "@/types/api";

const PAGE_SIZE = 50;

export interface UseUsersResult {
  users: User[];
  isLoading: boolean;
  hasMore: boolean;
  loadMore: () => void;
}

/**
 * Paginated loader for the dashboard user list. ``/users`` is admin-gated on
 * the server, so this only fetches when ``isAdmin`` is true — non-admins get
 * an empty list and never trigger a request. Errors surface a toast.
 */
export function useUsers(isAdmin: boolean): UseUsersResult {
  const { items, isLoading, hasMore, loadMore } = useInfiniteList<User>({
    enabled: isAdmin,
    onError: (error) =>
      toast({
        title: "Failed to load users",
        description: getErrorMessage(error),
        variant: "destructive",
      }),
    fetchPage: async (page) => {
      const res = await api.get<UserListResponse>(USER_ENDPOINTS.BASE, {
        params: { page, page_size: PAGE_SIZE },
      });
      return res.data ?? EMPTY_PAGE_RESULTS;
    },
  });
  return { users: items, isLoading, hasMore, loadMore };
}
