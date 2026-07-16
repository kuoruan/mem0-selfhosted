import { api } from "@/utils/api";
import { ENTITY_ENDPOINTS } from "@/utils/api-endpoints";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import type { Entity, EntityListResponse } from "@/types/api";

const PAGE_SIZE = 100;

export interface UseEntityUsersResult {
  users: Entity[];
  isLoading: boolean;
  hasMore: boolean;
  loadMore: () => void;
}

/**
 * Paginated loader for user-type entities (``GET /entities?type=user``).
 *
 * Thin wrapper over ``useInfiniteList``. ``/entities`` is not admin-gated —
 * every caller sees their own owned + granted entities, so this always fetches.
 * Owned entities sort first server-side. Errors are swallowed silently (the
 * picker just shows what it has).
 */
export function useEntityUsers(): UseEntityUsersResult {
  const { items, isLoading, hasMore, loadMore } = useInfiniteList<Entity>({
    fetchPage: async (page) => {
      const res = await api.get<EntityListResponse>(ENTITY_ENDPOINTS.BASE, {
        params: { type: "user", page, page_size: PAGE_SIZE },
      });
      return res.data;
    },
  });
  return { users: items, isLoading, hasMore, loadMore };
}
