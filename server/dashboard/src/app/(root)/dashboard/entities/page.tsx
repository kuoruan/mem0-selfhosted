"use client";

import { useMemo, useState } from "react";
import { format } from "date-fns";
import { Plus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { EmptyState } from "@/components/self-hosted/empty-state";
import InfiniteScrollSentinel from "@/components/shared/infinite-scroll-sentinel";
import DeleteConfirmationModal from "@/components/ui/delete-confirmation-modal";
import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { ENTITY_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import { useAuth } from "@/hooks/use-auth";
import EntityScopeSelect from "@/components/shared/entity-scope-select";
import type { Entity, EntityListResponse, UserInfo } from "@/types/api";
import CreateEntityDialog from "./create-entity-dialog";
import TransferOwnerDialog from "./transfer-owner-dialog";
import EditEntityNameDialog from "./edit-entity-name-dialog";
import EntityMemoryCount from "./entity-memory-count";
import EntityActions from "./entity-actions";
import ManagePermissionsSheet from "./manage-permissions-sheet";

export default function EntitiesPage() {
  const { user, isAdmin } = useAuth();
  const ownUserId = user?.id ?? "";
  const [manageEntity, setManageEntity] = useState<Entity | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [transferEntity, setTransferEntity] = useState<Entity | null>(null);
  const [editEntity, setEditEntity] = useState<Entity | null>(null);
  const [entityToDelete, setEntityToDelete] = useState<Entity | null>(null);
  const [showUnownedOnly, setShowUnownedOnly] = useState(false);
  const [scopeUserId, setScopeUserId] = useState<string>("");
  // Bump to reset the list to page 1 after a mutation (react-query invalidation).
  const [refreshNonce, setRefreshNonce] = useState(0);
  const reload = () => setRefreshNonce((n) => n + 1);

  const {
    items: entities = [],
    isLoading,
    hasMore,
    loadMore,
  } = useInfiniteList<Entity>({
    fetchPage: async (page) => {
      const params: Record<string, unknown> = { page, page_size: 50 };
      const effectiveScope = scopeUserId || ownUserId;
      if (effectiveScope !== "all" && effectiveScope) {
        params.user_id = effectiveScope;
      }
      const res = await api.get<EntityListResponse>(ENTITY_ENDPOINTS.BASE, {
        params,
      });
      return res.data ?? { results: [], next: null };
    },
    onError: (error) =>
      toast({
        title: "Failed to load entities",
        description: getErrorMessage(error),
        variant: "destructive",
      }),
    deps: [scopeUserId, ownUserId, refreshNonce],
  });

  // Infinite scroll: the sentinel fires `loadMore` on scroll-into-view.
  // Duplicate fires are safe — `useInfiniteList`'s `loadMore` guards via refs.

  const visibleEntities = useMemo(() => {
    const list = showUnownedOnly
      ? entities.filter((e) => e.owner == null)
      : entities;
    return [...list].sort((a, b) => {
      const au = a.owner == null ? 0 : 1;
      const bu = b.owner == null ? 0 : 1;
      if (au !== bu) return au - bu;
      if (a.type !== b.type) return a.type.localeCompare(b.type);
      return a.id.localeCompare(b.id);
    });
  }, [entities, showUnownedOnly]);

  const handleDelete = async () => {
    if (!entityToDelete) return;
    try {
      await api.delete(
        ENTITY_ENDPOINTS.BY_ID(entityToDelete.type, entityToDelete.id),
        { params: { namespace: entityToDelete.parent?.id } },
      );
      toast({ title: "Entity deleted", variant: "success" });
      setEntityToDelete(null);
      reload();
    } catch (error) {
      toast({
        title: "Failed to delete entity",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  const columns = [
    {
      key: "type" as keyof Entity,
      label: "Type",
      width: 90,
      render: (value: Entity["type"]) => (
        <Badge variant="outline" className="capitalize">
          {value}
        </Badge>
      ),
    },
    {
      key: "id" as keyof Entity,
      label: "ID",
      width: 240,
      render: (value: string) => (
        <span className="font-mono text-sm truncate">{value}</span>
      ),
    },
    {
      key: "owner" as keyof Entity,
      label: "Owner",
      width: 160,
      render: (_value: UserInfo | null, row: Entity) =>
        row.owner == null ? (
          <Badge
            variant="outline"
            className="border-amber-500/40 bg-amber-500/10 text-amber-700"
          >
            Unowned
          </Badge>
        ) : row.is_owner ? (
          <span className="text-sm">You</span>
        ) : (
          <span className="text-sm truncate" title={row.owner.email}>
            {row.owner.name}
          </span>
        ),
    },
    {
      key: "memories" as any,
      label: "Memories",
      width: 100,
      render: (_value: string, row: Entity) => (
        <EntityMemoryCount entity={row} />
      ),
    },
    {
      key: "updated_at" as keyof Entity,
      label: "Last Active",
      width: 120,
      render: (value: string | null) =>
        value ? format(new Date(value), "MMM d, yyyy") : "--",
    },
    {
      key: "actions" as any,
      label: "",
      width: 170,
      render: (_value: string, row: Entity) => (
        <EntityActions
          entity={row}
          isAdmin={isAdmin}
          onManage={setManageEntity}
          onTransfer={setTransferEntity}
          onEdit={setEditEntity}
          onDelete={setEntityToDelete}
        />
      ),
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-semibold font-fustat">Entities</h1>
        <div className="flex flex-wrap items-center gap-2">
          {isAdmin && (
            <>
              <EntityScopeSelect
                value={scopeUserId || ownUserId}
                onChange={setScopeUserId}
                ownUserId={ownUserId}
              />
              <label className="flex items-center gap-2 text-sm text-onSurface-default-tertiary">
                <input
                  type="checkbox"
                  className="size-4"
                  checked={showUnownedOnly}
                  disabled={(scopeUserId || ownUserId) !== "all"}
                  onChange={(e) => setShowUnownedOnly(e.target.checked)}
                />
                Unowned only
              </label>
            </>
          )}
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="size-4 mr-1" /> Create Entity
          </Button>
          <CreateEntityDialog
            open={createOpen}
            onClose={() => setCreateOpen(false)}
            isAdmin={isAdmin}
            onCreated={reload}
          />
        </div>
      </div>

      {isLoading && entities.length === 0 ? (
        <TableSkeleton rows={5} columns={6} />
      ) : visibleEntities.length === 0 ? (
        <EmptyState
          title={showUnownedOnly ? "No unowned entities" : "No entities yet"}
          description={
            showUnownedOnly
              ? "If you suspect legacy data, run Recount to surface unclaimed namespaces."
              : "Entities appear once memories are stored with a user_id, agent_id, app_id, or run_id."
          }
        />
      ) : (
        <>
          <Card className="border-memBorder-primary overflow-hidden">
            <DataTable
              data={visibleEntities}
              columns={columns}
              getRowKey={(row) => `${row.type}:${row.id}`}
            />
          </Card>
          <InfiniteScrollSentinel
            hasMore={hasMore}
            isLoading={isLoading}
            onLoadMore={loadMore}
          />
        </>
      )}

      <ManagePermissionsSheet
        entity={manageEntity}
        open={!!manageEntity}
        onClose={() => setManageEntity(null)}
        isAdmin={isAdmin}
      />

      <TransferOwnerDialog
        entity={transferEntity}
        open={!!transferEntity}
        onClose={() => setTransferEntity(null)}
        isAdmin={isAdmin}
        onTransferred={reload}
      />

      <EditEntityNameDialog
        entity={editEntity}
        open={!!editEntity}
        onClose={() => setEditEntity(null)}
        onSaved={reload}
      />

      <DeleteConfirmationModal
        isOpen={!!entityToDelete}
        onClose={() => setEntityToDelete(null)}
        onConfirm={handleDelete}
        title="Delete entity"
        description="All memories associated with this entity will be permanently removed and the namespace released for re-claim. This cannot be undone."
        itemName={entityToDelete?.id ?? ""}
        confirmButtonText="Delete"
      />
    </div>
  );
}
