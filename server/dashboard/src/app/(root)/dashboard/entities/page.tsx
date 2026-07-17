"use client";

import { useState } from "react";
import { format } from "date-fns";
import { Plus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import PaginationBar from "@/components/shared/pagination-bar";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { orDash, isWildcard, WILDCARD, EMPTY_PAGE_RESULTS } from "@/utils/helpers";
import { EmptyState } from "@/components/self-hosted/empty-state";
import DeleteConfirmationModal from "@/components/ui/delete-confirmation-modal";
import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { ENTITY_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import { useApiQuery } from "@/hooks/use-api-query";
import { useAuth } from "@/hooks/use-auth";
import EntityScopeSelect from "@/components/shared/entity-scope-select";
import type { Entity, EntityListResponse, UserInfo } from "@/types/api";
import CreateEntityDialog from "./create-entity-dialog";
import TransferOwnerDialog from "./transfer-owner-dialog";
import EditEntityNameDialog from "./edit-entity-name-dialog";
import EntityMemoryCount from "./entity-memory-count";
import EntityActions from "./entity-actions";
import ManagePermissionsSheet from "./manage-permissions-sheet";

const PERMISSION_BADGES: Record<string, string> = {
  owner: "border-amber-500/40 bg-amber-500/10 text-amber-700",
  admin: "border-red-500/40 bg-red-500/10 text-red-700",
  write: "border-blue-500/40 bg-blue-500/10 text-blue-700",
  read: "border-slate-500/40 bg-slate-500/10 text-slate-700",
};

const TYPE_OPTIONS = [
  { value: WILDCARD, label: "All" },
  { value: "user", label: "User" },
  { value: "agent", label: "Agent" },
  { value: "app", label: "App" },
  { value: "run", label: "Run" },
];

const PAGE_SIZE = 20;

export default function EntitiesPage() {
  const { user, isAdmin } = useAuth();
  const ownUserId = user?.id ?? "";
  const [page, setPage] = useState(0);
  const [manageEntity, setManageEntity] = useState<Entity | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [transferEntity, setTransferEntity] = useState<Entity | null>(null);
  const [editEntity, setEditEntity] = useState<Entity | null>(null);
  const [entityToDelete, setEntityToDelete] = useState<Entity | null>(null);
  const [showUnownedOnly, setShowUnownedOnly] = useState(false);
  const [scopeUserId, setScopeUserId] = useState<string>(WILDCARD);
  const [typeFilter, setTypeFilter] = useState<string>(WILDCARD);

  const [refreshNonce, setRefreshNonce] = useState(0);
  const reload = () => { setRefreshNonce((n) => n + 1); setPage(0); };

  const {
    data: pageData,
    isLoading,
  } = useApiQuery<EntityListResponse>(
    async () => {
      const params: Record<string, unknown> = { page: page + 1, page_size: PAGE_SIZE };
      const effectiveScope = scopeUserId || ownUserId;
      if (!isWildcard(effectiveScope) && effectiveScope) {
        params.scope_user_id = effectiveScope;
      }
      if (!isWildcard(typeFilter)) {
        params.type = typeFilter;
      }
      if (showUnownedOnly) {
        params.unowned_only = true;
      }
      const res = await api.get<EntityListResponse>(ENTITY_ENDPOINTS.BASE, {
        params,
      });
      return res.data ?? EMPTY_PAGE_RESULTS;
    },
    {
      enabled: !!ownUserId,
      errorToast: "Failed to load entities",
      initialData: EMPTY_PAGE_RESULTS,
      deps: [page, scopeUserId, ownUserId, refreshNonce, typeFilter, showUnownedOnly],
    },
  );

  const entities = pageData?.results ?? [];
  const total = pageData?.count ?? 0;

  const handleDelete = async () => {
    if (!entityToDelete) return;
    try {
      await api.delete(
        ENTITY_ENDPOINTS.BY_ID(entityToDelete.type, entityToDelete.id),
        { params: { parent_id: entityToDelete.parent?.id } },
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
      width: 200,
      render: (value: string) => (
        <span className="font-mono text-sm truncate">{value}</span>
      ),
    },
    {
      key: "name" as keyof Entity,
      label: "Name",
      width: 140,
      render: (value: string | null) => (
        <span className="text-sm">{orDash(value)}</span>
      ),
    },
    {
      key: "parent" as keyof Entity,
      label: "Parent",
      width: 160,
      render: (_value: Entity["parent"], row: Entity) =>
        row.parent ? (
          <span className="text-sm">
            {row.parent.id}{" "}
            <Badge variant="outline" className="text-[10px] px-1 py-0">
              {row.parent.type}
            </Badge>
          </span>
        ) : (
          <span className="text-sm text-onSurface-default-tertiary">--</span>
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
      key: "permission" as keyof Entity,
      label: "Permission",
      width: 100,
      render: (value: Entity["permission"]) =>
        value ? (
          <Badge
            variant="outline"
            className={PERMISSION_BADGES[value] ?? ""}
          >
            {value}
          </Badge>
        ) : (
          <span className="text-sm text-onSurface-default-tertiary">--</span>
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
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="size-4 mr-1" /> Create Entity
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {isAdmin && (
          <>
            <EntityScopeSelect
              value={scopeUserId || ownUserId}
              onChange={setScopeUserId}
              ownUserId={ownUserId}
              className="w-72"
            />
            <Label className="flex items-center gap-2 text-sm font-normal text-onSurface-default-tertiary">
              <Checkbox
                checked={showUnownedOnly}
                disabled={!isWildcard(scopeUserId || ownUserId)}
                onCheckedChange={(c) => setShowUnownedOnly(c === true)}
              />
              Unowned only
            </Label>
            <Separator orientation="vertical" className="h-5" />
          </>
        )}
        <Label className="text-sm font-normal text-onSurface-default-tertiary">Type:</Label>
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          <SelectTrigger className="w-28">
            <SelectValue placeholder="All" />
          </SelectTrigger>
          <SelectContent>
            {TYPE_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {isLoading && entities.length === 0 ? (
        <TableSkeleton rows={5} columns={9} />
      ) : entities.length === 0 ? (
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
              data={entities}
              columns={columns}
              getRowKey={(row) => `${row.type}:${row.id}`}
            />
          </Card>
          <PaginationBar
              page={page}
              total={total}
              pageSize={PAGE_SIZE}
              onPageChange={setPage}
            />
        </>
      )}

      <CreateEntityDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        isAdmin={isAdmin}
        onCreated={reload}
      />

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
