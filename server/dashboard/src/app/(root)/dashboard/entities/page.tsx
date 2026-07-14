"use client";

import { useMemo, useState } from "react";
import {
  ArrowRightLeft,
  Loader2,
  Plus,
  RefreshCw,
  Trash2,
  Users,
} from "lucide-react";
import { format } from "date-fns";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { EmptyState } from "@/components/self-hosted/empty-state";
import DeleteConfirmationModal from "@/components/ui/delete-confirmation-modal";
import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { ENTITY_ENDPOINTS, USER_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import { useApiQuery } from "@/hooks/use-api-query";
import { useAuth } from "@/hooks/use-auth";
import type { Entity, EntityType, User, UserInfo } from "@/types/api";
import ManagePermissionsSheet from "./manage-permissions-sheet";

const ENTITY_TYPES: EntityType[] = ["user", "agent", "app", "run"];

export default function EntitiesPage() {
  const { isAdmin } = useAuth();
  const [createOpen, setCreateOpen] = useState(false);
  const [newType, setNewType] = useState<EntityType>("user");
  const [newId, setNewId] = useState("");
  const [manageEntity, setManageEntity] = useState<Entity | null>(null);
  const [transferEntity, setTransferEntity] = useState<Entity | null>(null);
  const [transferUserId, setTransferUserId] = useState("");
  const [entityToDelete, setEntityToDelete] = useState<Entity | null>(null);
  const [showUnownedOnly, setShowUnownedOnly] = useState(false);
  const [newOwnerId, setNewOwnerId] = useState("");
  const [counts, setCounts] = useState<
    Record<string, { loading: boolean; value: number | null }>
  >({});

  const {
    data: entities = [],
    isLoading,
    refetch,
  } = useApiQuery<Entity[]>(
    async () => {
      const res = await api.get<Entity[]>(ENTITY_ENDPOINTS.BASE);
      return res.data ?? [];
    },
    { errorToast: "Failed to load entities", initialData: [] },
  );

  // Admin-only user list for the grant/transfer picker (non-admins keep UUID input).
  const { data: users = [] } = useApiQuery<User[]>(
    async () => {
      const res = await api.get(USER_ENDPOINTS.BASE, {
        params: { page: 1, page_size: 200 },
      });
      return res.data?.results ?? [];
    },
    { errorToast: "Failed to load users", initialData: [], enabled: isAdmin },
  );

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

  const handleCreate = async () => {
    try {
      const body: Record<string, unknown> = {
        type: newType,
        id: newId.trim(),
      };
      if (newType === "app" && newOwnerId.trim()) {
        body.owner_user_id = newOwnerId.trim();
      }
      await api.post(ENTITY_ENDPOINTS.BASE, body);
      toast({ title: "Entity created", variant: "success" });
      setCreateOpen(false);
      setNewId("");
      setNewType("user");
      setNewOwnerId("");
      setCounts({});
      void refetch();
    } catch (error) {
      toast({
        title: "Failed to create entity",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  const handleTransfer = async () => {
    if (!transferEntity) return;
    try {
      await api.post(
        ENTITY_ENDPOINTS.TRANSFER_OWNER(transferEntity.type, transferEntity.id),
        { user_id: transferUserId.trim() },
      );
      toast({ title: "Ownership transferred", variant: "success" });
      setTransferEntity(null);
      setTransferUserId("");
      setCounts({});
      void refetch();
    } catch (error) {
      toast({
        title: "Failed to transfer ownership",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  const handleDelete = async () => {
    if (!entityToDelete) return;
    try {
      await api.delete(
        ENTITY_ENDPOINTS.BY_ID(entityToDelete.type, entityToDelete.id),
        { params: { namespace: entityToDelete.parent?.id } },
      );
      toast({ title: "Entity deleted", variant: "success" });
      setEntityToDelete(null);
      setCounts({});
      void refetch();
    } catch (error) {
      toast({
        title: "Failed to delete entity",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  const fetchCount = async (entity: Entity) => {
    const key = `${entity.type}:${entity.id}`;
    setCounts((prev) => ({ ...prev, [key]: { loading: true, value: null } }));
    try {
      const res = await api.get<{ total_memories: number }>(
        ENTITY_ENDPOINTS.COUNT(entity.type, entity.id),
        { params: { namespace: entity.parent?.id } },
      );
      setCounts((prev) => ({
        ...prev,
        [key]: { loading: false, value: res.data.total_memories },
      }));
    } catch {
      setCounts((prev) => ({
        ...prev,
        [key]: { loading: false, value: null },
      }));
    }
  };

  const canManage = (e: Entity) => isAdmin || e.is_owner;
  const canGrantOrTransfer = (e: Entity) => e.type === "user" || e.type === "app";

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
      align: "right" as const,
      render: (_value: string, row: Entity) => {
        const key = `${row.type}:${row.id}`;
        const info = counts[key];
        if (info?.loading) {
          return <Loader2 className="size-4 animate-spin inline" />;
        }
        if (info?.value != null) {
          return (
            <button
              className="text-sm hover:underline inline-flex items-center gap-1"
              title="Click to refresh"
              onClick={() => fetchCount(row)}
            >
              {info.value}
              <RefreshCw className="size-3" />
            </button>
          );
        }
        return (
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            title="Count memories"
            onClick={() => fetchCount(row)}
          >
            <RefreshCw className="size-3.5" />
          </Button>
        );
      },
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
        <div className="flex gap-1 justify-end">
          {canManage(row) && (
            <>
              {canGrantOrTransfer(row) && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7"
                  title="Manage permissions"
                  onClick={() => setManageEntity(row)}
                >
                  <Users className="size-3.5" />
                </Button>
              )}
              {canGrantOrTransfer(row) && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-7"
                  title={row.owner ? "Transfer owner" : "Assign owner"}
                  onClick={() => {
                    setTransferEntity(row);
                    setTransferUserId("");
                  }}
                >
                  <ArrowRightLeft className="size-3.5" />
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon"
                className="size-7"
                title="Delete"
                onClick={() => setEntityToDelete(row)}
              >
                <Trash2 className="size-3.5 text-onSurface-danger-primary" />
              </Button>
            </>
          )}
        </div>
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
              <label className="flex items-center gap-2 text-sm text-onSurface-default-tertiary">
                <input
                  type="checkbox"
                  className="size-4"
                  checked={showUnownedOnly}
                  onChange={(e) => setShowUnownedOnly(e.target.checked)}
                />
                Unowned only
              </label>
            </>
          )}
          <Dialog open={createOpen} onOpenChange={setCreateOpen}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="size-4 mr-1" /> Create Entity
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Entity</DialogTitle>
              </DialogHeader>
              <div className="space-y-4 mt-2">
                <div className="space-y-2">
                  <Label>Type</Label>
                  <Select
                    value={newType}
                    onValueChange={(v) => setNewType(v as EntityType)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ENTITY_TYPES.map((t) => (
                        <SelectItem key={t} value={t} className="capitalize">
                          {t}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {newType === "agent" || newType === "run" ? (
                    <p className="text-xs text-onSurface-default-tertiary">
                      {newType} entities are auto-created on first write and cannot be manually created.
                    </p>
                  ) : null}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="entity-id">ID</Label>
                  <Input
                    id="entity-id"
                    value={newId}
                    onChange={(e) => setNewId(e.target.value)}
                    placeholder={
                      newType === "user"
                        ? "e.g. alice or a UUID"
                        : newType === "app"
                          ? "e.g. my-repo"
                          : "auto-created"
                    }
                    disabled={newType === "agent" || newType === "run"}
                  />
                  {newType === "user" ? (
                    <p className="text-xs text-onSurface-default-tertiary">
                      UUID entity_ids are reserved for the user with that ID. Cannot contain <code>:</code>.
                    </p>
                  ) : newType === "app" ? (
                    <p className="text-xs text-onSurface-default-tertiary">
                      App entities must be created by an admin. Requires an owner.
                    </p>
                  ) : null}
                </div>
                {newType === "app" && isAdmin && (
                  <div className="space-y-2">
                    <Label htmlFor="owner-id">Owner user ID</Label>
                    <Select value={newOwnerId} onValueChange={setNewOwnerId}>
                      <SelectTrigger id="owner-id">
                        <SelectValue placeholder="Select a user" />
                      </SelectTrigger>
                      <SelectContent>
                        {users.map((u) => (
                          <SelectItem key={u.id} value={u.id}>
                            {u.name} ({u.email})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                <Button
                  onClick={handleCreate}
                  disabled={
                    !newId.trim() ||
                    (newType === "agent" || newType === "run") ||
                    (newType === "app" && !isAdmin) ||
                    (newType === "app" && !newOwnerId.trim())
                  }
                  className="w-full"
                >
                  Create
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {isLoading ? (
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
        <Card className="border-memBorder-primary overflow-hidden">
          <DataTable
            data={visibleEntities}
            columns={columns}
            getRowKey={(row) => `${row.type}:${row.id}`}
          />
        </Card>
      )}

      <ManagePermissionsSheet
        entity={manageEntity}
        open={!!manageEntity}
        onClose={() => setManageEntity(null)}
        users={users}
        isAdmin={isAdmin}
      />

      {/* Transfer / assign owner dialog */}
      <Dialog
        open={!!transferEntity}
        onOpenChange={(o) => {
          if (!o) setTransferEntity(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {transferEntity?.owner ? "Transfer Owner" : "Assign Owner"}
            </DialogTitle>
          </DialogHeader>
          {transferEntity && (
            <div className="space-y-4 mt-2">
              <p className="text-sm text-onSurface-default-tertiary">
                {transferEntity.owner
                  ? "The previous owner keeps an explicit admin grant by default."
                  : "Claim this unowned namespace and assign it to a dashboard user."}
              </p>
              <div className="space-y-2">
                <Label htmlFor="transfer-user">
                  Target user{isAdmin ? "" : " UUID"}
                </Label>
                {isAdmin ? (
                  <Select
                    value={transferUserId}
                    onValueChange={setTransferUserId}
                  >
                    <SelectTrigger id="transfer-user">
                      <SelectValue placeholder="Select a user" />
                    </SelectTrigger>
                    <SelectContent>
                      {users.map((u) => (
                        <SelectItem key={u.id} value={u.id}>
                          {u.name} ({u.email})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="transfer-user"
                    value={transferUserId}
                    onChange={(e) => setTransferUserId(e.target.value)}
                    placeholder="dashboard user id"
                  />
                )}
              </div>
              <DialogFooter>
                <DialogClose asChild>
                  <Button variant="outline">Cancel</Button>
                </DialogClose>
                <Button
                  onClick={handleTransfer}
                  disabled={!transferUserId.trim()}
                >
                  {transferEntity.owner ? "Transfer" : "Assign"}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

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
