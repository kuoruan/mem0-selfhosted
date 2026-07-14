"use client";

import { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { ENTITY_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import type {
  Entity,
  EntityPermission,
  EntityPermissionLevel,
  User,
} from "@/types/api";

const PERMISSION_LEVELS: EntityPermissionLevel[] = ["read", "write", "admin"];

export default function ManagePermissionsSheet({
  entity,
  open,
  onClose,
  users,
  isAdmin,
}: {
  entity: Entity | null;
  open: boolean;
  onClose: () => void;
  users: User[];
  isAdmin: boolean;
}) {
  const [permissions, setPermissions] = useState<EntityPermission[]>([]);
  const [loading, setLoading] = useState(false);
  const [grantUserId, setGrantUserId] = useState("");
  const [grantLevel, setGrantLevel] = useState<EntityPermissionLevel>("read");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open || !entity) return;
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      try {
        const res = await api.get<EntityPermission[]>(
          ENTITY_ENDPOINTS.PERMISSIONS(entity.type, entity.id),
        );
        if (!cancelled) setPermissions(res.data ?? []);
      } catch (error) {
        if (!cancelled) {
          toast({
            title: "Failed to load permissions",
            description: getErrorMessage(error),
            variant: "destructive",
          });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [open, entity]);

  const handleGrant = async () => {
    if (!entity) return;
    setSubmitting(true);
    try {
      await api.post(ENTITY_ENDPOINTS.PERMISSIONS(entity.type, entity.id), {
        user_id: grantUserId.trim(),
        permission: grantLevel,
      });
      setGrantUserId("");
      setGrantLevel("read");
      // reload
      const res = await api.get<EntityPermission[]>(
        ENTITY_ENDPOINTS.PERMISSIONS(entity.type, entity.id),
      );
      setPermissions(res.data ?? []);
    } catch (error) {
      toast({
        title: "Failed to grant permission",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  const handleRevoke = async (perm: EntityPermission) => {
    if (!entity) return;
    try {
      await api.delete(
        ENTITY_ENDPOINTS.PERMISSION_BY_USER(
          entity.type,
          entity.id,
          perm.user.id,
        ),
      );
      setPermissions((prev) => prev.filter((p) => p.user.id !== perm.user.id));
    } catch (error) {
      toast({
        title: "Failed to revoke permission",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  return (
    <Sheet
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <SheetContent className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Manage permissions</SheetTitle>
          <SheetDescription>
            {entity ? `${entity.type}/${entity.id}` : ""}
          </SheetDescription>
        </SheetHeader>
        <div className="mt-4 space-y-4">
          <div className="space-y-2">
            <Label className="text-xs">Grant access</Label>
            <div className="flex gap-2">
              {isAdmin ? (
                <Select value={grantUserId} onValueChange={setGrantUserId}>
                  <SelectTrigger className="flex-1">
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
                  placeholder="User UUID"
                  value={grantUserId}
                  onChange={(e) => setGrantUserId(e.target.value)}
                  className="flex-1"
                />
              )}
              <Select
                value={grantLevel}
                onValueChange={(v) => setGrantLevel(v as EntityPermissionLevel)}
              >
                <SelectTrigger className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PERMISSION_LEVELS.map((l) => (
                    <SelectItem key={l} value={l} className="capitalize">
                      {l}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                size="sm"
                onClick={handleGrant}
                disabled={!grantUserId.trim() || submitting}
              >
                Add
              </Button>
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Shared with</Label>
            {loading ? (
              <p className="text-sm text-onSurface-default-tertiary">
                Loading…
              </p>
            ) : permissions.length === 0 ? (
              <p className="text-sm text-onSurface-default-tertiary">
                Not shared with anyone.
              </p>
            ) : (
              <div className="space-y-1">
                {permissions.map((p) => (
                  <div
                    key={p.id}
                    className="flex items-center justify-between gap-2 py-1"
                  >
                    <div className="min-w-0">
                      <p className="text-xs truncate">
                        {p.user.name}
                        <span className="text-onSurface-default-tertiary">
                          {" "}
                          ({p.user.email})
                        </span>
                      </p>
                      <p className="text-xs text-onSurface-default-tertiary capitalize">
                        {p.permission}
                        {p.granted_by
                          ? ` · by ${p.granted_by.name}`
                          : " · system"}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7"
                      title="Revoke"
                      onClick={() => handleRevoke(p)}
                    >
                      <Trash2 className="size-3.5 text-onSurface-danger-primary" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
