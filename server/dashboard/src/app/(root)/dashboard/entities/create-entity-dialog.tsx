"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import type { EntityType } from "@/types/api";
import UserSelect from "@/components/shared/user-select";

const ENTITY_TYPES: EntityType[] = ["user", "agent", "app", "run"];

interface CreateEntityDialogProps {
  open: boolean;
  onClose: () => void;
  isAdmin: boolean;
  onCreated: () => void;
}

export default function CreateEntityDialog({
  open,
  onClose,
  isAdmin,
  onCreated,
}: CreateEntityDialogProps) {
  const [newType, setNewType] = useState<EntityType>("user");
  const [newId, setNewId] = useState("");
  const [newOwnerId, setNewOwnerId] = useState("");

  // Reset the form each time the dialog opens so stale input from a previous
  // cancelled session never carries over.
  useEffect(() => {
    if (!open) return;
    setNewType("user");
    setNewId("");
    setNewOwnerId("");
  }, [open]);

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
      onClose();
      onCreated();
    } catch (error) {
      toast({
        title: "Failed to create entity",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
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
                {newType} entities are auto-created on first write and cannot be
                manually created.
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
                UUID entity_ids are reserved for the user with that ID. Cannot
                contain <code>:</code>.
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
              <UserSelect
                id="owner-id"
                value={newOwnerId}
                onChange={setNewOwnerId}
                isAdmin={isAdmin}
              />
            </div>
          )}
          <Button
            onClick={handleCreate}
            disabled={
              !newId.trim() ||
              newType === "agent" ||
              newType === "run" ||
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
  );
}
