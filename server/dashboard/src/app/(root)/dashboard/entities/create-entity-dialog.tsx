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
import { isUuidString } from "@/lib/validators";
import type { EntityType } from "@/types/api";
import UserSelect from "@/components/shared/user-select";
import { useAuth } from "@/hooks/use-auth";

const ENTITY_TYPES: EntityType[] = ["user", "app"];

interface CreateEntityForm {
  type: EntityType;
  id: string;
  name: string;
  ownerId: string;
}

const EMPTY_FORM: CreateEntityForm = {
  type: "user",
  id: "",
  name: "",
  ownerId: "",
};

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
  const { user } = useAuth();
  const [form, setForm] = useState<CreateEntityForm>(EMPTY_FORM);

  // Reset the form each time the dialog opens so stale input from a previous
  // cancelled session never carries over.
  useEffect(() => {
    if (!open) return;
    setForm(EMPTY_FORM);
  }, [open]);

  const updateField = <K extends keyof CreateEntityForm>(
    key: K,
    value: CreateEntityForm[K],
  ) => setForm((prev) => ({ ...prev, [key]: value }));

  const values = {
    type: form.type,
    id: form.id.trim(),
    name: form.name.trim(),
    ownerId: form.ownerId.trim(),
  };
  const isUuid = values.type === "user" && isUuidString(values.id);
  const isOwnUuid = isUuid && values.id === user?.id;

  const handleCreate = async () => {
    try {
      const body: Record<string, unknown> = {
        type: values.type,
        id: values.id,
      };
      if (values.name) body.name = values.name;
      if (values.type === "app" && values.ownerId) {
        body.owner_id = values.ownerId;
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
              value={form.type}
              onValueChange={(v) => updateField("type", v as EntityType)}
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
          </div>
          <div className="space-y-2">
            <Label htmlFor="entity-id">ID</Label>
            <Input
              id="entity-id"
              value={form.id}
              onChange={(e) => updateField("id", e.target.value)}
              placeholder={form.type === "user" ? "e.g. alice" : "e.g. my-repo"}
            />
            {form.type === "user" ? (
              <p className="text-xs text-onSurface-default-tertiary">
                Cannot contain <code>:</code>. UUID user_ids cannot be created
                manually; your own UUID is already yours by default.
              </p>
            ) : form.type === "app" ? (
              <p className="text-xs text-onSurface-default-tertiary">
                App entities must be created by an admin. Requires an owner.
              </p>
            ) : null}
            {isOwnUuid && (
              <p className="text-xs text-onSurface-info-primary">
                This is your own user_id — already yours by default; no entity
                needs to be created.
              </p>
            )}
            {isUuid && !isOwnUuid && (
              <p className="text-xs text-onSurface-danger-primary">
                UUID user_ids cannot be created manually. Use a non-UUID
                identifier (e.g. &#39;alice&#39;).
              </p>
            )}
          </div>
          <div className="space-y-2">
            <Label htmlFor="entity-name">Name (optional)</Label>
            <Input
              id="entity-name"
              value={form.name}
              onChange={(e) => updateField("name", e.target.value)}
              placeholder="Display name"
              maxLength={255}
            />
          </div>
          {form.type === "app" && isAdmin && (
            <div className="space-y-2">
              <Label htmlFor="owner-id">Owner user ID</Label>
              <UserSelect
                id="owner-id"
                value={form.ownerId}
                onChange={(v) => updateField("ownerId", v)}
                isAdmin={isAdmin}
              />
            </div>
          )}
          <Button
            onClick={handleCreate}
            disabled={
              !values.id ||
              isUuid ||
              (values.type === "app" && !(isAdmin && !!values.ownerId))
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
