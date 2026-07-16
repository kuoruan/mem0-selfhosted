"use client";

import { useEffect, useState } from "react";
import { Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { ENTITY_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import type { Entity } from "@/types/api";

interface EditEntityNameDialogProps {
  entity: Entity | null;
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

export default function EditEntityNameDialog({
  entity,
  open,
  onClose,
  onSaved,
}: EditEntityNameDialogProps) {
  const [name, setName] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    setName(entity?.name ?? "");
  }, [entity]);

  const handleSave = async () => {
    if (!entity || isSaving) return;
    setIsSaving(true);
    try {
      await api.patch(
        ENTITY_ENDPOINTS.UPDATE(entity.type, entity.id),
        { name: name.trim() },
        { params: { namespace: entity.parent?.id } },
      );
      toast({ title: "Entity name updated", variant: "success" });
      onClose();
      onSaved();
    } catch (error) {
      toast({
        title: "Failed to update entity name",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
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
          <DialogTitle>Edit Entity Name</DialogTitle>
        </DialogHeader>
        {entity && (
          <div className="space-y-4 mt-2">
            <div className="space-y-2">
              <Label htmlFor="edit-entity-id">Entity ID</Label>
              <Input
                id="edit-entity-id"
                value={`${entity.type}/${entity.id}`}
                disabled
                className="font-mono"
              />
              <p className="text-xs text-onSurface-default-tertiary">
                The entity ID cannot be changed.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-entity-name">Name</Label>
              <Input
                id="edit-entity-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Display name"
                maxLength={255}
              />
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">Cancel</Button>
              </DialogClose>
              <Button onClick={handleSave} disabled={isSaving}>
                {isSaving ? "Saving..." : "Save"}
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
