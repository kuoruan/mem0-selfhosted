"use client";

import { ArrowRightLeft, Pencil, Trash2, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Entity } from "@/types/api";

const canManage = (e: Entity, isAdmin: boolean) => isAdmin || e.is_owner;
const canGrantOrTransfer = (e: Entity) => e.type === "user" || e.type === "app";

interface EntityActionsProps {
  entity: Entity;
  isAdmin: boolean;
  onManage: (entity: Entity) => void;
  onTransfer: (entity: Entity) => void;
  onEdit: (entity: Entity) => void;
  onDelete: (entity: Entity) => void;
}

export default function EntityActions({
  entity,
  isAdmin,
  onManage,
  onTransfer,
  onEdit,
  onDelete,
}: EntityActionsProps) {
  if (!canManage(entity, isAdmin)) return null;

  const grantable = canGrantOrTransfer(entity);

  return (
    <div className="flex gap-1 justify-end">
      {grantable && (
        <Button
          variant="ghost"
          size="icon"
          className="size-7"
          title="Manage permissions"
          onClick={() => onManage(entity)}
        >
          <Users className="size-3.5" />
        </Button>
      )}
      {grantable && (
        <Button
          variant="ghost"
          size="icon"
          className="size-7"
          title={entity.owner ? "Transfer owner" : "Assign owner"}
          onClick={() => onTransfer(entity)}
        >
          <ArrowRightLeft className="size-3.5" />
        </Button>
      )}
      <Button
        variant="ghost"
        size="icon"
        className="size-7"
        title="Edit name"
        onClick={() => onEdit(entity)}
      >
        <Pencil className="size-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="size-7"
        title="Delete"
        onClick={() => onDelete(entity)}
      >
        <Trash2 className="size-3.5 text-onSurface-danger-primary" />
      </Button>
    </div>
  );
}
