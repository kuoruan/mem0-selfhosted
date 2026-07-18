"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
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
import UserSelect from "@/components/shared/user-select";

interface TransferOwnerDialogProps {
  entity: Entity | null;
  open: boolean;
  onClose: () => void;
  isAdmin: boolean;
  onTransferred: () => void;
}

export default function TransferOwnerDialog({
  entity,
  open,
  onClose,
  isAdmin,
  onTransferred,
}: TransferOwnerDialogProps) {
  const [newOwnerId, setNewOwnerId] = useState("");

  useEffect(() => {
    setNewOwnerId("");
  }, [entity]);

  const handleTransfer = async () => {
    if (!entity) return;
    try {
      await api.post(ENTITY_ENDPOINTS.TRANSFER_OWNER(entity.type, entity.id), {
        owner_id: newOwnerId.trim(),
      });
      toast({ title: "Ownership transferred", variant: "success" });
      onClose();
      onTransferred();
    } catch (error) {
      toast({
        title: "Failed to transfer ownership",
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
          <DialogTitle>
            {entity?.owner ? "Transfer Owner" : "Assign Owner"}
          </DialogTitle>
        </DialogHeader>
        {entity && (
          <div className="space-y-4 mt-2">
            <p className="text-sm text-onSurface-default-tertiary">
              {entity.owner
                ? "The previous owner keeps an explicit admin grant by default."
                : "Claim this unowned namespace and assign it to a dashboard user."}
            </p>
            <div className="space-y-2">
              <Label htmlFor="transfer-user">
                Target user{isAdmin ? "" : " UUID"}
              </Label>
              <UserSelect
                id="transfer-user"
                value={newOwnerId}
                onChange={setNewOwnerId}
                isAdmin={isAdmin}
                inputPlaceholder="dashboard user id"
              />
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">Cancel</Button>
              </DialogClose>
              <Button
                onClick={handleTransfer}
                disabled={!newOwnerId.trim()}
              >
                {entity.owner ? "Transfer" : "Assign"}
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
