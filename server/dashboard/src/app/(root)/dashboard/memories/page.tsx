"use client";

import { useEffect, useState } from "react";
import { RefreshCw, Trash2 } from "lucide-react";
import { format } from "date-fns";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { DataTable } from "@/components/shared/data-table";
import { TableSkeleton } from "@/components/shared/table-skeleton";
import { EmptyState } from "@/components/self-hosted/empty-state";
import DeleteConfirmationModal from "@/components/ui/delete-confirmation-modal";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { toast } from "@/components/ui/use-toast";
import { getErrorMessage } from "@/lib/error-message";
import { api } from "@/utils/api";
import { MEMORY_ENDPOINTS } from "@/utils/api-endpoints";
import { isWildcard, orDash } from "@/utils/helpers";
import { useAuth } from "@/hooks/use-auth";
import { useApiQuery } from "@/hooks/use-api-query";
import MemoryUserSelect from "@/components/shared/memory-user-select";
import MemoryAppSelect from "@/components/shared/memory-app-select";
import { CategoriesDisplay } from "@/components/ui/categories-display";
import { Memory } from "@/types/api";

const PAGE_SIZE = 20;
// Keep in sync with ALL_MEMORIES_LIMIT in server/main.py.
const MEMORY_FETCH_LIMIT = 1000;

export default function MemoriesPage() {
  const { user } = useAuth();
  const ownUserId = user?.id ?? "";
  const [selectedUserId, setSelectedUserId] = useState(ownUserId);
  const [appId, setAppId] = useState("");
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [memoryToDelete, setMemoryToDelete] = useState<Memory | null>(null);
  const [page, setPage] = useState(0);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";

  useEffect(() => {
    if (ownUserId)
      setSelectedUserId((prev) => (prev === "" ? ownUserId : prev));
  }, [ownUserId]);

  const {
    data: memories = [],
    isLoading,
    refetch,
  } = useApiQuery<Memory[]>(
    async () => {
      const params: Record<string, unknown> = {
        top_k: MEMORY_FETCH_LIMIT,
        user_id: selectedUserId,
      };
      if (appId) {
        params.app_id = appId;
      }
      const res = await api.get(MEMORY_ENDPOINTS.BASE, { params });
      const raw = res.data?.results ?? res.data ?? [];
      return Array.isArray(raw) ? raw : [];
    },
    {
      errorToast: "Failed to load memories",
      initialData: [],
      deps: [selectedUserId, appId],
    },
  );

  const handleUserChange = (id: string) => {
    setSelectedUserId(id);
    setPage(0);
  };

  const selectApp = (id: string) => {
    setAppId(id);
    setPage(0);
  };

  const totalPages = Math.ceil(memories.length / PAGE_SIZE);
  const paginatedMemories = memories.slice(
    page * PAGE_SIZE,
    (page + 1) * PAGE_SIZE,
  );

  const handleDelete = async () => {
    if (!memoryToDelete) return;
    try {
      await api.delete(MEMORY_ENDPOINTS.BY_ID(memoryToDelete.id));
      toast({ title: "Memory deleted", variant: "success" });
      if (selectedMemory?.id === memoryToDelete.id) setSelectedMemory(null);
      setMemoryToDelete(null);
      void refetch();
    } catch (error) {
      toast({
        title: "Failed to delete memory",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  const columns = [
    {
      key: "memory" as keyof Memory,
      label: "Content",
      width: 400,
      render: (value: string) => (
        <span className="line-clamp-2 text-sm">{value}</span>
      ),
    },
    {
      key: "user_id" as keyof Memory,
      label: "User",
      width: 100,
      render: orDash,
    },
    { key: "app_id" as keyof Memory, label: "App", width: 100, render: orDash },
    {
      key: "agent_id" as keyof Memory,
      label: "Agent",
      width: 100,
      render: orDash,
    },
    {
      key: "run_id" as keyof Memory,
      label: "Run",
      width: 100,
      render: (value: string | undefined) =>
        value ? (
          <span className="text-xs font-mono block truncate" title={value}>
            {value}
          </span>
        ) : (
          "--"
        ),
    },
    {
      key: "metadata" as keyof Memory,
      label: "Categories",
      width: 120,
      render: (_value: Memory["metadata"], row: Memory) => (
        <CategoriesDisplay categories={row.metadata?.categories} />
      ),
    },
    {
      key: "created_at" as keyof Memory,
      label: "Created",
      width: 120,
      render: (value: string) =>
        value ? format(new Date(value), "MMM d, yyyy") : "--",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold font-fustat">Memories</h1>
        <Button
          variant="ghost"
          size="smIcon"
          onClick={() => refetch()}
          disabled={isLoading}
          title="Refresh memories"
        >
          <RefreshCw className={`size-4 ${isLoading ? "animate-spin" : ""}`} />
        </Button>
        <MemoryUserSelect
          value={selectedUserId}
          onChange={handleUserChange}
          ownUserId={ownUserId}
          className="w-72"
        />
        <MemoryAppSelect value={appId} onChange={selectApp} className="w-72" />
      </div>

      {isLoading ? (
        <TableSkeleton rows={5} columns={7} />
      ) : memories.length === 0 ? (
        <EmptyState
          title="No memories yet"
          description="Create your first memory by sending a POST /memories request."
        >
          <pre className="text-xs text-left bg-surface-default-secondary p-3 rounded font-mono overflow-x-auto mt-3 max-w-lg">
            {`curl -X POST ${apiUrl}/memories \\
  -H "X-API-Key: <your-key>" \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "I like hiking"}], "user_id": "${isWildcard(selectedUserId) ? ownUserId : selectedUserId || "alice"}"`}
          </pre>
        </EmptyState>
      ) : (
        <>
          <Card className="border-memBorder-primary overflow-hidden">
            <DataTable
              data={paginatedMemories}
              columns={columns}
              getRowKey={(row) => row.id}
              onRowClick={(row) => setSelectedMemory(row)}
              getRowClassName={(row) =>
                selectedMemory?.id === row.id
                  ? "bg-surface-default-tertiary"
                  : undefined
              }
            />
          </Card>
          {totalPages > 1 && (
            <div className="flex items-center justify-between text-sm text-onSurface-default-tertiary">
              <span>
                {page * PAGE_SIZE + 1}–
                {Math.min((page + 1) * PAGE_SIZE, memories.length)} of{" "}
                {memories.length}
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page >= totalPages - 1}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </>
      )}

      <Sheet
        open={!!selectedMemory}
        onOpenChange={(open) => {
          if (!open) setSelectedMemory(null);
        }}
      >
        <SheetContent className="sm:max-w-md flex flex-col">
          <SheetHeader className="shrink-0">
            <SheetTitle>Memory Detail</SheetTitle>
            <SheetDescription className="sr-only">
              View memory content and metadata
            </SheetDescription>
          </SheetHeader>
          {selectedMemory && (
            <div className="flex-1 min-h-0 mt-2 space-y-4 overflow-y-auto">
              <div className="space-y-1">
                <Label className="text-xs text-onSurface-default-tertiary">
                  Content
                </Label>
                <p className="text-sm">{selectedMemory.memory}</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <Label className="text-xs text-onSurface-default-tertiary">
                    ID
                  </Label>
                  <p className="text-xs font-mono break-all">
                    {selectedMemory.id}
                  </p>
                </div>
                {selectedMemory.user_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      User
                    </Label>
                    <p className="text-sm">{selectedMemory.user_id}</p>
                  </div>
                )}
                {selectedMemory.app_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      App
                    </Label>
                    <p className="text-sm">{selectedMemory.app_id}</p>
                  </div>
                )}
                {selectedMemory.agent_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Agent
                    </Label>
                    <p className="text-sm">{selectedMemory.agent_id}</p>
                  </div>
                )}
                {selectedMemory.run_id && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Run
                    </Label>
                    <p className="text-sm">{selectedMemory.run_id}</p>
                  </div>
                )}
                {selectedMemory.hash && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Hash
                    </Label>
                    <p className="text-xs font-mono break-all">
                      {selectedMemory.hash}
                    </p>
                  </div>
                )}
                {selectedMemory.created_at && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Created
                    </Label>
                    <p className="text-sm">
                      {format(
                        new Date(selectedMemory.created_at),
                        "MMM d, yyyy, h:mm:ss a",
                      )}
                    </p>
                  </div>
                )}
                {selectedMemory.updated_at && (
                  <div className="space-y-1">
                    <Label className="text-xs text-onSurface-default-tertiary">
                      Updated
                    </Label>
                    <p className="text-sm">
                      {format(
                        new Date(selectedMemory.updated_at),
                        "MMM d, yyyy, h:mm:ss a",
                      )}
                    </p>
                  </div>
                )}
              </div>
              {selectedMemory.metadata && (
                <div className="space-y-1">
                  <Label className="text-xs text-onSurface-default-tertiary">
                    Metadata
                  </Label>
                  <pre className="text-xs font-mono overflow-auto max-h-60 bg-surface-default-secondary p-2 rounded">
                    {JSON.stringify(selectedMemory.metadata, null, 2)}
                  </pre>
                </div>
              )}
              <Button
                variant="outline"
                size="sm"
                className="text-onSurface-danger-primary"
                onClick={() => setMemoryToDelete(selectedMemory)}
              >
                <Trash2 className="size-3.5 mr-1" />
                Delete memory
              </Button>
            </div>
          )}
        </SheetContent>
      </Sheet>

      <DeleteConfirmationModal
        isOpen={!!memoryToDelete}
        onClose={() => setMemoryToDelete(null)}
        onConfirm={handleDelete}
        title="Delete memory"
        description="This memory will be permanently removed. This cannot be undone."
        itemName={memoryToDelete?.id ?? ""}
        confirmButtonText="Delete"
      />
    </div>
  );
}
