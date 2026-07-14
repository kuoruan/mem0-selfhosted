"use client";

import { useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import { api } from "@/utils/api";
import { ENTITY_ENDPOINTS } from "@/utils/api-endpoints";
import { getErrorMessage } from "@/lib/error-message";
import type { Entity } from "@/types/api";

interface EntityMemoryCountProps {
  entity: Entity;
}

export default function EntityMemoryCount({ entity }: EntityMemoryCountProps) {
  const [info, setInfo] = useState<{ loading: boolean; value: number | null }>({
    loading: false,
    value: null,
  });

  const fetchCount = async () => {
    setInfo({ loading: true, value: null });
    try {
      const res = await api.get<{ total_memories: number }>(
        ENTITY_ENDPOINTS.COUNT(entity.type, entity.id),
        { params: { namespace: entity.parent?.id } },
      );
      setInfo({ loading: false, value: res.data.total_memories });
    } catch (error) {
      setInfo({ loading: false, value: null });
      toast({
        title: "Failed to load memory count",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    }
  };

  if (info.loading) {
    return <Loader2 className="size-4 animate-spin inline" />;
  }
  if (info.value != null) {
    return (
      <button
        className="text-sm hover:underline inline-flex items-center gap-1"
        title="Click to refresh"
        onClick={fetchCount}
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
      onClick={fetchCount}
    >
      <RefreshCw className="size-3.5" />
    </Button>
  );
}
