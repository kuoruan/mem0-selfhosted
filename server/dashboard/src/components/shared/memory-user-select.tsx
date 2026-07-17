"use client";

import { useMemo } from "react";
import Combobox, { type ComboboxOption } from "@/components/shared/combobox";
import { useEntityUsers } from "@/hooks/use-entity-users";
import { WILDCARD } from "@/utils/helpers";

interface MemoryUserSelectProps {
  value: string;
  onChange: (value: string) => void;
  /** Caller's own account UUID — pinned to the top of the list as "My user ID". */
  ownUserId: string;
  className?: string;
  disabled?: boolean;
}

/**
 * User-scope picker for the memories page. Supports three tiers:
 * 1. "My user ID" — the operator's own UUID namespace (default).
 * 2. "All user IDs" — sends ``user_id="*"`` (admin=list_all, member=accessible OR).
 * 3. Specific user entities — owned ones marked "(yours)", granted ones also listed.
 */
export default function MemoryUserSelect({
  value,
  onChange,
  ownUserId,
  className,
  disabled,
}: MemoryUserSelectProps) {
  const { users: entityUsers, isLoading, hasMore, loadMore } = useEntityUsers();

  const options: ComboboxOption[] = useMemo(() => {
    const presets: ComboboxOption[] = [
      {
        value: ownUserId,
        label: "My user ID",
        search: "My user ID",
      },
      {
        value: WILDCARD,
        label: "All user IDs",
        search: "All user IDs",
      },
    ];
    const entityOpts: ComboboxOption[] = entityUsers
      .filter((e) => e.id !== ownUserId)
      .map((e) => ({
        value: e.id,
        label: e.is_owner ? `${e.id} (yours)` : e.id,
        search: e.id,
      }));
    return [...presets, ...entityOpts];
  }, [entityUsers, ownUserId]);

  return (
    <Combobox
      value={value}
      onChange={onChange}
      options={options}
      isLoading={isLoading}
      hasMore={hasMore}
      onLoadMore={loadMore}
      placeholder="Select user scope"
      searchPlaceholder="Search user..."
      emptyText="No user found."
      disabled={disabled || !ownUserId}
      className={className}
    />
  );
}
