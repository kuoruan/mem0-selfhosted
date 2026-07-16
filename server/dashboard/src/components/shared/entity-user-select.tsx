"use client";

import { useMemo } from "react";
import Combobox, { type ComboboxOption } from "@/components/shared/combobox";
import { useEntityUsers } from "@/hooks/use-entity-users";

interface EntityUserSelectProps {
  value: string;
  onChange: (value: string) => void;
  /** Caller's own account UUID — pinned to the top of the list as "My user ID". */
  ownUserId: string;
  className?: string;
  disabled?: boolean;
}

/**
 * Picker for a ``user_id`` namespace on the memories page (not a dashboard
 * account — distinct from ``UserSelect``). Delegates rendering/scroll to
 * ``Combobox``; the caller's own UUID is pinned first as "My user ID" (it's
 * an account identity, not an entities-table row), and owned namespaces are
 * marked "(yours)".
 */
export default function EntityUserSelect({
  value,
  onChange,
  ownUserId,
  className,
  disabled,
}: EntityUserSelectProps) {
  const { users: entityUsers, isLoading, hasMore, loadMore } = useEntityUsers();

  const options: ComboboxOption[] = useMemo(() => {
    const own: ComboboxOption[] = ownUserId
      ? [
          {
            value: ownUserId,
            label: "My user ID",
            search: "My user ID",
          },
        ]
      : [];
    const entityOpts: ComboboxOption[] = entityUsers
      .filter((e) => e.id !== ownUserId)
      .map((e) => ({
        value: e.id,
        label: e.is_owner ? `${e.id} (yours)` : e.id,
        search: e.id,
      }));
    return [...own, ...entityOpts];
  }, [ownUserId, entityUsers]);

  return (
    <Combobox
      value={value}
      onChange={onChange}
      options={options}
      isLoading={isLoading}
      hasMore={hasMore}
      onLoadMore={loadMore}
      placeholder="Select entity user"
      searchPlaceholder="Search entity user..."
      emptyText="No entity user found."
      disabled={disabled || !ownUserId}
      className={className}
    />
  );
}
