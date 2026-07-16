"use client";

import { useMemo } from "react";
import { useUsers } from "@/hooks/use-users";
import Combobox from "@/components/shared/combobox";
import type { ComboboxOption } from "@/components/shared/combobox";

interface EntityScopeSelectProps {
  value: string;
  onChange: (value: string) => void;
  ownUserId: string;
  className?: string;
}

/**
 * Entity-scope selector for the Entities page (admin only).
 *
 * Presents "My entities" (the operator's own namespace), "All entities"
 * (admin bypass — includes unowned), and the paginated dashboard user list
 * from ``/users``. The operator's own user is excluded from the user list
 * to avoid duplicating "My entities".
 */
export default function EntityScopeSelect({
  value,
  onChange,
  ownUserId,
  className,
}: EntityScopeSelectProps) {
  const { users, isLoading, hasMore, loadMore } = useUsers(true);

  const options: ComboboxOption[] = useMemo(() => {
    const presets: ComboboxOption[] = [
      { value: ownUserId, label: "My entities", search: "My entities" },
      { value: "all", label: "All entities", search: "All entities" },
    ];
    const userOptions: ComboboxOption[] = users
      .filter((u) => u.id !== ownUserId)
      .map((u) => ({
        value: u.id,
        label: `${u.name} (${u.email})`,
        search: `${u.name} ${u.email} ${u.id}`,
      }));
    return [...presets, ...userOptions];
  }, [users, ownUserId]);

  return (
    <Combobox
      value={value}
      onChange={onChange}
      options={options}
      isLoading={isLoading}
      hasMore={hasMore}
      onLoadMore={loadMore}
      placeholder="Scope"
      searchPlaceholder="Search user..."
      emptyText="No user found."
      className={className}
    />
  );
}
