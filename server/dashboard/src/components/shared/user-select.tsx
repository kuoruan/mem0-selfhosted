"use client";

import { useMemo } from "react";
import { Input } from "@/components/ui/input";
import { useUsers } from "@/hooks/use-users";
import Combobox from "@/components/shared/combobox";
import type { ComboboxOption } from "@/components/shared/combobox";

interface UserSelectProps {
  value: string;
  onChange: (value: string) => void;
  isAdmin: boolean;
  id?: string;
  className?: string;
  placeholder?: string;
  inputPlaceholder?: string;
}

/**
 * User picker. Admins get a searchable Combobox backed by the paginated
 * ``/users`` list (loads more on scroll); everyone else gets a free-text UUID
 * input. Used by the entity create / transfer / permissions dialogs.
 */
export default function UserSelect({
  value,
  onChange,
  isAdmin,
  id,
  className,
  placeholder = "Select a user",
  inputPlaceholder = "User UUID",
}: UserSelectProps) {
  const { users, isLoading, hasMore, loadMore } = useUsers(isAdmin);

  const options: ComboboxOption[] = useMemo(
    () =>
      users.map((u) => ({
        value: u.id,
        label: `${u.name} (${u.email})`,
        search: `${u.name} ${u.email} ${u.id}`,
      })),
    [users],
  );

  if (!isAdmin) {
    return (
      <Input
        id={id}
        className={className}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={inputPlaceholder}
      />
    );
  }

  return (
    <Combobox
      value={value}
      onChange={onChange}
      options={options}
      isLoading={isLoading}
      hasMore={hasMore}
      onLoadMore={loadMore}
      placeholder={placeholder}
      searchPlaceholder="Search user..."
      emptyText="No user found."
      id={id}
      className={className}
    />
  );
}
