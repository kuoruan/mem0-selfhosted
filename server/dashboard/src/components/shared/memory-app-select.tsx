"use client";

import { useMemo } from "react";
import Combobox, { type ComboboxOption } from "@/components/shared/combobox";
import { useEntityApps } from "@/hooks/use-entity-apps";

interface MemoryAppSelectProps {
  value: string;
  onChange: (value: string) => void;
  className?: string;
  disabled?: boolean;
}

/**
 * Picker for an ``app_id`` namespace on the memories page. Delegates
 * rendering/scroll to ``Combobox``; the first option is "All apps" (empty
 * string — no app filter), and owned apps are marked "(yours)".
 */
export default function MemoryAppSelect({
  value,
  onChange,
  className,
  disabled,
}: MemoryAppSelectProps) {
  const { apps: entityApps, isLoading, hasMore, loadMore } = useEntityApps();

  const options: ComboboxOption[] = useMemo(() => {
    const all: ComboboxOption[] = [
      { value: "", label: "All apps", search: "All apps" },
    ];
    const entityOpts: ComboboxOption[] = entityApps.map((e) => ({
      value: e.id,
      label: e.is_owner ? `${e.id} (yours)` : e.id,
      search: e.id,
    }));
    return [...all, ...entityOpts];
  }, [entityApps]);

  return (
    <Combobox
      value={value}
      onChange={onChange}
      options={options}
      isLoading={isLoading}
      hasMore={hasMore}
      onLoadMore={loadMore}
      placeholder="Select app"
      searchPlaceholder="Search app..."
      emptyText="No app found."
      disabled={disabled}
      className={className}
    />
  );
}
