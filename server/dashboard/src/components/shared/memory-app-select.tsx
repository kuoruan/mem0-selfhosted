"use client";

import { useMemo } from "react";
import Combobox, { type ComboboxOption } from "@/components/shared/combobox";
import { useAppEntities } from "@/hooks/use-app-entities";

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
 *
 * "All apps" is the empty string rather than the ``*`` wildcard on purpose:
 * any ``app_id`` filter (``*`` included) only matches memories that carry an
 * ``app_id``, excluding app-less memories. An empty value omits ``app_id`` so
 * the app dimension is left unconstrained.
 */
export default function MemoryAppSelect({
  value,
  onChange,
  className,
  disabled,
}: MemoryAppSelectProps) {
  const { appEntities, isLoading, hasMore, loadMore } = useAppEntities();

  const options: ComboboxOption[] = useMemo(() => {
    const all: ComboboxOption[] = [
      { value: "", label: "All apps", search: "All apps" },
    ];
    const entityOpts: ComboboxOption[] = appEntities.map((e) => {
      const displayName = e.name || e.id;
      return {
        value: e.id,
        label: e.is_owner ? `${displayName} (yours)` : displayName,
        search: e.name ? `${e.name} ${e.id}` : e.id,
      };
    });
    return [...all, ...entityOpts];
  }, [appEntities]);

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
