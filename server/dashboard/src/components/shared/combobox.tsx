"use client";

import { type UIEvent, useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

export interface ComboboxOption {
  value: string;
  label: string;
  search: string;
}

interface ComboboxProps {
  value: string;
  onChange: (value: string) => void;
  options: ComboboxOption[];
  isLoading: boolean;
  hasMore: boolean;
  onLoadMore: () => void;
  placeholder?: string;
  searchPlaceholder?: string;
  emptyText?: string;
  id?: string;
  className?: string;
  disabled?: boolean;
}

/**
 * Generic paginated-search combobox. Renders a Popover with a Command list
 * and loads more items on scroll. Shared by the user picker, entity-scope
 * selector, and entity-user selector.
 */
export default function Combobox({
  value,
  onChange,
  options,
  isLoading,
  hasMore,
  onLoadMore,
  placeholder = "Select...",
  searchPlaceholder = "Search...",
  emptyText = "No results found.",
  id,
  className,
  disabled,
}: ComboboxProps) {
  const [open, setOpen] = useState(false);
  const selected = options.find((o) => o.value === value) ?? null;

  const handleScroll = (e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    if (
      !isLoading &&
      hasMore &&
      el.scrollHeight - el.scrollTop - el.clientHeight < 32
    ) {
      onLoadMore();
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          id={id}
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          className={cn(
            "w-full justify-between font-normal",
            !value && "text-onSurface-default-tertiary",
            className,
          )}
        >
          <span className="truncate">
            {selected ? selected.label : value || placeholder}
          </span>
          <ChevronsUpDown className="ml-2 size-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="min-w-[16rem] w-[var(--radix-popover-trigger-width)] p-0"
        align="start"
      >
        <Command>
          <CommandInput placeholder={searchPlaceholder} />
          <CommandList onScroll={handleScroll}>
            <CommandEmpty>{isLoading ? "Loading..." : emptyText}</CommandEmpty>
            <CommandGroup>
              {options.map((o) => (
                <CommandItem
                  key={o.value}
                  value={o.search}
                  onSelect={() => {
                    onChange(o.value);
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 size-4",
                      value === o.value ? "opacity-100" : "opacity-0",
                    )}
                  />
                  {o.label}
                </CommandItem>
              ))}
            </CommandGroup>
            {hasMore && (
              <div className="py-2 text-center text-xs text-onSurface-default-tertiary">
                {isLoading ? "Loading..." : "Scroll for more"}
              </div>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
