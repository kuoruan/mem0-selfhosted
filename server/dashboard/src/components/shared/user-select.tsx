"use client";

import { useMemo, useState, type UIEvent } from "react";
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
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useUsers } from "@/hooks/use-users";

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
  const [open, setOpen] = useState(false);

  const selected = useMemo(
    () => users.find((u) => u.id === value) ?? null,
    [users, value],
  );

  const handleScroll = (e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    if (
      !isLoading &&
      hasMore &&
      el.scrollHeight - el.scrollTop - el.clientHeight < 32
    ) {
      loadMore();
    }
  };

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
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          id={id}
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn(
            "w-full justify-between font-normal",
            !value && "text-onSurface-default-tertiary",
            className,
          )}
        >
          <span className="truncate">
            {selected
              ? `${selected.name} (${selected.email})`
              : value || placeholder}
          </span>
          <ChevronsUpDown className="ml-2 size-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="min-w-[16rem] w-[var(--radix-popover-trigger-width)] p-0"
        align="start"
      >
        <Command>
          <CommandInput placeholder="Search user..." />
          <CommandList onScroll={handleScroll}>
            <CommandEmpty>
              {isLoading ? "Loading..." : "No user found."}
            </CommandEmpty>
            <CommandGroup>
              {users.map((u) => (
                <CommandItem
                  key={u.id}
                  value={`${u.name} ${u.email} ${u.id}`}
                  onSelect={() => {
                    onChange(u.id);
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 size-4",
                      value === u.id ? "opacity-100" : "opacity-0",
                    )}
                  />
                  {u.name} ({u.email})
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
