"use client";

import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface SimpleSelectOption {
  value: string;
  label: string;
}

interface SimpleSelectProps {
  value: string;
  onValueChange: (value: string) => void;
  options: SimpleSelectOption[];
  placeholder?: string;
  className?: string;
  itemClassName?: string;
}

/**
 * Minimal wrapper over Select + SelectTrigger + SelectContent + SelectItem.
 * Use when you just need a fixed set of options without extra customisation.
 */
export default function SimpleSelect({
  value,
  onValueChange,
  options,
  placeholder,
  className,
  itemClassName,
}: SimpleSelectProps) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger className={cn("h-9", className)}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((opt) => (
          <SelectItem
            key={opt.value}
            value={opt.value}
            className={itemClassName}
          >
            {opt.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
