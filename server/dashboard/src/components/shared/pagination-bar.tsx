"use client";

import { Button } from "@/components/ui/button";

interface PaginationBarProps {
  page: number;
  total: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

export default function PaginationBar({
  page,
  total,
  pageSize,
  onPageChange,
}: PaginationBarProps) {
  const totalPages = Math.ceil(total / pageSize);
  if (totalPages <= 1) return null;

  return (
    <div className="flex items-center justify-between text-sm text-onSurface-default-tertiary">
      <span>
        {page * pageSize + 1}–{Math.min((page + 1) * pageSize, total)} of{" "}
        {total}
      </span>
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={page === 0}
          onClick={() => onPageChange(page - 1)}
        >
          Previous
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={page >= totalPages - 1}
          onClick={() => onPageChange(page + 1)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
