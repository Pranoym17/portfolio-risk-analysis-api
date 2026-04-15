"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type Column<T> = {
  key: string;
  header: string;
  align?: "left" | "right";
  className?: string;
  render: (row: T) => React.ReactNode;
};

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  className,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  className?: string;
}) {
  const template = columns.map((column) => (column.align === "right" ? "minmax(90px,0.75fr)" : "minmax(150px,1.2fr)")).join(" ");

  return (
    <div className={cn("overflow-hidden rounded-[22px] border border-[var(--line)] bg-[rgba(255,255,255,0.02)]", className)}>
      <div
        className="grid gap-4 border-b border-[var(--line)] bg-[rgba(255,255,255,0.03)] px-5 py-4 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-faint)]"
        style={{ gridTemplateColumns: template }}
      >
        {columns.map((column) => (
          <div key={column.key} className={cn(column.align === "right" ? "text-right" : "text-left", column.className)}>
            {column.header}
          </div>
        ))}
      </div>
      <div className="table-scroll max-h-[520px] overflow-auto">
        {rows.map((row, index) => (
          <motion.div
            key={rowKey(row)}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.22, delay: index * 0.02 }}
            className="grid gap-4 border-b border-[rgba(124,145,172,0.12)] px-5 py-4 text-sm transition hover:bg-[rgba(255,255,255,0.03)] last:border-b-0"
            style={{ gridTemplateColumns: template }}
          >
            {columns.map((column) => (
              <div key={column.key} className={cn(column.align === "right" ? "text-right" : "text-left", column.className)}>
                {column.render(row)}
              </div>
            ))}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
