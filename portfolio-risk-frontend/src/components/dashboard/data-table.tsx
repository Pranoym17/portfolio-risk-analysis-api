"use client"

import { cn } from "@/lib/utils"
import { motion } from "framer-motion"
import { ArrowUpDown, ArrowUp, ArrowDown, ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Column<T> {
  key: keyof T | string
  header: string
  align?: "left" | "center" | "right"
  width?: string
  sortable?: boolean
  render?: (row: T, index: number) => React.ReactNode
}

interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  className?: string
  sortColumn?: string
  sortDirection?: "asc" | "desc"
  onSort?: (column: string) => void
  currentPage?: number
  totalPages?: number
  onPageChange?: (page: number) => void
  emptyState?: React.ReactNode
  rowKey?: (row: T, index: number) => string
}

export function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  className,
  sortColumn,
  sortDirection,
  onSort,
  currentPage = 1,
  totalPages = 1,
  onPageChange,
  emptyState,
  rowKey,
}: DataTableProps<T>) {
  const alignClasses = {
    left: "text-left",
    center: "text-center",
    right: "text-right",
  }

  const SortIcon = sortDirection === "asc" ? ArrowUp : sortDirection === "desc" ? ArrowDown : ArrowUpDown

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={cn("bg-card border border-border/60 rounded-lg overflow-hidden", className)}
    >
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border bg-surface-2/50">
              {columns.map((column) => (
                <th
                  key={String(column.key)}
                  className={cn(
                    "px-4 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground",
                    alignClasses[column.align || "left"],
                    column.width
                  )}
                >
                  {column.sortable && onSort ? (
                    <button
                      onClick={() => onSort(String(column.key))}
                      className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
                    >
                      {column.header}
                      <SortIcon className={cn(
                        "h-3.5 w-3.5",
                        sortColumn === column.key ? "text-foreground" : "text-muted-foreground/50"
                      )} />
                    </button>
                  ) : (
                    column.header
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-12 text-center">
                  {emptyState || (
                    <div className="text-muted-foreground">
                      <p className="font-medium">No data available</p>
                      <p className="text-sm mt-1">Data will appear here once available.</p>
                    </div>
                  )}
                </td>
              </tr>
            ) : (
              data.map((row, index) => (
                <motion.tr
                  key={rowKey ? rowKey(row, index) : index}
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2, delay: index * 0.02 }}
                  className="border-b border-border/40 last:border-0 hover:bg-surface-1 transition-colors"
                >
                  {columns.map((column) => (
                    <td
                      key={String(column.key)}
                      className={cn(
                        "px-4 py-3.5 text-sm",
                        alignClasses[column.align || "left"],
                        column.width
                      )}
                    >
                      {column.render 
                        ? column.render(row, index)
                        : String(row[column.key as keyof T] ?? "")}
                    </td>
                  ))}
                </motion.tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      {/* Pagination */}
      {totalPages > 1 && onPageChange && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-border/50">
          <p className="text-xs text-muted-foreground">
            Page {currentPage} of {totalPages}
          </p>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              disabled={currentPage <= 1}
              onClick={() => onPageChange(currentPage - 1)}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              disabled={currentPage >= totalPages}
              onClick={() => onPageChange(currentPage + 1)}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </motion.div>
  )
}

// Status badge for tables
interface StatusBadgeProps {
  status: "positive" | "negative" | "neutral" | "warning"
  children: React.ReactNode
}

export function StatusBadge({ status, children }: StatusBadgeProps) {
  const styles = {
    positive: "bg-positive/10 text-positive border-positive/20",
    negative: "bg-negative/10 text-negative border-negative/20",
    neutral: "bg-muted text-muted-foreground border-border",
    warning: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  }

  return (
    <span className={cn(
      "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border",
      styles[status]
    )}>
      {children}
    </span>
  )
}
