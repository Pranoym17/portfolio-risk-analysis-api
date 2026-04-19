"use client"

import { cn } from "@/lib/utils"
import { motion } from "framer-motion"
import { MoreHorizontal, Maximize2, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface ChartPanelProps {
  title: string
  subtitle?: string
  children: React.ReactNode
  className?: string
  actions?: React.ReactNode
  showDefaultActions?: boolean
  height?: "sm" | "md" | "lg" | "xl"
  legend?: React.ReactNode
}

export function ChartPanel({
  title,
  subtitle,
  children,
  className,
  actions,
  showDefaultActions = true,
  height = "md",
  legend,
}: ChartPanelProps) {
  const heights = {
    sm: "h-48",
    md: "h-64",
    lg: "h-80",
    xl: "h-96",
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={cn(
        "relative bg-card border border-border/60 rounded-lg overflow-hidden",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border/50">
        <div>
          <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          {subtitle && (
            <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
          )}
        </div>
        
        <div className="flex items-center gap-1">
          {actions}
          {showDefaultActions && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <MoreHorizontal className="h-4 w-4" />
                  <span className="sr-only">Chart options</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-40">
                <DropdownMenuItem>
                  <Maximize2 className="h-4 w-4 mr-2" />
                  Expand
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <Download className="h-4 w-4 mr-2" />
                  Export PNG
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>
      
      {/* Chart area */}
      <div className={cn("relative px-5 py-4", heights[height])}>
        {/* Subtle grid background */}
        <div 
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(to right, currentColor 1px, transparent 1px),
              linear-gradient(to bottom, currentColor 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px'
          }}
        />
        
        {/* Chart content */}
        <div className="relative h-full w-full">
          {children}
        </div>
      </div>
      
      {/* Legend */}
      {legend && (
        <div className="px-5 pb-4 flex items-center gap-4 flex-wrap">
          {legend}
        </div>
      )}
    </motion.div>
  )
}

// Chart Legend Item component
interface LegendItemProps {
  color: string
  label: string
  value?: string
}

export function LegendItem({ color, label, value }: LegendItemProps) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span 
        className="h-2 w-2 rounded-full shrink-0" 
        style={{ backgroundColor: color }} 
      />
      <span className="text-muted-foreground">{label}</span>
      {value && (
        <span className="font-medium text-foreground tabular-nums">{value}</span>
      )}
    </div>
  )
}
