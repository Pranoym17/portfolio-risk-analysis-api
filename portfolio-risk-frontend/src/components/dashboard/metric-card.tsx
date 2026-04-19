"use client"

import { cn } from "@/lib/utils"
import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react"
import { motion } from "framer-motion"

interface MetricCardProps {
  label: string
  value: string | number
  change?: number
  changeLabel?: string
  trend?: "up" | "down" | "neutral"
  icon?: React.ReactNode
  className?: string
  variant?: "default" | "highlight" | "compact"
  sparkline?: number[]
}

export function MetricCard({
  label,
  value,
  change,
  changeLabel,
  trend,
  icon,
  className,
  variant = "default",
  sparkline,
}: MetricCardProps) {
  const trendColor = {
    up: "text-positive",
    down: "text-negative", 
    neutral: "text-muted-foreground",
  }

  const TrendIcon = trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : Minus

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "group relative bg-card border border-border/60 rounded-lg overflow-hidden transition-all duration-200",
        "hover:border-border hover:shadow-sm",
        variant === "highlight" && "border-l-2 border-l-primary",
        variant === "compact" ? "p-4" : "p-5",
        className
      )}
    >
      {/* Subtle top gradient */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            {icon && (
              <span className="text-muted-foreground">{icon}</span>
            )}
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {label}
            </span>
          </div>
          
          <div className="flex items-baseline gap-2">
            <span className={cn(
              "font-semibold tabular-nums text-foreground",
              variant === "compact" ? "text-xl" : "text-2xl"
            )}>
              {value}
            </span>
            
            {change !== undefined && trend && (
              <span className={cn("flex items-center gap-0.5 text-sm font-medium", trendColor[trend])}>
                <TrendIcon className="h-3.5 w-3.5" />
                {Math.abs(change).toFixed(2)}%
              </span>
            )}
          </div>
          
          {changeLabel && (
            <span className="text-xs text-muted-foreground mt-1 block">
              {changeLabel}
            </span>
          )}
        </div>
        
        {/* Mini sparkline visualization */}
        {sparkline && sparkline.length > 0 && (
          <div className="h-10 w-16 flex items-end gap-px">
            {sparkline.map((val, i) => {
              const max = Math.max(...sparkline)
              const min = Math.min(...sparkline)
              const range = max - min || 1
              const height = ((val - min) / range) * 100
              const isLast = i === sparkline.length - 1
              
              return (
                <div
                  key={i}
                  className={cn(
                    "flex-1 rounded-sm transition-all",
                    isLast ? "bg-primary" : "bg-muted-foreground/20"
                  )}
                  style={{ height: `${Math.max(height, 8)}%` }}
                />
              )
            })}
          </div>
        )}
      </div>
    </motion.div>
  )
}
