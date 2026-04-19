"use client"

import { cn } from "@/lib/utils"
import Link from "next/link"

interface LogoProps {
  className?: string
  size?: "sm" | "md" | "lg"
  showText?: boolean
  href?: string
}

export function Logo({ className, size = "md", showText = true, href = "/" }: LogoProps) {
  const sizes = {
    sm: "h-6 w-6",
    md: "h-8 w-8", 
    lg: "h-10 w-10",
  }

  const textSizes = {
    sm: "text-base",
    md: "text-lg",
    lg: "text-xl",
  }

  const content = (
    <div className={cn("flex items-center gap-2.5", className)}>
      <div className={cn("relative", sizes[size])}>
        {/* Abstract chart/mountain mark */}
        <svg viewBox="0 0 32 32" fill="none" className="h-full w-full">
          <rect 
            x="2" y="2" 
            width="28" height="28" 
            rx="6" 
            className="fill-primary"
          />
          <path
            d="M8 22L12 14L17 18L24 8"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-primary-foreground"
          />
          <circle cx="24" cy="8" r="2" className="fill-primary-foreground" />
        </svg>
      </div>
      {showText && (
        <span className={cn("font-semibold tracking-tight text-foreground", textSizes[size])}>
          Risk Terminal
        </span>
      )}
    </div>
  )

  if (href) {
    return (
      <Link href={href} className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm">
        {content}
      </Link>
    )
  }

  return content
}
