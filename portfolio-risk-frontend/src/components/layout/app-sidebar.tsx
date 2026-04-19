"use client"

import { cn } from "@/lib/utils"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Logo } from "@/components/brand/logo"
import { motion } from "framer-motion"
import {
  LayoutDashboard,
  Briefcase,
  BarChart3,
  Shield,
  Search,
  Settings,
  HelpCircle,
  ChevronRight,
  Plus,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { usePortfolioData } from "@/components/providers/portfolio-provider"
import { useAuth } from "@/components/providers/auth-provider"

const mainNavItems = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/portfolio", label: "Portfolio", icon: Briefcase },
  { href: "/risk", label: "Risk Analytics", icon: BarChart3 },
  { href: "/holdings", label: "Holdings", icon: Search },
]

const secondaryNavItems = [
  { href: "/settings", label: "Settings", icon: Settings },
  { href: "/contact", label: "Help & Support", icon: HelpCircle },
]

interface AppSidebarProps {
  className?: string
}

export function AppSidebar({ className }: AppSidebarProps) {
  const pathname = usePathname()
  const { portfolios, activePortfolioId, selectPortfolio } = usePortfolioData()
  const { user } = useAuth()
  const recentPortfolios = portfolios.slice(0, 5)
  const initials = (user?.email?.slice(0, 2) ?? "RT").toUpperCase()

  return (
    <aside className={cn(
      "flex flex-col h-full w-64 bg-sidebar border-r border-sidebar-border",
      className
    )}>
      {/* Logo */}
      <div className="h-16 flex items-center px-5 border-b border-sidebar-border">
        <Logo href="/dashboard" />
      </div>

      {/* Quick action */}
      <div className="p-4">
        <Button className="w-full justify-start gap-2" size="sm" asChild>
          <Link href="/portfolio">
            <Plus className="h-4 w-4" />
            Open Portfolio
          </Link>
        </Button>
      </div>

      <ScrollArea className="flex-1 px-3">
        {/* Main navigation */}
        <nav className="space-y-1">
          {mainNavItems.map((item) => {
            const isActive = pathname.startsWith(item.href)
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "group flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all relative",
                  isActive 
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-sidebar-primary rounded-full"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                  />
                )}
                <item.icon className={cn(
                  "h-4 w-4 shrink-0 transition-colors",
                  isActive ? "text-sidebar-primary" : "text-sidebar-foreground/50 group-hover:text-sidebar-foreground/70"
                )} />
                {item.label}
                {isActive && (
                  <ChevronRight className="h-4 w-4 ml-auto text-sidebar-foreground/40" />
                )}
              </Link>
            )
          })}
        </nav>

        <Separator className="my-4 bg-sidebar-border" />

        {/* Quick portfolios */}
        <div className="mb-4">
          <h3 className="px-3 mb-2 text-xs font-semibold uppercase tracking-wider text-sidebar-foreground/50">
            Recent Portfolios
          </h3>
          <div className="space-y-0.5">
            {recentPortfolios.length === 0 ? (
              <div className="px-3 py-2 text-sm text-sidebar-foreground/50">
                Create a portfolio to start analyzing risk.
              </div>
            ) : (
              recentPortfolios.map((portfolio) => (
                <Link
                  key={portfolio.id}
                  href="/portfolio"
                  onClick={() => selectPortfolio(portfolio.id)}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                    activePortfolioId === portfolio.id
                      ? "bg-sidebar-accent text-sidebar-foreground"
                      : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
                  )}
                >
                  <Shield className="h-3.5 w-3.5 shrink-0 text-sidebar-foreground/40" />
                  <span className="truncate">{portfolio.name}</span>
                </Link>
              ))
            )}
          </div>
        </div>

        <Separator className="my-4 bg-sidebar-border" />

        {/* Secondary navigation */}
        <nav className="space-y-1">
          {secondaryNavItems.map((item) => {
            const isActive = pathname.startsWith(item.href)
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  isActive 
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
                )}
              >
                <item.icon className="h-4 w-4 shrink-0" />
                {item.label}
              </Link>
            )
          })}
        </nav>
      </ScrollArea>

      {/* User section */}
      <div className="p-4 border-t border-sidebar-border">
        <Link
          href="/profile"
          className="flex items-center gap-3 p-2 rounded-md hover:bg-sidebar-accent/50 transition-colors"
        >
          <div className="h-8 w-8 rounded-full bg-sidebar-accent flex items-center justify-center text-sm font-semibold text-sidebar-accent-foreground">
            {initials}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-sidebar-foreground truncate">{user?.email ?? "Workspace"}</p>
            <p className="text-xs text-sidebar-foreground/60 truncate">
              {portfolios.length} portfolio{portfolios.length === 1 ? "" : "s"}
            </p>
          </div>
        </Link>
      </div>
    </aside>
  )
}
