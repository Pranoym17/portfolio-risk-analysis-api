"use client"

import { cn } from "@/lib/utils"
import { Search, Bell, Menu, ChevronDown, Sun, Moon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useTheme } from "next-themes"
import { usePortfolioData } from "@/components/providers/portfolio-provider"
import { useAuth } from "@/components/providers/auth-provider"
import Link from "next/link"

interface AppTopbarProps {
  className?: string
  title?: string
  subtitle?: string
  onMenuClick?: () => void
  actions?: React.ReactNode
}

export function AppTopbar({
  className,
  title,
  subtitle,
  onMenuClick,
  actions,
}: AppTopbarProps) {
  const { setTheme, theme } = useTheme()
  const { user, logout } = useAuth()
  const { portfolios, activePortfolio, selectPortfolio } = usePortfolioData()
  const initials = (user?.email?.slice(0, 2) ?? "RT").toUpperCase()

  return (
    <header className={cn(
      "sticky top-0 z-40 h-16 bg-background/95 backdrop-blur-sm border-b border-border flex items-center gap-4 px-6",
      className
    )}>
      {/* Mobile menu button */}
      <Button
        variant="ghost"
        size="icon"
        className="lg:hidden"
        onClick={onMenuClick}
      >
        <Menu className="h-5 w-5" />
        <span className="sr-only">Toggle menu</span>
      </Button>

      {/* Page title */}
      {title && (
        <div className="hidden sm:block">
          <h1 className="text-lg font-semibold text-foreground">{title}</h1>
          {subtitle && (
            <p className="text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
      )}

      {/* Portfolio switcher */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" className="hidden md:flex gap-2 h-9 px-3 text-sm">
            <span className="max-w-[180px] truncate">
              {activePortfolio?.name ?? "Select portfolio"}
            </span>
            <ChevronDown className="h-4 w-4 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-56">
          <DropdownMenuLabel>Switch Portfolio</DropdownMenuLabel>
          <DropdownMenuSeparator />
          {portfolios.length === 0 ? (
            <DropdownMenuItem disabled>No portfolios yet</DropdownMenuItem>
          ) : (
            portfolios.map((portfolio) => (
              <DropdownMenuItem
                key={portfolio.id}
                onClick={() => selectPortfolio(portfolio.id)}
              >
                {portfolio.name}
              </DropdownMenuItem>
            ))
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem asChild>
            <Link href="/portfolio">View all portfolios</Link>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Search */}
      <div className="flex-1 max-w-md ml-auto">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search portfolios, tickers..."
            className="pl-9 h-9 bg-surface-1 border-border/60 text-sm"
          />
          <kbd className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 hidden lg:inline-flex h-5 items-center gap-1 rounded border border-border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
            <span className="text-xs">⌘</span>K
          </kbd>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        {actions}
        
        {/* Theme toggle */}
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="h-9 w-9 relative">
          <Bell className="h-4 w-4" />
          <span className="absolute top-2 right-2 h-2 w-2 rounded-full bg-primary" />
          <span className="sr-only">View notifications</span>
        </Button>

        {/* User menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-9 w-9 rounded-full p-0">
              <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-semibold text-primary uppercase">
                {initials}
              </div>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>
              <div className="flex flex-col space-y-1">
                <p className="text-sm font-medium">Authenticated Workspace</p>
                <p className="text-xs text-muted-foreground">{user?.email ?? "No active user"}</p>
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link href="/profile">Profile</Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <Link href="/settings">Settings</Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={logout}>Sign out</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
