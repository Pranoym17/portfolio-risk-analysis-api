"use client"

import { useMemo, useState } from "react"
import Link from "next/link"
import {
  MoreHorizontal,
  Plus,
  Search,
  Shield,
  Trash2,
  Wallet,
} from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Badge } from "@/components/ui/badge"
import { usePortfolioData } from "@/components/providers/portfolio-provider"
import { cn, fmtPct, getErrorMessage } from "@/lib/utils"

export default function PortfolioPage() {
  const {
    portfolios,
    activePortfolioId,
    selectPortfolio,
    createPortfolio,
    deletePortfolio,
    isLoading,
  } = usePortfolioData()
  const [searchQuery, setSearchQuery] = useState("")
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [newPortfolioName, setNewPortfolioName] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)

  const filteredPortfolios = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    if (!query) return portfolios
    return portfolios.filter((portfolio) => portfolio.name.toLowerCase().includes(query))
  }, [portfolios, searchQuery])

  const totalHoldings = portfolios.reduce((sum, portfolio) => sum + portfolio.holdings.length, 0)

  const handleCreate = async () => {
    if (!newPortfolioName.trim()) return
    setIsSubmitting(true)
    try {
      const created = await createPortfolio(newPortfolioName.trim())
      toast.success(`Created ${created.name}`)
      setNewPortfolioName("")
      setIsCreateOpen(false)
    } catch (error) {
      toast.error(getErrorMessage(error, "Unable to create portfolio"))
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDelete = async (portfolioId: number, name: string) => {
    try {
      await deletePortfolio(portfolioId)
      toast.success(`Deleted ${name}`)
    } catch (error) {
      toast.error(getErrorMessage(error, "Unable to delete portfolio"))
    }
  }

  return (
    <div className="min-h-screen">
      <div className="border-b border-border/60 bg-card/50">
        <div className="px-6 lg:px-8 py-6">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <Wallet className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight">Portfolios</h1>
                  <p className="text-sm text-muted-foreground">
                    Create portfolios, choose an active workspace, and prepare holdings for risk analysis.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="text-right">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Portfolios
                </p>
                <p className="text-2xl font-semibold tabular-nums">{portfolios.length}</p>
              </div>
              <div className="h-10 w-px bg-border" />
              <div className="text-right">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Holdings Stored
                </p>
                <p className="text-2xl font-semibold tabular-nums">{totalHoldings}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="border-b border-border/40 bg-surface-1/50">
        <div className="px-6 lg:px-8 py-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search portfolios..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-72 pl-9 h-9 bg-background"
              />
            </div>

            <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
              <DialogTrigger asChild>
                <Button size="sm" className="h-9">
                  <Plus className="h-4 w-4 mr-2" />
                  New Portfolio
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create portfolio</DialogTitle>
                  <DialogDescription>
                    Start with a name now, then add holdings and run analysis from the workspace.
                  </DialogDescription>
                </DialogHeader>
                <Input
                  placeholder="Example: Core Equity Allocation"
                  value={newPortfolioName}
                  onChange={(e) => setNewPortfolioName(e.target.value)}
                  className="bg-surface-1"
                />
                <DialogFooter>
                  <Button variant="outline" onClick={() => setIsCreateOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleCreate} disabled={isSubmitting || !newPortfolioName.trim()}>
                    {isSubmitting ? "Creating..." : "Create"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>

      <div className="px-6 lg:px-8 py-8">
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading portfolios...</p>
        ) : filteredPortfolios.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border/60 bg-card p-10 text-center">
            <h2 className="text-lg font-semibold text-foreground">No matching portfolios</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Create a new workspace or clear your search to see all portfolios.
            </p>
          </div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
            {filteredPortfolios.map((portfolio) => {
              const totalWeight = portfolio.holdings.reduce((sum, holding) => sum + holding.weight, 0)
              const topHoldings = [...portfolio.holdings].sort((a, b) => b.weight - a.weight).slice(0, 4)
              const isActive = activePortfolioId === portfolio.id

              return (
                <div
                  key={portfolio.id}
                  className={cn(
                    "group relative overflow-hidden rounded-xl border bg-card transition-all",
                    isActive ? "border-primary/50 shadow-lg shadow-primary/5" : "border-border/60 hover:border-border"
                  )}
                >
                  <div className={cn(
                    "absolute inset-x-0 top-0 h-1",
                    isActive ? "bg-gradient-to-r from-primary to-chart-2" : "bg-gradient-to-r from-border to-transparent"
                  )} />

                  <div className="p-6">
                    <div className="mb-4 flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-lg font-semibold">{portfolio.name}</h3>
                          {isActive && <Badge>Active</Badge>}
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          Portfolio ID #{portfolio.id}
                        </p>
                      </div>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => selectPortfolio(portfolio.id)}>
                            Set as active
                          </DropdownMenuItem>
                          <DropdownMenuItem asChild>
                            <Link href="/holdings">Edit holdings</Link>
                          </DropdownMenuItem>
                          <DropdownMenuItem asChild>
                            <Link href="/risk">Run analysis</Link>
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={() => void handleDelete(portfolio.id, portfolio.name)}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="mb-5 grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs uppercase tracking-wider text-muted-foreground">Holdings</p>
                        <p className="mt-1 text-2xl font-semibold tabular-nums">{portfolio.holdings.length}</p>
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-wider text-muted-foreground">Weight Check</p>
                        <p className={cn(
                          "mt-1 text-2xl font-semibold tabular-nums",
                          Math.abs(totalWeight - 1) < 1e-6 ? "text-positive" : "text-amber-400"
                        )}>
                          {fmtPct(totalWeight)}
                        </p>
                      </div>
                    </div>

                    <div className="mb-5">
                      <div className="mb-2 flex items-center justify-between">
                        <p className="text-xs uppercase tracking-wider text-muted-foreground">Top Weights</p>
                        <Badge variant="secondary">{topHoldings.length} shown</Badge>
                      </div>
                      {topHoldings.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No holdings yet.</p>
                      ) : (
                        <div className="space-y-3">
                          {topHoldings.map((holding) => (
                            <div key={holding.id}>
                              <div className="mb-1 flex items-center justify-between text-xs">
                                <span className="font-medium text-foreground">{holding.ticker}</span>
                                <span className="tabular-nums text-muted-foreground">{fmtPct(holding.weight)}</span>
                              </div>
                              <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                                <div className="h-full rounded-full bg-primary" style={{ width: `${holding.weight * 100}%` }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-3 pt-4 border-t border-border/50">
                      <Button
                        variant={isActive ? "secondary" : "outline"}
                        onClick={() => selectPortfolio(portfolio.id)}
                      >
                        {isActive ? "Active Workspace" : "Set Active"}
                      </Button>
                      <Button asChild>
                        <Link href={portfolio.holdings.length ? "/risk" : "/holdings"}>
                          {portfolio.holdings.length ? "Analyze" : "Add Holdings"}
                        </Link>
                      </Button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {portfolios.length === 0 && !isLoading && (
          <div className="mt-6 rounded-xl border border-border/60 bg-surface-1/50 p-6">
            <div className="flex items-start gap-3">
              <Shield className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <h2 className="text-base font-semibold text-foreground">Backend-ready workflow</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Once a portfolio exists, you can replace holdings, validate tickers, and request risk metrics directly from the API-backed screens.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
