"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import {
  AlertTriangle,
  CheckCircle2,
  Plus,
  Save,
  Search,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { usePortfolioData } from "@/components/providers/portfolio-provider"
import { validateTicker } from "@/lib/api"
import { fmtPct, getErrorMessage } from "@/lib/utils"

type EditorRow = {
  id: string
  ticker: string
  weightPct: string
  status?: "idle" | "valid" | "invalid"
  message?: string
}

function createRow(partial?: Partial<EditorRow>): EditorRow {
  return {
    id: crypto.randomUUID(),
    ticker: "",
    weightPct: "",
    status: "idle",
    ...partial,
  }
}

export default function HoldingsPage() {
  const {
    portfolios,
    activePortfolio,
    activePortfolioId,
    selectPortfolio,
    replaceHoldings,
  } = usePortfolioData()
  const [rows, setRows] = useState<EditorRow[]>([])
  const [isSaving, setIsSaving] = useState(false)
  const [isValidating, setIsValidating] = useState(false)

  useEffect(() => {
    if (!activePortfolio) {
      setRows([createRow()])
      return
    }

    if (activePortfolio.holdings.length === 0) {
      setRows([createRow()])
      return
    }

    setRows(
      activePortfolio.holdings.map((holding) =>
        createRow({
          ticker: holding.ticker,
          weightPct: String(Number((holding.weight * 100).toFixed(4))),
        }),
      ),
    )
  }, [activePortfolio])

  const completedRows = useMemo(
    () =>
      rows.filter((row) => row.ticker.trim() && row.weightPct.trim()),
    [rows],
  )

  const incompleteRows = useMemo(
    () =>
      rows.filter((row) => Boolean(row.ticker.trim()) !== Boolean(row.weightPct.trim())),
    [rows],
  )

  const totalWeightPct = useMemo(
    () =>
      completedRows.reduce((sum, row) => {
        const value = Number(row.weightPct)
        return sum + (Number.isFinite(value) ? value : 0)
      }, 0),
    [completedRows],
  )

  const cleanedRows = useMemo(
    () =>
      rows.filter((row) => row.ticker.trim() || row.weightPct.trim()),
    [rows],
  )

  const updateRow = (id: string, patch: Partial<EditorRow>) => {
    setRows((current) =>
      current.map((row) => row.id === id ? { ...row, ...patch } : row),
    )
  }

  const removeRow = (id: string) => {
    setRows((current) => {
      const next = current.filter((row) => row.id !== id)
      return next.length > 0 ? next : [createRow()]
    })
  }

  const addRow = () => {
    setRows((current) => [...current, createRow()])
  }

  const runValidation = async () => {
    const rowsToValidate = cleanedRows.filter((row) => row.ticker.trim())
    if (rowsToValidate.length === 0) {
      toast.error("Add at least one ticker before validating")
      return
    }

    setIsValidating(true)
    try {
      const results = await Promise.all(
        rowsToValidate.map((row) => validateTicker(row.ticker.trim().toUpperCase())),
      )

      setRows((current) =>
        current.map((row) => {
          const result = results.find(
            (item) => item.ticker === row.ticker.trim().toUpperCase(),
          )
          if (!result) return row
          return {
            ...row,
            status: result.is_valid ? "valid" : "invalid",
            message: result.is_valid
              ? `${result.rows_returned} pricing rows returned`
              : result.error || "Ticker validation failed",
          }
        }),
      )

      const invalidCount = results.filter((result) => !result.is_valid).length
      if (invalidCount === 0) {
        toast.success("All tickers validated successfully")
      } else {
        toast.warning(`${invalidCount} ticker${invalidCount === 1 ? "" : "s"} failed validation`)
      }
    } catch (error) {
      toast.error(getErrorMessage(error, "Unable to validate tickers"))
    } finally {
      setIsValidating(false)
    }
  }

  const handleSave = async () => {
    if (!activePortfolioId) {
      toast.error("Select a portfolio before saving holdings")
      return
    }

    const holdings = cleanedRows.map((row) => ({
      ticker: row.ticker.trim().toUpperCase(),
      weight: Number(row.weightPct) / 100,
    }))

    if (holdings.length === 0) {
      toast.error("Add at least one holding before saving")
      return
    }

    if (holdings.some((holding) => !holding.ticker || !Number.isFinite(holding.weight) || holding.weight <= 0)) {
      toast.error("Every row needs a ticker and a positive weight")
      return
    }

    const total = holdings.reduce((sum, holding) => sum + holding.weight, 0)
    if (Math.abs(total - 1) > 0.00001) {
      toast.error("Weights must total exactly 100% before saving")
      return
    }

    setIsSaving(true)
    try {
      await replaceHoldings(activePortfolioId, holdings)
      toast.success("Holdings updated")
    } catch (error) {
      toast.error(getErrorMessage(error, "Unable to save holdings"))
    } finally {
      setIsSaving(false)
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
                  <Search className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight">Holdings Editor</h1>
                  <p className="text-sm text-muted-foreground">
                    Maintain ticker weights exactly as your backend expects them.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="min-w-[240px]">
                <Select
                  value={activePortfolioId ? String(activePortfolioId) : undefined}
                  onValueChange={(value) => selectPortfolio(Number(value))}
                >
                  <SelectTrigger className="h-10">
                    <SelectValue placeholder="Select portfolio" />
                  </SelectTrigger>
                  <SelectContent>
                    {portfolios.map((portfolio) => (
                      <SelectItem key={portfolio.id} value={String(portfolio.id)}>
                        {portfolio.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Badge variant={Math.abs(totalWeightPct - 100) < 0.001 ? "default" : "secondary"}>
                Total {totalWeightPct.toFixed(2)}%
              </Badge>
              {incompleteRows.length > 0 && (
                <Badge variant="secondary">
                  {incompleteRows.length} incomplete row{incompleteRows.length === 1 ? "" : "s"}
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>

      {!activePortfolio ? (
        <div className="px-6 lg:px-8 py-10">
          <div className="rounded-xl border border-dashed border-border/60 bg-card p-10 text-center">
            <h2 className="text-lg font-semibold text-foreground">No active portfolio selected</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Create a portfolio first, then come back here to load weights into the backend.
            </p>
            <Button className="mt-5" asChild>
              <Link href="/portfolio">Open portfolio workspace</Link>
            </Button>
          </div>
        </div>
      ) : (
        <div className="px-6 lg:px-8 py-8">
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
            <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
              <div className="flex items-center justify-between border-b border-border/50 px-6 py-4">
                <div>
                  <h2 className="font-semibold">{activePortfolio.name}</h2>
                  <p className="text-sm text-muted-foreground">
                    Weights should sum to exactly 100% before saving.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" onClick={runValidation} disabled={isValidating}>
                    {isValidating ? "Validating..." : "Validate tickers"}
                  </Button>
                  <Button onClick={handleSave} disabled={isSaving}>
                    <Save className="h-4 w-4 mr-2" />
                    {isSaving ? "Saving..." : "Save holdings"}
                  </Button>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border bg-surface-2/50">
                      <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Ticker</th>
                      <th className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Weight %</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Validation</th>
                      <th className="px-6 py-3 w-16" />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.id} className="border-b border-border/40 last:border-0">
                        <td className="px-6 py-4">
                          <Input
                            value={row.ticker}
                            onChange={(e) =>
                              updateRow(row.id, {
                                ticker: e.target.value.toUpperCase(),
                                status: "idle",
                                message: undefined,
                              })
                            }
                            placeholder="AAPL"
                            className="bg-surface-1"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center justify-end gap-2">
                            <Input
                              value={row.weightPct}
                              onChange={(e) =>
                                updateRow(row.id, {
                                  weightPct: e.target.value,
                                })
                              }
                              placeholder="25"
                              className="w-28 bg-surface-1 text-right tabular-nums"
                            />
                            <span className="text-sm text-muted-foreground">%</span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          {row.status === "valid" ? (
                            <div className="flex items-center gap-2 text-sm text-positive">
                              <CheckCircle2 className="h-4 w-4" />
                              <span>{row.message}</span>
                            </div>
                          ) : row.status === "invalid" ? (
                            <div className="flex items-center gap-2 text-sm text-amber-400">
                              <AlertTriangle className="h-4 w-4" />
                              <span>{row.message}</span>
                            </div>
                          ) : (
                            <span className="text-sm text-muted-foreground">Not checked yet</span>
                          )}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <Button variant="ghost" size="icon" onClick={() => removeRow(row.id)}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex items-center justify-between border-t border-border/50 px-6 py-4 bg-surface-1/40">
                <Button variant="outline" onClick={addRow}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add row
                </Button>
                <p className="text-sm text-muted-foreground">
                  Backend payload total: <span className="font-medium tabular-nums text-foreground">{fmtPct(totalWeightPct / 100)}</span>
                </p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-xl border border-border/60 bg-card p-5">
                <h3 className="font-semibold text-foreground">Editor rules</h3>
                <div className="mt-4 space-y-3 text-sm text-muted-foreground">
                  <p>Tickers are normalized to uppercase before sending to the API.</p>
                  <p>The backend rejects holdings unless weights sum to exactly 1.0.</p>
                  <p>Ticker validation checks whether pricing history is available for analysis.</p>
                </div>
              </div>

              <div className="rounded-xl border border-border/60 bg-card p-5">
                <h3 className="font-semibold text-foreground">Current state</h3>
                <div className="mt-4 space-y-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Rows in editor</span>
                    <span className="font-medium tabular-nums">{rows.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Complete rows</span>
                    <span className="font-medium tabular-nums">{completedRows.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Saved holdings</span>
                    <span className="font-medium tabular-nums">{activePortfolio.holdings.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Total weight</span>
                    <span className="font-medium tabular-nums">{totalWeightPct.toFixed(2)}%</span>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-border/60 bg-card p-5">
                <h3 className="font-semibold text-foreground">Next step</h3>
                <p className="mt-3 text-sm text-muted-foreground">
                  Once this portfolio validates cleanly, open the risk workspace to compute rolling metrics and attribution from the backend.
                </p>
                <Button className="mt-4 w-full" asChild>
                  <Link href="/risk">Open risk workspace</Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
