"use client"

import { use, useEffect, useState } from "react"
import Link from "next/link"
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Search,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { validateTicker } from "@/lib/api"
import type { TickerValidationResult } from "@/lib/types"
import { getErrorMessage } from "@/lib/utils"

export default function TickerPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = use(params)
  const normalized = symbol.toUpperCase()
  const [currentSymbol, setCurrentSymbol] = useState(normalized)
  const [result, setResult] = useState<TickerValidationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    const run = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const response = await validateTicker(normalized)
        if (!cancelled) setResult(response)
      } catch (validationError) {
        if (!cancelled) {
          setError(getErrorMessage(validationError, "Unable to validate ticker"))
          setResult(null)
        }
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    void run()
    return () => {
      cancelled = true
    }
  }, [normalized])

  return (
    <div className="min-h-screen">
      <div className="border-b border-border/60 bg-card/50">
        <div className="px-6 lg:px-8 py-6">
          <nav className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
            <Link href="/tickers" className="hover:text-foreground transition-colors">Tickers</Link>
            <ArrowRight className="h-4 w-4" />
            <span className="text-foreground font-medium">{normalized}</span>
          </nav>

          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-surface-2 font-mono text-xl font-bold">
                  {normalized.slice(0, 2)}
                </div>
                <div>
                  <div className="flex items-center gap-3">
                    <h1 className="text-2xl font-semibold tracking-tight">{normalized}</h1>
                    {result && (
                      <Badge variant={result.is_valid ? "default" : "secondary"}>
                        {result.is_valid ? "Valid" : "Needs review"}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Backend pricing-history validation workspace
                  </p>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <div className="relative w-56">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={currentSymbol}
                  onChange={(e) => setCurrentSymbol(e.target.value.toUpperCase())}
                  className="pl-9 bg-surface-1"
                />
              </div>
              <Button asChild>
                <Link href={`/ticker/${encodeURIComponent(currentSymbol.trim() || normalized)}`}>
                  Check symbol
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="px-6 lg:px-8 py-8">
        {isLoading ? (
          <div className="rounded-xl border border-border/60 bg-card p-6 text-sm text-muted-foreground">
            Validating ticker against the backend...
          </div>
        ) : error ? (
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-amber-400 mt-0.5" />
              <div>
                <h2 className="font-semibold text-foreground">Validation failed</h2>
                <p className="mt-1 text-sm text-muted-foreground">{error}</p>
              </div>
            </div>
          </div>
        ) : result ? (
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2 rounded-xl border border-border/60 bg-card p-6">
              <div className="flex items-start gap-3">
                {result.is_valid ? (
                  <CheckCircle2 className="h-6 w-6 text-positive mt-0.5" />
                ) : (
                  <AlertTriangle className="h-6 w-6 text-amber-400 mt-0.5" />
                )}
                <div>
                  <h2 className="text-lg font-semibold text-foreground">
                    {result.is_valid ? "Ticker is usable for analysis" : "Ticker could not be confirmed"}
                  </h2>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {result.is_valid
                      ? "The backend returned enough pricing rows for this symbol to be considered valid."
                      : result.error || "The backend did not return enough pricing history to validate this symbol."}
                  </p>
                </div>
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <div className="rounded-lg bg-surface-2 p-4">
                  <p className="text-xs text-muted-foreground mb-1">Ticker</p>
                  <p className="text-xl font-semibold tabular-nums">{result.ticker}</p>
                </div>
                <div className="rounded-lg bg-surface-2 p-4">
                  <p className="text-xs text-muted-foreground mb-1">Rows returned</p>
                  <p className="text-xl font-semibold tabular-nums">{result.rows_returned}</p>
                </div>
                <div className="rounded-lg bg-surface-2 p-4">
                  <p className="text-xs text-muted-foreground mb-1">Status</p>
                  <p className="text-xl font-semibold">{result.is_valid ? "Valid" : "Invalid"}</p>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-xl border border-border/60 bg-card p-5">
                <div className="flex items-center gap-3">
                  <Database className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold text-foreground">What this means</h3>
                </div>
                <div className="mt-4 space-y-3 text-sm text-muted-foreground">
                  <p>This screen checks the backend&apos;s `tickers/validate` endpoint only.</p>
                  <p>It does not imply real-time quote coverage, company fundamentals, or news support.</p>
                  <p>Use it to preflight symbols before saving holdings.</p>
                </div>
              </div>

              <div className="rounded-xl border border-border/60 bg-card p-5">
                <h3 className="font-semibold text-foreground">Next action</h3>
                <Button className="mt-4 w-full" asChild>
                  <Link href="/holdings">Back to holdings editor</Link>
                </Button>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
