"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  RefreshCw,
  Shield,
  TrendingUp,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { MetricCard } from "@/components/dashboard/metric-card"
import { ChartPanel, LegendItem } from "@/components/dashboard/chart-panel"
import { DataTable } from "@/components/dashboard/data-table"
import { usePortfolioData } from "@/components/providers/portfolio-provider"
import { getAttribution, getRisk } from "@/lib/api"
import type { AttributionResponse, RiskResponse } from "@/lib/types"
import { fmtPct, fmtRatio, fmtSignedPct, getErrorMessage } from "@/lib/utils"

const defaultRiskParams = {
  period: "1y",
  interval: "1d",
  benchmark: "SPY",
  return_type: "simple",
  risk_free: 0.02,
  var_level: 0.95,
  trading_days: 252,
  rolling_window: 30,
}

function buildSparkline(points?: { value: number }[]) {
  return points?.slice(-12).map((point) => point.value) ?? []
}

export default function DashboardPage() {
  const { activePortfolio, activePortfolioId, refreshPortfolios, portfolios, isLoading: portfoliosLoading } = usePortfolioData()
  const [risk, setRisk] = useState<RiskResponse | null>(null)
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!activePortfolioId) {
      setRisk(null)
      setAttribution(null)
      setError(null)
      return
    }

    let cancelled = false

    const load = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const [riskResponse, attributionResponse] = await Promise.all([
          getRisk(activePortfolioId, defaultRiskParams),
          getAttribution(activePortfolioId, {
            period: defaultRiskParams.period,
            interval: defaultRiskParams.interval,
            trading_days: defaultRiskParams.trading_days,
          }),
        ])
        if (!cancelled) {
          setRisk(riskResponse)
          setAttribution(attributionResponse)
        }
      } catch (loadError) {
        if (!cancelled) {
          setRisk(null)
          setAttribution(null)
          setError(getErrorMessage(loadError, "Unable to load dashboard metrics"))
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false)
        }
      }
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [activePortfolioId])

  const totalWeight = useMemo(
    () => activePortfolio?.holdings.reduce((sum, holding) => sum + holding.weight, 0) ?? 0,
    [activePortfolio],
  )

  const topHoldings = useMemo(() => {
    if (!activePortfolio) return []
    return [...activePortfolio.holdings]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 8)
      .map((holding) => ({
        ticker: holding.ticker,
        weight: holding.weight,
        contribution:
          attribution?.attribution.find((item) => item.ticker === holding.ticker)?.trc_pct ?? null,
        sector:
          attribution?.attribution.find((item) => item.ticker === holding.ticker)?.sector ?? "Unclassified",
      }))
  }, [activePortfolio, attribution])

  if (portfoliosLoading) {
    return <div className="text-sm text-muted-foreground">Loading dashboard workspace...</div>
  }

  if (portfolios.length === 0) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center">
        <div className="max-w-xl rounded-2xl border border-border/60 bg-card p-8 text-center">
          <Badge variant="secondary" className="mb-4">No portfolio yet</Badge>
          <h1 className="text-2xl font-semibold text-foreground">Create your first portfolio</h1>
          <p className="mt-3 text-muted-foreground">
            Your backend is ready for portfolio creation, holdings upload, ticker validation, and full risk analysis.
          </p>
          <div className="mt-6 flex items-center justify-center gap-3">
            <Button asChild>
              <Link href="/portfolio">Open portfolio workspace</Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/holdings">Go to holdings editor</Link>
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            {activePortfolio?.name ?? "Select a portfolio"} overview
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" className="gap-2" onClick={() => void refreshPortfolios()}>
            <RefreshCw className="h-4 w-4" />
            Refresh portfolios
          </Button>
          <Button size="sm" className="gap-2" asChild>
            <Link href="/risk">
              Full Analysis
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <MetricCard
            label="Holdings"
            value={activePortfolio?.holdings.length ?? 0}
            change={undefined}
            icon={<TrendingUp className="h-4 w-4" />}
            sparkline={activePortfolio?.holdings.map((holding) => holding.weight)}
            variant="highlight"
          />
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <MetricCard
            label="Sharpe Ratio"
            value={risk ? fmtRatio(risk.metrics.sharpe_ratio) : "--"}
            change={undefined}
            icon={<Activity className="h-4 w-4" />}
            sparkline={buildSparkline(risk?.rolling?.sharpe)}
          />
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <MetricCard
            label="Volatility"
            value={risk ? fmtPct(risk.metrics.volatility) : "--"}
            icon={<BarChart3 className="h-4 w-4" />}
            sparkline={buildSparkline(risk?.rolling?.volatility)}
          />
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <MetricCard
            label="VaR"
            value={risk ? fmtPct(risk.metrics.value_at_risk) : "--"}
            icon={<Shield className="h-4 w-4" />}
            sparkline={buildSparkline(risk?.rolling?.beta)}
          />
        </motion.div>
      </div>

      {!activePortfolio?.holdings.length ? (
        <div className="rounded-2xl border border-dashed border-border/60 bg-card p-8">
          <h2 className="text-lg font-semibold text-foreground">This portfolio has no holdings yet</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Add weights that sum to 100% in the holdings editor before requesting risk analytics.
          </p>
          <Button className="mt-5" asChild>
            <Link href="/holdings">Open holdings editor</Link>
          </Button>
        </div>
      ) : error ? (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/5 p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-amber-400 mt-0.5" />
            <div>
              <h2 className="text-base font-semibold text-foreground">Analysis needs attention</h2>
              <p className="mt-1 text-sm text-muted-foreground">{error}</p>
              <div className="mt-4 flex gap-3">
                <Button variant="outline" asChild>
                  <Link href="/holdings">Review holdings</Link>
                </Button>
                <Button asChild>
                  <Link href="/risk">Open risk workspace</Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <ChartPanel
                title="Rolling Analytics"
                subtitle={`30-day metrics for ${activePortfolio?.name}`}
                height="lg"
                legend={
                  <>
                    <LegendItem color="hsl(var(--chart-1))" label="Volatility" value={risk ? fmtPct(risk.metrics.volatility) : "--"} />
                    <LegendItem color="hsl(var(--chart-2))" label="Sharpe" value={risk ? fmtRatio(risk.metrics.sharpe_ratio) : "--"} />
                    <LegendItem color="hsl(var(--chart-3))" label="Beta" value={risk ? fmtRatio(risk.metrics.beta_vs_benchmark) : "--"} />
                  </>
                }
              >
                <div className="grid h-full grid-cols-3 gap-4">
                  {[
                    { label: "Volatility", points: risk?.rolling?.volatility ?? [], color: "bg-chart-1/70" },
                    { label: "Sharpe", points: risk?.rolling?.sharpe ?? [], color: "bg-chart-2/70" },
                    { label: "Beta", points: risk?.rolling?.beta ?? [], color: "bg-chart-3/70" },
                  ].map((series) => {
                    const values = series.points.map((point) => point.value)
                    const max = Math.max(...values, 1)
                    const min = Math.min(...values, 0)
                    const range = max - min || 1

                    return (
                      <div key={series.label} className="rounded-xl border border-border/50 bg-surface-1/40 p-4">
                        <div className="mb-3 flex items-center justify-between">
                          <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                            {series.label}
                          </p>
                          <Badge variant="secondary" className="text-[10px]">
                            {series.points.at(-1)?.date?.slice(0, 10) ?? "latest"}
                          </Badge>
                        </div>
                        <div className="flex h-[180px] items-end gap-1">
                          {series.points.slice(-24).map((point, index) => {
                            const height = ((point.value - min) / range) * 100
                            return (
                              <div
                                key={`${series.label}-${index}`}
                                className={`flex-1 rounded-t-sm ${series.color}`}
                                style={{ height: `${Math.max(height, 8)}%` }}
                                title={`${point.date}: ${point.value.toFixed(4)}`}
                              />
                            )
                          })}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </ChartPanel>
            </div>

            <div className="space-y-6">
              <ChartPanel
                title="Portfolio Composition"
                subtitle="Current weights"
                height="sm"
                showDefaultActions={false}
              >
                <div className="space-y-3">
                  {topHoldings.slice(0, 5).map((holding) => (
                    <div key={holding.ticker}>
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
              </ChartPanel>

              <div className="rounded-lg border border-border/60 bg-card p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-foreground">Health Summary</h3>
                  <Badge variant="secondary">{fmtPct(totalWeight, 1)} allocated</Badge>
                </div>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Benchmark</span>
                    <span className="font-medium">{risk?.metrics.benchmark_ticker ?? defaultRiskParams.benchmark}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Tickers used</span>
                    <span className="font-medium">{risk?.tickers_used.length ?? activePortfolio?.holdings.length ?? 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Dropped tickers</span>
                    <span className="font-medium">{risk?.tickers_dropped.length ?? 0}</span>
                  </div>
                  <div className="pt-3 border-t border-border/50">
                    <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2">Attribution summary</p>
                    <p className="text-sm text-foreground">
                      {attribution?.summary ?? "Risk attribution will appear after analysis runs successfully."}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-foreground">Current Holdings</h2>
                <Button variant="ghost" size="sm" className="gap-1" asChild>
                  <Link href="/holdings">
                    Edit holdings
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </div>
              <DataTable
                columns={[
                  {
                    key: "ticker",
                    header: "Ticker",
                    render: (row) => (
                      <div>
                        <span className="font-medium text-foreground">{String(row.ticker)}</span>
                        <p className="text-xs text-muted-foreground">{String(row.sector)}</p>
                      </div>
                    ),
                  },
                  {
                    key: "weight",
                    header: "Weight",
                    align: "right",
                    render: (row) => <span className="tabular-nums">{fmtPct(Number(row.weight))}</span>,
                  },
                  {
                    key: "contribution",
                    header: "Risk Contribution",
                    align: "right",
                    render: (row) => (
                      <span className="tabular-nums">
                        {row.contribution === null ? "--" : fmtPct(Number(row.contribution))}
                      </span>
                    ),
                  },
                ]}
                data={topHoldings}
              />
            </div>

            <div className="space-y-6">
              <div className="rounded-lg border border-border/60 bg-card p-5">
                <h3 className="text-sm font-semibold text-foreground mb-4">Key Risk Metrics</h3>
                <div className="space-y-3">
                  {[
                    ["Annual Return", risk ? fmtSignedPct(risk.metrics.annual_return) : "--"],
                    ["Max Drawdown", risk ? fmtPct(risk.metrics.max_drawdown) : "--"],
                    ["Worst Day", risk ? fmtPct(risk.metrics.worst_day) : "--"],
                    ["Beta", risk ? fmtRatio(risk.metrics.beta_vs_benchmark) : "--"],
                  ].map(([label, value]) => (
                    <div key={label} className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">{label}</span>
                      <span className="text-sm font-medium tabular-nums">{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-border/60 bg-card p-5">
                <h3 className="text-sm font-semibold text-foreground mb-4">Next Steps</h3>
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p>Use the holdings editor to rebalance weights and validate tickers before analysis.</p>
                  <p>Open the risk page for rolling metrics, VaR, and sector attribution.</p>
                  <p>Review dropped tickers immediately if the backend returns incomplete market data.</p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {isLoading && (
        <div className="text-sm text-muted-foreground">Refreshing analytics from the backend...</div>
      )}
    </div>
  )
}
