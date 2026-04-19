"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Info,
  Shield,
} from "lucide-react"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { usePortfolioData } from "@/components/providers/portfolio-provider"
import { getAttribution, getRisk } from "@/lib/api"
import type { AttributionResponse, RiskResponse } from "@/lib/types"
import { cn, fmtPct, fmtRatio, fmtSignedPct, getErrorMessage } from "@/lib/utils"

type RiskFormState = {
  period: string
  interval: string
  benchmark: string
  risk_free: string
  var_level: string
  trading_days: string
  rolling_window: string
  return_type: string
}

const defaultForm: RiskFormState = {
  period: "1y",
  interval: "1d",
  benchmark: "SPY",
  risk_free: "0.02",
  var_level: "0.95",
  trading_days: "252",
  rolling_window: "30",
  return_type: "simple",
}

export default function RiskPage() {
  const { portfolios, activePortfolioId, activePortfolio, selectPortfolio } = usePortfolioData()
  const [form, setForm] = useState<RiskFormState>(defaultForm)
  const [risk, setRisk] = useState<RiskResponse | null>(null)
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runAnalysis = async (options?: { silent?: boolean }) => {
    if (!activePortfolioId) {
      toast.error("Select a portfolio first")
      return
    }

    setIsLoading(true)
    setError(null)
    try {
      const riskParams = {
        period: form.period,
        interval: form.interval,
        benchmark: form.benchmark.trim().toUpperCase(),
        risk_free: Number(form.risk_free),
        var_level: Number(form.var_level),
        trading_days: Number(form.trading_days),
        rolling_window: Number(form.rolling_window),
        return_type: form.return_type,
      }

      const [riskResponse, attributionResponse] = await Promise.all([
        getRisk(activePortfolioId, riskParams),
        getAttribution(activePortfolioId, {
          period: form.period,
          interval: form.interval,
          trading_days: Number(form.trading_days),
        }),
      ])

      setRisk(riskResponse)
      setAttribution(attributionResponse)
      if (!options?.silent) {
        toast.success("Risk analysis updated")
      }
    } catch (analysisError) {
      const message = getErrorMessage(analysisError, "Unable to run risk analysis")
      setRisk(null)
      setAttribution(null)
      setError(message)
      toast.error(message)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (activePortfolioId && activePortfolio?.holdings.length) {
      void runAnalysis({ silent: true })
    } else {
      setRisk(null)
      setAttribution(null)
      setError(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePortfolioId])

  const riskCards = useMemo(() => ([
    {
      name: "Sharpe Ratio",
      value: risk ? fmtRatio(risk.metrics.sharpe_ratio) : "--",
      benchmark: risk?.metrics.benchmark_ticker ?? form.benchmark,
      description: "Risk-adjusted return",
      status: "good",
    },
    {
      name: "Sortino Ratio",
      value: risk ? fmtRatio(risk.metrics.sortino_ratio) : "--",
      benchmark: "Downside",
      description: "Downside-adjusted return",
      status: "good",
    },
    {
      name: "Beta",
      value: risk ? fmtRatio(risk.metrics.beta_vs_benchmark) : "--",
      benchmark: "vs benchmark",
      description: "Market sensitivity",
      status: "neutral",
    },
    {
      name: "Annual Return",
      value: risk ? fmtSignedPct(risk.metrics.annual_return) : "--",
      benchmark: "annualized",
      description: "Portfolio annual return",
      status: "good",
    },
    {
      name: "Max Drawdown",
      value: risk ? fmtPct(risk.metrics.max_drawdown) : "--",
      benchmark: "worst peak-trough",
      description: "Maximum drawdown",
      status: "warning",
    },
    {
      name: "Volatility",
      value: risk ? fmtPct(risk.metrics.volatility) : "--",
      benchmark: "annualized",
      description: "Standard deviation of returns",
      status: "good",
    },
  ]), [form.benchmark, risk])

  return (
    <TooltipProvider>
      <div className="min-h-screen">
        <div className="border-b border-border/60 bg-card/50">
          <div className="px-6 lg:px-8 py-6">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Shield className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-semibold tracking-tight">Risk Analytics</h1>
                    <p className="text-sm text-muted-foreground">
                      Live analysis from your backend metrics and attribution endpoints.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Select
                  value={activePortfolioId ? String(activePortfolioId) : undefined}
                  onValueChange={(value) => selectPortfolio(Number(value))}
                >
                  <SelectTrigger className="w-56 h-9">
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
                <Button onClick={() => void runAnalysis()} disabled={isLoading || !activePortfolioId}>
                  {isLoading ? "Running..." : "Run analysis"}
                </Button>
              </div>
            </div>
          </div>
        </div>

        <div className="px-6 lg:px-8 py-8 space-y-8">
          {!activePortfolio ? (
            <div className="rounded-xl border border-dashed border-border/60 bg-card p-10 text-center">
              <h2 className="text-lg font-semibold text-foreground">Choose a portfolio to analyze</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Risk analysis requires a saved portfolio with holdings that sum to 100%.
              </p>
              <Button className="mt-5" asChild>
                <Link href="/portfolio">Open portfolio workspace</Link>
              </Button>
            </div>
          ) : (
            <>
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {riskCards.map((metric) => (
                    <div
                      key={metric.name}
                      className="relative rounded-xl border border-border/60 bg-card p-4 overflow-hidden"
                    >
                      <div className={cn(
                        "absolute left-0 top-0 h-full w-0.5",
                        metric.status === "good" ? "bg-positive/60" :
                        metric.status === "warning" ? "bg-amber-400/60" : "bg-primary/60"
                      )} />
                      <div className="flex items-start justify-between mb-2">
                        <p className="text-xs font-medium text-muted-foreground">{metric.name}</p>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-xs">{metric.description}</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <p className="text-2xl font-semibold tabular-nums">{metric.value}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{metric.benchmark}</p>
                    </div>
                  ))}
                </div>

                <div className="rounded-xl border border-border/60 bg-card p-5">
                  <h2 className="font-semibold text-foreground">Analysis Controls</h2>
                  <div className="mt-4 space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-muted-foreground">Period</label>
                        <Select value={form.period} onValueChange={(value) => setForm((current) => ({ ...current, period: value }))}>
                          <SelectTrigger className="mt-1 h-10">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="6mo">6 months</SelectItem>
                            <SelectItem value="1y">1 year</SelectItem>
                            <SelectItem value="2y">2 years</SelectItem>
                            <SelectItem value="5y">5 years</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">Interval</label>
                        <Select value={form.interval} onValueChange={(value) => setForm((current) => ({ ...current, interval: value }))}>
                          <SelectTrigger className="mt-1 h-10">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1d">Daily</SelectItem>
                            <SelectItem value="1wk">Weekly</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div>
                      <label className="text-xs text-muted-foreground">Benchmark ticker</label>
                      <Input
                        value={form.benchmark}
                        onChange={(e) => setForm((current) => ({ ...current, benchmark: e.target.value.toUpperCase() }))}
                        className="mt-1 h-10 bg-surface-1"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-muted-foreground">Risk-free rate</label>
                        <Input
                          value={form.risk_free}
                          onChange={(e) => setForm((current) => ({ ...current, risk_free: e.target.value }))}
                          className="mt-1 h-10 bg-surface-1"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">VaR level</label>
                        <Input
                          value={form.var_level}
                          onChange={(e) => setForm((current) => ({ ...current, var_level: e.target.value }))}
                          className="mt-1 h-10 bg-surface-1"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-muted-foreground">Trading days</label>
                        <Input
                          value={form.trading_days}
                          onChange={(e) => setForm((current) => ({ ...current, trading_days: e.target.value }))}
                          className="mt-1 h-10 bg-surface-1"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-muted-foreground">Rolling window</label>
                        <Input
                          value={form.rolling_window}
                          onChange={(e) => setForm((current) => ({ ...current, rolling_window: e.target.value }))}
                          className="mt-1 h-10 bg-surface-1"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="text-xs text-muted-foreground">Return type</label>
                      <Select value={form.return_type} onValueChange={(value) => setForm((current) => ({ ...current, return_type: value }))}>
                        <SelectTrigger className="mt-1 h-10">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="simple">Simple</SelectItem>
                          <SelectItem value="log">Log</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>
              </div>

              {error ? (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-6">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-amber-400 mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-foreground">Analysis failed</h3>
                      <p className="mt-1 text-sm text-muted-foreground">{error}</p>
                      <Button className="mt-4" asChild>
                        <Link href="/holdings">Review holdings</Link>
                      </Button>
                    </div>
                  </div>
                </div>
              ) : risk ? (
                <>
                  <div className="grid gap-6 lg:grid-cols-2">
                    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
                      <div className="px-6 py-4 border-b border-border/50">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="font-semibold">Rolling Volatility</h3>
                            <p className="text-sm text-muted-foreground">Backend-derived rolling annualized volatility</p>
                          </div>
                          <Badge variant="secondary">Window {risk.rolling?.window ?? form.rolling_window}</Badge>
                        </div>
                      </div>
                      <div className="p-6">
                        <ChartContainer
                          config={{
                            value: { label: "Volatility", color: "var(--color-primary)" },
                          }}
                          className="h-[280px] w-full"
                        >
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={risk.rolling?.volatility ?? []}>
                              <defs>
                                <linearGradient id="volGradient" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.24} />
                                  <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0} />
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                              <XAxis dataKey="date" hide />
                              <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} axisLine={false} tickLine={false} />
                              <ChartTooltip content={<ChartTooltipContent />} />
                              <Area type="monotone" dataKey="value" stroke="var(--color-primary)" strokeWidth={2} fill="url(#volGradient)" />
                            </AreaChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                      </div>
                    </div>

                    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
                      <div className="px-6 py-4 border-b border-border/50">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="font-semibold">Rolling Sharpe</h3>
                            <p className="text-sm text-muted-foreground">Risk-adjusted performance through time</p>
                          </div>
                          <Badge variant="secondary">Benchmark {risk.metrics.benchmark_ticker}</Badge>
                        </div>
                      </div>
                      <div className="p-6">
                        <ChartContainer
                          config={{
                            value: { label: "Sharpe", color: "var(--color-chart-2)" },
                          }}
                          className="h-[280px] w-full"
                        >
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={risk.rolling?.sharpe ?? []}>
                              <defs>
                                <linearGradient id="sharpeGradient" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="var(--color-chart-2)" stopOpacity={0.24} />
                                  <stop offset="95%" stopColor="var(--color-chart-2)" stopOpacity={0} />
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                              <XAxis dataKey="date" hide />
                              <YAxis axisLine={false} tickLine={false} />
                              <ReferenceLine y={0} stroke="var(--color-border)" />
                              <ChartTooltip content={<ChartTooltipContent />} />
                              <Area type="monotone" dataKey="value" stroke="var(--color-chart-2)" strokeWidth={2} fill="url(#sharpeGradient)" />
                            </AreaChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-6 lg:grid-cols-3">
                    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
                      <div className="px-6 py-4 border-b border-border/50">
                        <h3 className="font-semibold">Value at Risk</h3>
                        <p className="text-sm text-muted-foreground">Current daily estimate from API metrics</p>
                      </div>
                      <div className="p-6 space-y-4">
                        <div className="rounded-lg bg-surface-2 p-4">
                          <p className="text-xs text-muted-foreground mb-1">Configured confidence</p>
                          <p className="text-xl font-semibold tabular-nums">{(Number(form.var_level) * 100).toFixed(0)}%</p>
                        </div>
                        <div className="rounded-lg bg-surface-2 p-4">
                          <p className="text-xs text-muted-foreground mb-1">Estimated daily VaR</p>
                          <p className="text-xl font-semibold tabular-nums text-negative">
                            {fmtPct(risk.metrics.value_at_risk)}
                          </p>
                        </div>
                        <div className="rounded-lg bg-surface-2 p-4">
                          <p className="text-xs text-muted-foreground mb-1">Worst day</p>
                          <p className="text-xl font-semibold tabular-nums text-negative">
                            {fmtPct(risk.metrics.worst_day)}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="lg:col-span-2 rounded-xl border border-border/60 bg-card overflow-hidden">
                      <div className="px-6 py-4 border-b border-border/50">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="font-semibold">Sector Risk Contribution</h3>
                            <p className="text-sm text-muted-foreground">Attribution by sector from covariance-based contribution</p>
                          </div>
                          <Badge variant="secondary">{attribution?.sector_attribution.length ?? 0} sectors</Badge>
                        </div>
                      </div>
                      <div className="p-6">
                        <ChartContainer
                          config={{
                            trc_pct: { label: "Risk Contribution", color: "var(--color-chart-3)" },
                          }}
                          className="h-[280px] w-full"
                        >
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={
                                attribution?.sector_attribution.map((item) => ({
                                  sector: item.sector,
                                  trc_pct: item.trc_pct,
                                })) ?? []
                              }
                              layout="vertical"
                              margin={{ top: 0, right: 20, left: 80, bottom: 0 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" horizontal={false} />
                              <XAxis type="number" tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} axisLine={false} tickLine={false} />
                              <YAxis type="category" dataKey="sector" axisLine={false} tickLine={false} width={72} />
                              <ChartTooltip content={<ChartTooltipContent />} />
                              <Bar dataKey="trc_pct" fill="var(--color-chart-3)" radius={[0, 4, 4, 0]} barSize={14} />
                            </BarChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-6 lg:grid-cols-2">
                    <div className="rounded-xl border border-border/60 bg-card overflow-hidden">
                      <div className="px-6 py-4 border-b border-border/50">
                        <h3 className="font-semibold">Ticker Attribution</h3>
                        <p className="text-sm text-muted-foreground">Marginal and total risk contribution by holding</p>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-border bg-surface-2/50">
                              <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Ticker</th>
                              <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Weight</th>
                              <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">TRC %</th>
                              <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sector</th>
                            </tr>
                          </thead>
                          <tbody>
                            {attribution?.attribution.map((item) => (
                              <tr key={item.ticker} className="border-b border-border/40 last:border-0">
                                <td className="px-4 py-3 font-medium">{item.ticker}</td>
                                <td className="px-4 py-3 text-right tabular-nums">{fmtPct(item.weight)}</td>
                                <td className="px-4 py-3 text-right tabular-nums">{fmtPct(item.trc_pct)}</td>
                                <td className="px-4 py-3 text-right text-sm text-muted-foreground">{item.sector}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div className="rounded-xl border border-border/60 bg-card p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <BarChart3 className="h-5 w-5 text-primary" />
                        <h3 className="font-semibold">Analysis Notes</h3>
                      </div>
                      <div className="space-y-4 text-sm">
                        <div className="flex items-start justify-between gap-4">
                          <span className="text-muted-foreground">Annual return</span>
                          <span className={cn(
                            "font-medium tabular-nums flex items-center gap-1",
                            (risk.metrics.annual_return ?? 0) >= 0 ? "text-positive" : "text-negative"
                          )}>
                            {(risk.metrics.annual_return ?? 0) >= 0 ? <ArrowUpRight className="h-4 w-4" /> : <ArrowDownRight className="h-4 w-4" />}
                            {fmtSignedPct(risk.metrics.annual_return)}
                          </span>
                        </div>
                        <div className="flex items-start justify-between gap-4">
                          <span className="text-muted-foreground">Tickers dropped</span>
                          <span className="font-medium">{risk.tickers_dropped.length}</span>
                        </div>
                        <div className="flex items-start justify-between gap-4">
                          <span className="text-muted-foreground">Covariance matrix assets</span>
                          <span className="font-medium">{Object.keys(risk.metrics.covariance_matrix).length}</span>
                        </div>
                        <div className="pt-4 border-t border-border/50">
                          <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2">Backend summary</p>
                          <p className="text-foreground leading-relaxed">
                            {attribution?.summary ?? "Run analysis to generate attribution commentary."}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="rounded-xl border border-border/60 bg-card p-6 text-sm text-muted-foreground">
                  Waiting for the first successful analysis run.
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </TooltipProvider>
  )
}
