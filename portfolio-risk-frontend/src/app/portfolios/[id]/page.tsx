"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { AuthGate } from "@/components/auth/AuthGate";
import { benchmarks, intervals, periods, returnTypes } from "@/lib/design-system";
import { getAttribution, getPortfolio, getRisk } from "@/lib/api";
import type { AttributionResponse, PortfolioOut, RiskConfig, RiskResponse } from "@/lib/types";
import { fmtPct, getErrorMessage } from "@/lib/utils";
import { AttributionPanel } from "@/components/analytics/AttributionPanel";
import { CovarianceHeatmap } from "@/components/analytics/CovarianceHeatmap";
import { RiskKpis } from "@/components/analytics/RiskKpis";
import { RollingCharts } from "@/components/analytics/RollingCharts";
import { SectorChart } from "@/components/analytics/SectorChart";
import { HoldingsEditor } from "@/components/portfolio/HoldingsEditor";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input, Select } from "@/components/ui/Input";
import { EmptyState, ErrorState, SkeletonBlock } from "@/components/ui/StatePanel";

const defaultConfig: RiskConfig = {
  period: "1y",
  interval: "1d",
  benchmark: "SPY",
  return_type: "simple",
  risk_free: 0.02,
  var_level: 0.95,
  trading_days: 252,
  rolling_window: 30,
};

export default function PortfolioDetailPage() {
  const params = useParams<{ id: string }>();
  const portfolioId = Number(params.id);

  const [portfolio, setPortfolio] = useState<PortfolioOut | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null);
  const [config, setConfig] = useState<RiskConfig>(defaultConfig);
  const [loading, setLoading] = useState(true);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadAll(nextConfig = config) {
    setLoading(true);
    setAnalyticsLoading(true);
    setError(null);
    try {
      const nextPortfolio = await getPortfolio(portfolioId);
      setPortfolio(nextPortfolio);
      const [nextRisk, nextAttribution] = await Promise.all([
        getRisk(portfolioId, nextConfig),
        getAttribution(portfolioId, {
          period: nextConfig.period,
          interval: nextConfig.interval,
          trading_days: nextConfig.trading_days,
        }),
      ]);
      setRisk(nextRisk);
      setAttribution(nextAttribution);
    } catch (nextError: unknown) {
      setError(getErrorMessage(nextError, "Unable to load portfolio workspace"));
      setRisk(null);
      setAttribution(null);
    } finally {
      setLoading(false);
      setAnalyticsLoading(false);
    }
  }

  useEffect(() => {
    async function initialLoad() {
      setLoading(true);
      setAnalyticsLoading(true);
      setError(null);
      try {
        const nextPortfolio = await getPortfolio(portfolioId);
        setPortfolio(nextPortfolio);
        const [nextRisk, nextAttribution] = await Promise.all([
          getRisk(portfolioId, defaultConfig),
          getAttribution(portfolioId, {
            period: defaultConfig.period,
            interval: defaultConfig.interval,
            trading_days: defaultConfig.trading_days,
          }),
        ]);
        setRisk(nextRisk);
        setAttribution(nextAttribution);
      } catch (nextError: unknown) {
        setError(getErrorMessage(nextError, "Unable to load portfolio workspace"));
        setRisk(null);
        setAttribution(null);
      } finally {
        setLoading(false);
        setAnalyticsLoading(false);
      }
    }

    void initialLoad();
  }, [portfolioId]);

  return (
    <AuthGate>
      <div className="space-y-5">
        <div className="flex items-center justify-between gap-3">
          <Link
            href="/portfolios"
            className="focus-ring inline-flex items-center gap-2 rounded-[12px] border border-[var(--border)] bg-[var(--bg-elevated)] px-3.5 py-2.5 text-sm font-medium text-[var(--text)] transition hover:bg-[var(--bg-muted)]"
          >
            <ArrowLeft size={16} />
            Back to overview
          </Link>
          {portfolio ? <Badge tone="accent">{portfolio.holdings.length} holdings</Badge> : null}
        </div>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.15fr_0.85fr]">
          <Card className="rounded-[20px]">
            <CardHeader className="block">
              <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Portfolio Workspace</div>
              <CardTitle className="mt-2 text-2xl tracking-[-0.03em]">
                {portfolio?.name ?? (loading ? "Loading portfolio..." : "Portfolio")}
              </CardTitle>
              <CardDescription>
                Build holdings, tune analysis controls, and review premium risk diagnostics in one continuous workflow.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Portfolio id</div>
                <div className="metric-value mt-3 text-3xl font-semibold tracking-[-0.04em]">{portfolioId}</div>
              </div>
              <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Benchmark</div>
                <div className="metric-value mt-3 text-3xl font-semibold tracking-[-0.04em]">{config.benchmark}</div>
              </div>
              <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Dropped tickers</div>
                <div className="metric-value mt-3 text-3xl font-semibold tracking-[-0.04em]">{risk?.tickers_dropped.length ?? 0}</div>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-[20px]">
            <CardHeader className="block">
              <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Analysis Controls</div>
              <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Tune the calculation context</CardTitle>
              <CardDescription>These controls map directly to the backend risk and attribution endpoints.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Period</span>
                <Select value={config.period} onChange={(event) => setConfig((current) => ({ ...current, period: event.target.value }))}>
                  {periods.map((period) => (
                    <option key={period.value} value={period.value}>
                      {period.label}
                    </option>
                  ))}
                </Select>
              </label>

              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Interval</span>
                <Select value={config.interval} onChange={(event) => setConfig((current) => ({ ...current, interval: event.target.value }))}>
                  {intervals.map((interval) => (
                    <option key={interval.value} value={interval.value}>
                      {interval.label}
                    </option>
                  ))}
                </Select>
              </label>

              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Benchmark</span>
                <Select value={config.benchmark} onChange={(event) => setConfig((current) => ({ ...current, benchmark: event.target.value }))}>
                  {benchmarks.map((benchmark) => (
                    <option key={benchmark} value={benchmark}>
                      {benchmark}
                    </option>
                  ))}
                </Select>
              </label>

              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Return type</span>
                <Select
                  value={config.return_type}
                  onChange={(event) => setConfig((current) => ({ ...current, return_type: event.target.value }))}
                >
                  {returnTypes.map((returnType) => (
                    <option key={returnType.value} value={returnType.value}>
                      {returnType.label}
                    </option>
                  ))}
                </Select>
              </label>

              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Risk-free rate</span>
                <Input
                  value={String(config.risk_free)}
                  onChange={(event) => setConfig((current) => ({ ...current, risk_free: Number(event.target.value) || 0 }))}
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">VaR level</span>
                <Input
                  value={String(config.var_level)}
                  onChange={(event) => setConfig((current) => ({ ...current, var_level: Number(event.target.value) || 0.95 }))}
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Trading days</span>
                <Input
                  value={String(config.trading_days)}
                  onChange={(event) => setConfig((current) => ({ ...current, trading_days: Number(event.target.value) || 252 }))}
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Rolling window</span>
                <Input
                  value={String(config.rolling_window)}
                  onChange={(event) => setConfig((current) => ({ ...current, rolling_window: Number(event.target.value) || 30 }))}
                />
              </label>

              <div className="sm:col-span-2">
                <Button className="w-full" onClick={() => loadAll(config)} loading={analyticsLoading}>
                  Refresh Analytics
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>

        {error ? <ErrorState title="Workspace unavailable" body={error} onRetry={() => loadAll(config)} /> : null}

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[0.95fr_1.05fr]">
          {portfolio ? (
            <HoldingsEditor
              portfolio={portfolio}
              onUpdated={(nextPortfolio) => {
                setPortfolio(nextPortfolio);
                loadAll(config);
              }}
            />
          ) : (
            <SkeletonBlock className="h-[520px]" />
          )}

          <Card className="rounded-[20px]">
            <CardHeader className="block">
              <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Portfolio Health</div>
              <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Interpretation layer</CardTitle>
              <CardDescription>Use this panel to quickly judge portfolio quality before diving into individual charts.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!risk || !attribution ? (
                <EmptyState
                  title="Analytics not ready"
                  body="Save holdings and refresh analytics to populate the health panel with meaningful diagnostics."
                />
              ) : (
                <>
                  <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                    <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Health Readout</div>
                    <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">{attribution.summary}</p>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                      <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Benchmark beta</div>
                      <div className="metric-value mt-3 text-3xl font-semibold tracking-[-0.04em]">{risk.metrics.beta_vs_benchmark?.toFixed(2) ?? "--"}</div>
                    </div>
                    <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                      <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Top sector risk</div>
                      <div className="mt-3 text-lg font-semibold">{attribution.sector_attribution[0]?.sector ?? "Unknown"}</div>
                      <div className="mt-1 text-sm text-[var(--text-soft)]">
                        {attribution.sector_attribution[0] ? fmtPct(attribution.sector_attribution[0].trc_pct) : "--"}
                      </div>
                    </div>
                  </div>

                  {risk.tickers_dropped.length > 0 ? (
                    <div className="rounded-[16px] border border-[#e7d3a6] bg-[var(--amber-soft)] p-4">
                      <div className="text-sm font-semibold text-[var(--amber)]">Dropped tickers</div>
                      <div className="mt-2 text-sm leading-6 text-[#7d5a16]">
                        {risk.tickers_dropped.join(", ")} returned no usable price history for the selected window.
                      </div>
                    </div>
                  ) : null}
                </>
              )}
            </CardContent>
          </Card>
        </section>

        {analyticsLoading ? (
          <div className="space-y-4">
            <SkeletonBlock className="h-[124px]" />
            <SkeletonBlock className="h-[360px]" />
            <SkeletonBlock className="h-[360px]" />
          </div>
        ) : risk && attribution ? (
          <>
            <RiskKpis risk={risk} />
            <RollingCharts risk={risk} />
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[0.9fr_1.1fr]">
              <SectorChart attribution={attribution} />
              <CovarianceHeatmap matrix={risk.metrics.covariance_matrix} />
            </div>
            <AttributionPanel attribution={attribution} />
          </>
        ) : (
          <EmptyState
            title="Analytics unavailable"
            body="Add valid holdings and refresh analytics to unlock the full dashboard."
          />
        )}
      </div>
    </AuthGate>
  );
}
