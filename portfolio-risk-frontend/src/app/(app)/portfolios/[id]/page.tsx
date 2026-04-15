"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft, TriangleAlert } from "lucide-react";
import { Chart } from "@/components/analytics/Chart";
import { ChartPanel } from "@/components/dashboard/ChartPanel";
import { ControlPanel } from "@/components/dashboard/ControlPanel";
import { DataTable } from "@/components/dashboard/DataTable";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { HoldingsEditor } from "@/components/portfolio/HoldingsEditor";
import { Button } from "@/components/ui/Button";
import { Input, Select } from "@/components/ui/Input";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/StatePanel";
import { chartPalette, benchmarks, intervals, periods, returnTypes } from "@/lib/design-system";
import { getAttribution, getPortfolio, getRisk } from "@/lib/api";
import type { AttributionResponse, PortfolioOut, RiskConfig, RiskResponse } from "@/lib/types";
import { fmtPct, fmtRatio, getErrorMessage } from "@/lib/utils";

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

function sample(points: Array<{ value: number }>, fallback: number) {
  const raw = points.slice(-18).map((point) => point.value);
  if (raw.length >= 6) return raw;
  return Array.from({ length: 18 }, (_, index) => fallback + Math.sin(index / 2.2) * fallback * 0.08);
}

export default function PortfolioWorkspacePage() {
  const params = useParams<{ id: string }>();
  const portfolioId = Number(params.id);

  const [portfolio, setPortfolio] = useState<PortfolioOut | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null);
  const [config, setConfig] = useState<RiskConfig>(defaultConfig);
  const [loading, setLoading] = useState(true);
  const [savingAnalytics, setSavingAnalytics] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadAll(nextConfig = config) {
    setLoading(true);
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
    } catch (nextError) {
      setError(getErrorMessage(nextError, "Unable to load the portfolio workspace."));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    async function loadInitial() {
      setLoading(true);
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
      } catch (nextError) {
        setError(getErrorMessage(nextError, "Unable to load the portfolio workspace."));
      } finally {
        setLoading(false);
      }
    }

    void loadInitial();
  }, [portfolioId]);

  const volatilitySeries = sample(risk?.rolling?.volatility ?? [], risk?.metrics.volatility ?? 0.15);
  const sharpeSeries = sample(risk?.rolling?.sharpe ?? [], risk?.metrics.sharpe_ratio ?? 1);
  const betaSeries = sample(risk?.rolling?.beta ?? [], risk?.metrics.beta_vs_benchmark ?? 1);

  const allocationRows = useMemo(
    () =>
      Object.entries(risk?.weights_used ?? {}).map(([ticker, weight]) => ({
        ticker,
        weight,
      })),
    [risk?.weights_used],
  );

  if (loading && !portfolio) return <LoadingState title="Loading portfolio workspace..." />;

  return (
    <div className="space-y-5">
      {error ? <ErrorState title="Workspace unavailable" body={error} /> : null}

      <section className="panel hero-panel rounded-[30px] p-6">
        <div className="grid gap-5 xl:grid-cols-[0.92fr_1.08fr]">
          <div>
            <Link href="/portfolios" className="inline-flex items-center gap-2 text-sm text-[var(--text-soft)] transition hover:text-[var(--text)]">
              <ArrowLeft size={15} />
              Back to portfolio library
            </Link>
            <div className="mt-5 flex flex-wrap gap-2">
              <span className="glass-strip rounded-full px-3 py-1.5 text-xs text-[var(--text-soft)]">Portfolio #{portfolioId}</span>
              <span className="glass-strip rounded-full px-3 py-1.5 text-xs text-[var(--text-soft)]">Benchmark {config.benchmark}</span>
              {risk?.tickers_dropped.length ? (
                <span className="rounded-full border border-[rgba(255,183,111,0.18)] bg-[rgba(255,183,111,0.12)] px-3 py-1.5 text-xs text-[var(--accent-3)]">
                  {risk.tickers_dropped.length} dropped
                </span>
              ) : null}
            </div>
            <h2 className="mt-4 text-4xl font-semibold tracking-[-0.06em]">{portfolio?.name ?? "Portfolio"}</h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
              This workspace keeps construction, rolling analytics, allocation, and benchmark sensitivity close together so you can move from edit to diagnosis without losing context.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <MetricCard label="Sharpe" value={fmtRatio(risk?.metrics.sharpe_ratio)} detail="risk-adjusted return" points={sharpeSeries} tone="accent" />
            <MetricCard label="Annual Return" value={fmtPct(risk?.metrics.annual_return ?? Number.NaN)} detail="annualized" points={sharpeSeries} tone="good" />
            <MetricCard label="Volatility" value={fmtPct(risk?.metrics.volatility ?? Number.NaN)} detail="annualized" points={volatilitySeries} tone="warn" />
            <MetricCard label="VaR" value={fmtPct(risk?.metrics.value_at_risk ?? Number.NaN)} detail="95% confidence" points={betaSeries} tone="bad" />
          </div>
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-[1.26fr_0.74fr]">
        <ChartPanel kicker="Rolling Metrics" title="Performance, risk, and benchmark sensitivity" description="A wide analytics surface keeps the main review anchored to time-series behavior.">
          <div className="rounded-[22px] border border-[var(--line)] bg-[rgba(6,10,20,0.52)] p-3">
            <Chart
              style={{ height: 410 }}
              option={{
                animationDuration: 650,
                tooltip: {
                  trigger: "axis",
                  backgroundColor: chartPalette.tooltipBg,
                  borderColor: chartPalette.border,
                  borderWidth: 1,
                  textStyle: { color: chartPalette.tooltipText },
                },
                legend: {
                  top: 0,
                  right: 6,
                  textStyle: { color: chartPalette.axis, fontSize: 11 },
                },
                grid: { top: 30, left: 50, right: 24, bottom: 42 },
                xAxis: {
                  type: "category",
                  data: risk?.rolling?.volatility.map((point) => point.date) ?? [],
                  boundaryGap: false,
                  axisLabel: { color: chartPalette.axis, fontSize: 10 },
                  axisLine: { lineStyle: { color: chartPalette.border } },
                },
                yAxis: {
                  type: "value",
                  axisLabel: { color: chartPalette.axis, fontSize: 10 },
                  splitLine: { lineStyle: { color: chartPalette.grid } },
                },
                series: [
                  {
                    name: "Volatility",
                    type: "line",
                    data: risk?.rolling?.volatility.map((point) => point.value) ?? [],
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { width: 2.4, color: chartPalette.primary },
                    areaStyle: { color: chartPalette.areaPrimary },
                  },
                  {
                    name: "Sharpe",
                    type: "line",
                    data: risk?.rolling?.sharpe.map((point) => point.value) ?? [],
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { width: 2.1, color: chartPalette.teal },
                    areaStyle: { color: chartPalette.areaTeal },
                  },
                  {
                    name: "Beta",
                    type: "line",
                    data: risk?.rolling?.beta.map((point) => point.value) ?? [],
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { width: 2.1, color: chartPalette.amber },
                    areaStyle: { color: chartPalette.areaAmber },
                  },
                ],
              }}
            />
          </div>
        </ChartPanel>

        <div className="grid gap-5">
          <ChartPanel kicker="Allocation" title="Current weights" description="Understand weight structure before changing the book.">
            {allocationRows.length ? (
              <div className="rounded-[22px] border border-[var(--line)] bg-[rgba(6,10,20,0.52)] p-3">
                <Chart
                  style={{ height: 270 }}
                  option={{
                    animationDuration: 550,
                    tooltip: {
                      trigger: "item",
                      backgroundColor: chartPalette.tooltipBg,
                      borderColor: chartPalette.border,
                      borderWidth: 1,
                      textStyle: { color: chartPalette.tooltipText },
                    },
                    series: [
                      {
                        type: "pie",
                        radius: ["54%", "78%"],
                        center: ["50%", "50%"],
                        label: { color: chartPalette.axis, formatter: "{b}\n{d}%" },
                        itemStyle: { borderColor: "#0b1220", borderWidth: 3 },
                        data: allocationRows.map((row, index) => ({
                          name: row.ticker,
                          value: row.weight,
                          itemStyle: {
                            color: [chartPalette.primary, chartPalette.teal, chartPalette.amber, chartPalette.red, chartPalette.primarySoft][index % 5],
                          },
                        })),
                      },
                    ],
                  }}
                />
              </div>
            ) : (
              <EmptyState title="No allocation yet" body="Add holdings to start seeing weight structure." />
            )}
          </ChartPanel>

          <ChartPanel kicker="Interpretation" title="Current read" description="Summarize what matters without leaving the workspace.">
            <div className="grid gap-3">
              <div className="glass-strip rounded-[20px] p-4">
                <div className="eyebrow text-[var(--text-faint)]">Summary</div>
                <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">
                  {attribution?.summary ?? "Refresh analytics after saving holdings to generate a summary."}
                </p>
              </div>
              {risk?.tickers_dropped.length ? (
                <div className="rounded-[20px] border border-[rgba(255,183,111,0.18)] bg-[rgba(255,183,111,0.1)] p-4">
                  <div className="flex items-center gap-2 text-[var(--accent-3)]">
                    <TriangleAlert size={16} />
                    <span className="text-sm font-semibold">Dropped tickers</span>
                  </div>
                  <p className="mt-2 text-sm text-[var(--text-soft)]">{risk.tickers_dropped.join(", ")}</p>
                </div>
              ) : null}
            </div>
          </ChartPanel>
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-[0.88fr_1.12fr]">
        <ControlPanel title="Analysis parameters" description="Adjust benchmark assumptions and refresh analytics without leaving the workspace.">
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Period</span>
              <Select value={config.period} onChange={(event) => setConfig((current) => ({ ...current, period: event.target.value }))}>
                {periods.map((period) => (
                  <option key={period.value} value={period.value}>{period.label}</option>
                ))}
              </Select>
            </label>
            <label className="space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Interval</span>
              <Select value={config.interval} onChange={(event) => setConfig((current) => ({ ...current, interval: event.target.value }))}>
                {intervals.map((interval) => (
                  <option key={interval.value} value={interval.value}>{interval.label}</option>
                ))}
              </Select>
            </label>
            <label className="space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Benchmark</span>
              <Select value={config.benchmark} onChange={(event) => setConfig((current) => ({ ...current, benchmark: event.target.value }))}>
                {benchmarks.map((benchmark) => (
                  <option key={benchmark} value={benchmark}>{benchmark}</option>
                ))}
              </Select>
            </label>
            <label className="space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Return Type</span>
              <Select value={config.return_type} onChange={(event) => setConfig((current) => ({ ...current, return_type: event.target.value }))}>
                {returnTypes.map((returnType) => (
                  <option key={returnType.value} value={returnType.value}>{returnType.label}</option>
                ))}
              </Select>
            </label>
            <label className="space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Risk-Free Rate</span>
              <Input value={String(config.risk_free)} onChange={(event) => setConfig((current) => ({ ...current, risk_free: Number(event.target.value) || 0 }))} />
            </label>
            <label className="space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Rolling Window</span>
              <Input value={String(config.rolling_window)} onChange={(event) => setConfig((current) => ({ ...current, rolling_window: Number(event.target.value) || 30 }))} />
            </label>
          </div>
          <div className="mt-5">
            <Button
              loading={savingAnalytics}
              onClick={async () => {
                setSavingAnalytics(true);
                await loadAll(config);
                setSavingAnalytics(false);
              }}
            >
              Refresh Analytics
            </Button>
          </div>
        </ControlPanel>

        {portfolio ? (
          <HoldingsEditor
            portfolio={portfolio}
            onUpdated={(nextPortfolio) => {
              setPortfolio(nextPortfolio);
              void loadAll(config);
            }}
          />
        ) : null}
      </section>

      <section className="grid gap-5 xl:grid-cols-[1.08fr_0.92fr]">
        <ChartPanel kicker="Holdings" title="Current positions" description="A tighter table treatment for the active book.">
          {portfolio?.holdings.length ? (
            <DataTable
              rows={portfolio.holdings}
              rowKey={(row) => String(row.id)}
              columns={[
                { key: "ticker", header: "Ticker", render: (row) => <span className="mono font-semibold text-[var(--text)]">{row.ticker}</span> },
                { key: "weight", header: "Weight", align: "right", render: (row) => <span className="mono text-[var(--text)]">{fmtPct(row.weight)}</span> },
                {
                  key: "status",
                  header: "Status",
                  align: "right",
                  render: () => <span className="rounded-full bg-[rgba(88,199,152,0.12)] px-2.5 py-1 text-[11px] uppercase tracking-[0.12em] text-[var(--success)]">Active</span>,
                },
              ]}
            />
          ) : (
            <EmptyState title="No holdings yet" body="Use the holdings editor to start constructing the portfolio." />
          )}
        </ChartPanel>

        <ChartPanel kicker="Sector Notes" title="Risk concentration" description="Where attribution is currently clustering.">
          <div className="grid gap-3">
            {(attribution?.sector_attribution ?? []).slice(0, 4).map((item) => (
              <div key={item.sector} className="glass-strip rounded-[18px] p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-[var(--text)]">{item.sector}</span>
                  <span className="mono text-sm text-[var(--text-soft)]">{fmtPct(item.trc_pct)}</span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-[rgba(255,255,255,0.06)]">
                  <div className="h-full rounded-full bg-[var(--accent)]" style={{ width: `${Math.min(item.trc_pct * 100, 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </ChartPanel>
      </section>
    </div>
  );
}
