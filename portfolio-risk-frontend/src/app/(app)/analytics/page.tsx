"use client";

import { useEffect, useMemo, useState } from "react";
import { Chart } from "@/components/analytics/Chart";
import { ChartPanel } from "@/components/dashboard/ChartPanel";
import { listPortfolios, getAttribution, getRisk } from "@/lib/api";
import { chartPalette } from "@/lib/design-system";
import type { AttributionResponse, PortfolioOut, RiskResponse } from "@/lib/types";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/StatePanel";
import { fmtPct, fmtRatio, getErrorMessage } from "@/lib/utils";

export default function AnalyticsPage() {
  const [portfolio, setPortfolio] = useState<PortfolioOut | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const portfolios = await listPortfolios();
        if (!portfolios[0]) {
          setPortfolio(null);
          setRisk(null);
          setAttribution(null);
          return;
        }
        setPortfolio(portfolios[0]);
        const [nextRisk, nextAttribution] = await Promise.all([
          getRisk(portfolios[0].id, {
            period: "1y",
            interval: "1d",
            benchmark: "SPY",
            return_type: "simple",
            risk_free: 0.02,
            var_level: 0.95,
            trading_days: 252,
            rolling_window: 30,
          }),
          getAttribution(portfolios[0].id, {
            period: "1y",
            interval: "1d",
            trading_days: 252,
          }),
        ]);
        setRisk(nextRisk);
        setAttribution(nextAttribution);
      } catch (nextError) {
        setError(getErrorMessage(nextError, "Unable to load the analytics workspace."));
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  const covariance = useMemo(() => risk?.metrics.covariance_matrix ?? {}, [risk?.metrics.covariance_matrix]);
  const tickers = Object.keys(covariance);
  const heatmapValues = useMemo(
    () =>
      tickers.flatMap((rowTicker, rowIndex) =>
        tickers.map((colTicker, colIndex) => [colIndex, rowIndex, covariance[rowTicker]?.[colTicker] ?? 0]),
      ),
    [covariance, tickers],
  );

  if (loading) return <LoadingState title="Loading analytics..." />;

  return (
    <div className="space-y-5">
      {error ? <ErrorState title="Analytics unavailable" body={error} /> : null}

      {!portfolio || !risk ? (
        <EmptyState title="No analytics yet" body="Create a portfolio with holdings to unlock the full risk analytics workspace." />
      ) : (
        <>
          <section className="panel hero-panel rounded-[30px] p-6">
            <div className="eyebrow text-[var(--accent)]">Risk Analytics Workspace</div>
            <h2 className="mt-3 text-4xl font-semibold tracking-[-0.06em]">{portfolio.name}</h2>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-[var(--text-soft)]">
              Dive deeper into rolling metrics, contribution structure, and covariance behavior without leaving the portfolio analytics environment.
            </p>
          </section>

          <section className="grid gap-5 xl:grid-cols-[1.18fr_0.82fr]">
            <ChartPanel kicker="Rolling View" title="Volatility, Sharpe, and beta" description="One advanced chart surface instead of separate repetitive blocks.">
              <div className="rounded-[22px] border border-[var(--line)] bg-[rgba(6,10,20,0.52)] p-3">
                <Chart
                  style={{ height: 400 }}
                  option={{
                    animationDuration: 700,
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
                    grid: { top: 30, left: 48, right: 24, bottom: 38 },
                    xAxis: {
                      type: "category",
                      data: risk.rolling?.volatility.map((point) => point.date) ?? [],
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
                        smooth: true,
                        showSymbol: false,
                        lineStyle: { color: chartPalette.primary, width: 2.5 },
                        areaStyle: { color: chartPalette.areaPrimary },
                        data: risk.rolling?.volatility.map((point) => point.value) ?? [],
                      },
                      {
                        name: "Sharpe",
                        type: "line",
                        smooth: true,
                        showSymbol: false,
                        lineStyle: { color: chartPalette.teal, width: 2.2 },
                        areaStyle: { color: chartPalette.areaTeal },
                        data: risk.rolling?.sharpe.map((point) => point.value) ?? [],
                      },
                      {
                        name: "Beta",
                        type: "line",
                        smooth: true,
                        showSymbol: false,
                        lineStyle: { color: chartPalette.amber, width: 2.2 },
                        areaStyle: { color: chartPalette.areaAmber },
                        data: risk.rolling?.beta.map((point) => point.value) ?? [],
                      },
                    ],
                  }}
                />
              </div>
            </ChartPanel>

            <ChartPanel kicker="Metrics" title="Current quantitative read" description="High-value numbers framed like a real analytics surface.">
              <div className="grid gap-3">
                {[
                  ["Sharpe Ratio", fmtRatio(risk.metrics.sharpe_ratio)],
                  ["Sortino Ratio", fmtRatio(risk.metrics.sortino_ratio)],
                  ["Max Drawdown", fmtPct(risk.metrics.max_drawdown)],
                  ["Worst Day", fmtPct(risk.metrics.worst_day)],
                ].map(([label, value]) => (
                  <div key={label} className="glass-strip rounded-[18px] p-4">
                    <div className="eyebrow text-[var(--text-faint)]">{label}</div>
                    <div className="metric-value mt-2 text-3xl font-semibold tracking-[-0.05em]">{value}</div>
                  </div>
                ))}
              </div>
            </ChartPanel>
          </section>

          <section className="grid gap-5 xl:grid-cols-[0.92fr_1.08fr]">
            <ChartPanel kicker="Covariance Matrix" title="Holdings covariance heatmap" description="A cleaner matrix treatment for portfolio interaction structure.">
              <div className="rounded-[22px] border border-[var(--line)] bg-[rgba(6,10,20,0.52)] p-3">
                <Chart
                  style={{ height: 380 }}
                  option={{
                    animationDuration: 550,
                    tooltip: {
                      position: "top",
                      backgroundColor: chartPalette.tooltipBg,
                      borderColor: chartPalette.border,
                      borderWidth: 1,
                      textStyle: { color: chartPalette.tooltipText },
                    },
                    grid: { top: 50, left: 70, right: 12, bottom: 30 },
                    xAxis: {
                      type: "category",
                      data: tickers,
                      splitArea: { show: true },
                      axisLabel: { color: chartPalette.axis, fontSize: 10 },
                    },
                    yAxis: {
                      type: "category",
                      data: tickers,
                      splitArea: { show: true },
                      axisLabel: { color: chartPalette.axis, fontSize: 10 },
                    },
                    visualMap: {
                      min: Math.min(...heatmapValues.map((item) => Number(item[2])), 0),
                      max: Math.max(...heatmapValues.map((item) => Number(item[2])), 1),
                      calculable: false,
                      orient: "horizontal",
                      left: "center",
                      bottom: 0,
                      textStyle: { color: chartPalette.axis },
                      inRange: {
                        color: ["#101826", "#243551", "#7ba2ff"],
                      },
                    },
                    series: [
                      {
                        type: "heatmap",
                        data: heatmapValues,
                        label: { show: false },
                        emphasis: { itemStyle: { borderColor: "#fff", borderWidth: 1 } },
                      },
                    ],
                  }}
                />
              </div>
            </ChartPanel>

            <ChartPanel kicker="Attribution" title="Top risk contributors" description="Contribution breakdown with better framing and density.">
              <div className="grid gap-3">
                {(attribution?.attribution ?? []).slice(0, 6).map((item) => (
                  <div key={item.ticker} className="glass-strip rounded-[18px] p-4">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="mono text-sm font-semibold text-[var(--text)]">{item.ticker}</p>
                        <p className="mt-1 text-sm text-[var(--text-soft)]">{item.sector}</p>
                      </div>
                      <div className="text-right">
                        <p className="mono text-sm text-[var(--text)]">{fmtPct(item.trc_pct)}</p>
                        <p className="mt-1 text-xs text-[var(--text-faint)]">TRC%</p>
                      </div>
                    </div>
                    <div className="mt-3 h-2 rounded-full bg-[rgba(255,255,255,0.06)]">
                      <div className="h-full rounded-full bg-[var(--accent)]" style={{ width: `${Math.min(item.trc_pct * 100, 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </ChartPanel>
          </section>
        </>
      )}
    </div>
  );
}
