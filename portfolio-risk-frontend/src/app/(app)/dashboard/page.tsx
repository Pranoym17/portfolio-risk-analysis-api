"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowRight, Sparkles, TriangleAlert } from "lucide-react";
import { Chart } from "@/components/analytics/Chart";
import { ChartPanel } from "@/components/dashboard/ChartPanel";
import { DataTable } from "@/components/dashboard/DataTable";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { PortfolioCard } from "@/components/portfolio/PortfolioCard";
import { Button } from "@/components/ui/Button";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/StatePanel";
import { chartPalette } from "@/lib/design-system";
import { getAttribution, getRisk, listPortfolios } from "@/lib/api";
import type { AttributionResponse, PortfolioOut, RiskResponse } from "@/lib/types";
import { fmtPct, fmtRatio, getErrorMessage } from "@/lib/utils";

const defaultRiskConfig = {
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

export default function DashboardPage() {
  const [portfolios, setPortfolios] = useState<PortfolioOut[]>([]);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const list = await listPortfolios();
        setPortfolios(list);
        if (!list[0]) {
          setRisk(null);
          setAttribution(null);
          return;
        }
        const [nextRisk, nextAttribution] = await Promise.all([
          getRisk(list[0].id, defaultRiskConfig),
          getAttribution(list[0].id, {
            period: "1y",
            interval: "1d",
            trading_days: 252,
          }),
        ]);
        setRisk(nextRisk);
        setAttribution(nextAttribution);
      } catch (nextError) {
        setError(getErrorMessage(nextError, "Unable to load the dashboard workspace."));
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  const featured = portfolios[0];
  const metrics = risk?.metrics;
  const volatilitySeries = sample(risk?.rolling?.volatility ?? [], metrics?.volatility ?? 0.15);
  const sharpeSeries = sample(risk?.rolling?.sharpe ?? [], metrics?.sharpe_ratio ?? 1);
  const betaSeries = sample(risk?.rolling?.beta ?? [], metrics?.beta_vs_benchmark ?? 1);

  const holdingsPreview = useMemo(() => featured?.holdings.slice(0, 6) ?? [], [featured]);

  if (loading) {
    return <LoadingState title="Loading dashboard..." />;
  }

  return (
    <div className="space-y-5">
      {error ? <ErrorState title="Dashboard unavailable" body={error} /> : null}

      <section className="panel hero-panel rounded-[30px] p-5 sm:p-6">
        <div className="grid gap-5 xl:grid-cols-[0.92fr_1.08fr]">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="glass-strip rounded-full px-3 py-1.5 text-xs text-[var(--text-soft)]">Portfolio intelligence</span>
              <span className="glass-strip rounded-full px-3 py-1.5 text-xs text-[var(--text-soft)]">Dark analytics workspace</span>
            </div>
            <h2 className="mt-4 max-w-2xl text-4xl font-semibold tracking-[-0.06em] text-[var(--text)]">
              {featured ? featured.name : "Create your first portfolio to activate the dashboard"}
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
              The command center keeps live risk, benchmark context, holdings readiness, and portfolio diagnostics visible without collapsing into repetitive dashboard cards.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/portfolios/create">
                <Button>
                  <Sparkles size={15} />
                  New Portfolio
                </Button>
              </Link>
              {featured ? (
                <Link href={`/portfolios/${featured.id}`}>
                  <Button variant="secondary">
                    Open Workspace
                    <ArrowRight size={15} />
                  </Button>
                </Link>
              ) : null}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <MetricCard label="Sharpe" value={fmtRatio(metrics?.sharpe_ratio)} detail="risk-adjusted return" points={sharpeSeries} tone="accent" />
            <MetricCard label="Annual Return" value={fmtPct(metrics?.annual_return ?? Number.NaN)} detail="annualized" points={sharpeSeries} tone="good" />
            <MetricCard label="Volatility" value={fmtPct(metrics?.volatility ?? Number.NaN)} detail="annualized" points={volatilitySeries} tone="warn" />
            <MetricCard label="VaR" value={fmtPct(metrics?.value_at_risk ?? Number.NaN)} detail="95% confidence" points={betaSeries} tone="bad" />
          </div>
        </div>
      </section>

      {!featured ? (
        <EmptyState
          title="No portfolio data yet"
          body="Create a portfolio first, then add holdings so charts, table previews, and risk diagnostics can populate this dashboard."
          actionLabel="Create Portfolio"
          onAction={() => {
            window.location.href = "/portfolios/create";
          }}
        />
      ) : (
        <>
          <section className="grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
            <ChartPanel
              kicker="Performance Lens"
              title="Rolling performance and benchmark context"
              description="A wider focal chart keeps the dashboard anchored around the portfolio story instead of flattening into equal-sized panels."
            >
              <div className="rounded-[22px] border border-[var(--line)] bg-[rgba(6,10,20,0.5)] p-3">
                <Chart
                  style={{ height: 360 }}
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
                    grid: { top: 30, left: 48, right: 22, bottom: 38 },
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
                        lineStyle: { width: 2.5, color: chartPalette.primary },
                        areaStyle: { color: chartPalette.areaPrimary },
                      },
                      {
                        name: "Sharpe",
                        type: "line",
                        data: risk?.rolling?.sharpe.map((point) => point.value) ?? [],
                        smooth: true,
                        showSymbol: false,
                        lineStyle: { width: 2.2, color: chartPalette.teal },
                        areaStyle: { color: chartPalette.areaTeal },
                      },
                    ],
                  }}
                />
              </div>
            </ChartPanel>

            <div className="grid gap-5">
              <ChartPanel kicker="Portfolio" title="Featured book" description="Jump directly into the active analytics workspace.">
                <PortfolioCard portfolio={featured} featured />
              </ChartPanel>

              <ChartPanel kicker="Signals" title="Analyst notes" description="Highlight the most important diagnostic surfaces instead of adding more generic stat boxes.">
                <div className="grid gap-3">
                  <div className="rounded-[20px] border border-[rgba(255,183,111,0.18)] bg-[rgba(255,183,111,0.08)] p-4">
                    <div className="flex items-center gap-2 text-[var(--accent-3)]">
                      <TriangleAlert size={16} />
                      <span className="text-sm font-semibold">Allocation concentration</span>
                    </div>
                    <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
                      {attribution?.sector_attribution[0]
                        ? `${attribution.sector_attribution[0].sector} is currently the largest risk contributor.`
                        : "Run analytics on a portfolio with holdings to surface sector attribution notes."}
                    </p>
                  </div>
                  <div className="glass-strip rounded-[20px] p-4">
                    <div className="eyebrow text-[var(--text-faint)]">Benchmark</div>
                    <div className="mt-2 text-2xl font-semibold tracking-[-0.05em]">{metrics?.benchmark_ticker ?? "SPY"}</div>
                    <p className="mt-2 text-sm text-[var(--text-soft)]">Current beta: {fmtRatio(metrics?.beta_vs_benchmark)}</p>
                  </div>
                </div>
              </ChartPanel>
            </div>
          </section>

          <section className="grid gap-5 xl:grid-cols-[0.78fr_1.22fr]">
            <ChartPanel kicker="Allocation" title="Weight structure" description="A quick allocation read before drilling further into the holdings table.">
              <div className="rounded-[22px] border border-[var(--line)] bg-[rgba(6,10,20,0.5)] p-3">
                <Chart
                  style={{ height: 300 }}
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
                        radius: ["52%", "78%"],
                        center: ["50%", "52%"],
                        label: { color: chartPalette.axis, formatter: "{b}\n{d}%" },
                        itemStyle: { borderColor: "#0d1726", borderWidth: 3 },
                        data: (risk?.weights_used ? Object.entries(risk.weights_used) : []).map(([ticker, weight], index) => ({
                          name: ticker,
                          value: weight,
                          itemStyle: {
                            color: [chartPalette.primary, chartPalette.teal, chartPalette.amber, chartPalette.red, chartPalette.primarySoft][index % 5],
                          },
                        })),
                      },
                    ],
                  }}
                />
              </div>
            </ChartPanel>

            <ChartPanel kicker="Holdings Preview" title="Selected book holdings" description="A tighter, more readable table treatment for scanning the active book.">
              <DataTable
                rows={holdingsPreview}
                rowKey={(row) => String(row.id)}
                columns={[
                  { key: "ticker", header: "Ticker", render: (row) => <span className="mono font-semibold text-[var(--text)]">{row.ticker}</span> },
                  { key: "weight", header: "Weight", align: "right", render: (row) => <span className="mono text-[var(--text)]">{fmtPct(row.weight)}</span> },
                  {
                    key: "status",
                    header: "Status",
                    align: "right",
                    render: () => <span className="rounded-full bg-[rgba(88,199,152,0.12)] px-2.5 py-1 text-[11px] uppercase tracking-[0.12em] text-[var(--success)]">Ready</span>,
                  },
                ]}
              />
            </ChartPanel>
          </section>
        </>
      )}
    </div>
  );
}
