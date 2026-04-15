"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { AuthGate } from "@/components/auth/AuthGate";
import { getAttribution, getRisk, listPortfolios } from "@/lib/api";
import type { AttributionResponse, PortfolioOut, RiskResponse } from "@/lib/types";
import { fmtPct, fmtRatio, getErrorMessage } from "@/lib/utils";
import { CreatePortfolio } from "@/components/portfolio/CreatePortfolio";
import { PortfolioList } from "@/components/portfolio/PortfolioList";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState, ErrorState, SkeletonBlock } from "@/components/ui/StatePanel";

export default function PortfoliosPage() {
  const [items, setItems] = useState<PortfolioOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRisk, setSelectedRisk] = useState<RiskResponse | null>(null);
  const [selectedAttribution, setSelectedAttribution] = useState<AttributionResponse | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);

    try {
      const data = await listPortfolios();
      setItems(data);

      if (data[0]) {
        const [risk, attribution] = await Promise.all([
          getRisk(data[0].id, {
            period: "1y",
            interval: "1d",
            benchmark: "SPY",
            return_type: "simple",
            risk_free: 0.02,
            var_level: 0.95,
            trading_days: 252,
            rolling_window: 30,
          }),
          getAttribution(data[0].id, {
            period: "1y",
            interval: "1d",
            trading_days: 252,
          }),
        ]);
        setSelectedRisk(risk);
        setSelectedAttribution(attribution);
      } else {
        setSelectedRisk(null);
        setSelectedAttribution(null);
      }
    } catch (nextError: unknown) {
      setError(getErrorMessage(nextError, "Unable to load portfolios"));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    async function initialLoad() {
      await refresh();
    }

    void initialLoad();
  }, []);

  const totalHoldings = items.reduce((count, portfolio) => count + portfolio.holdings.length, 0);

  return (
    <AuthGate>
      <div className="space-y-5">
        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.1fr_0.9fr]">
          <Card className="rounded-[20px]">
            <CardHeader className="block">
              <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Overview</div>
              <CardTitle className="mt-2 text-2xl tracking-[-0.03em]">A calm command center for portfolio risk</CardTitle>
              <CardDescription>
                Create and inspect portfolios, then move from holdings construction into benchmark-aware risk and attribution analysis.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-3">
              {[
                ["Portfolios", String(items.length)],
                ["Holdings staged", String(totalHoldings)],
                ["Selected benchmark", selectedRisk?.metrics.benchmark_ticker ?? "SPY"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                  <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">{label}</div>
                  <div className="metric-value mt-3 text-3xl font-semibold tracking-[-0.04em] text-[var(--text)]">{value}</div>
                </div>
              ))}
            </CardContent>
          </Card>

          <CreatePortfolio onCreated={refresh} />
        </section>

        {error ? <ErrorState title="Portfolio overview unavailable" body={error} onRetry={refresh} /> : null}

        <section className="grid grid-cols-1 gap-4 2xl:grid-cols-[0.95fr_1.05fr]">
          <PortfolioList items={items} loading={loading} onChanged={refresh} />

          <Card className="rounded-[20px]">
            <CardHeader className="block">
              <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Selected Snapshot</div>
              <CardTitle className="mt-2 text-xl tracking-[-0.03em]">
                {items[0] ? `${items[0].name} at a glance` : "Portfolio snapshot"}
              </CardTitle>
              <CardDescription>
                A quick preview of the first available portfolio so the main overview stays information-rich.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {loading ? (
                <div className="space-y-3">
                  <SkeletonBlock className="h-[88px]" />
                  <SkeletonBlock className="h-[88px]" />
                  <SkeletonBlock className="h-[120px]" />
                </div>
              ) : !items[0] ? (
                <EmptyState
                  title="No portfolio selected"
                  body="Create a portfolio to populate this overview and unlock the detailed workspace."
                />
              ) : !selectedRisk || !selectedAttribution ? (
                <ErrorState
                  title="Analytics preview unavailable"
                  body="The portfolio exists, but analytics could not be loaded yet. Add holdings or revisit the backend response."
                  onRetry={refresh}
                />
              ) : (
                <>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                      <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Volatility</div>
                      <div className="metric-value mt-3 text-3xl font-semibold tracking-[-0.04em]">{fmtPct(selectedRisk.metrics.volatility)}</div>
                    </div>
                    <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                      <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Sharpe</div>
                      <div className="metric-value mt-3 text-3xl font-semibold tracking-[-0.04em]">{fmtRatio(selectedRisk.metrics.sharpe_ratio)}</div>
                    </div>
                  </div>

                  <div className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <div className="text-sm font-semibold text-[var(--text)]">Top risk sector</div>
                        <div className="mt-1 text-sm text-[var(--text-soft)]">{selectedAttribution.sector_attribution[0]?.sector ?? "Unknown"}</div>
                      </div>
                      {selectedAttribution.sector_attribution[0] ? (
                        <Badge tone="accent">{fmtPct(selectedAttribution.sector_attribution[0].trc_pct)}</Badge>
                      ) : null}
                    </div>
                    <p className="mt-3 text-sm leading-6 text-[var(--text-soft)]">{selectedAttribution.summary}</p>
                  </div>

                  <Link
                    href={`/portfolios/${items[0].id}`}
                    className="focus-ring inline-flex items-center gap-2 rounded-[12px] border border-[var(--border)] bg-[var(--bg-elevated)] px-3.5 py-2.5 text-sm font-medium text-[var(--text)] transition hover:bg-[var(--bg-muted)]"
                  >
                    Open detailed workspace
                    <ArrowRight size={16} />
                  </Link>
                </>
              )}
            </CardContent>
          </Card>
        </section>
      </div>
    </AuthGate>
  );
}
