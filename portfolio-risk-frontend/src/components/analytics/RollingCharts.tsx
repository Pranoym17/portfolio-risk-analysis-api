import type { RiskResponse } from "@/lib/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { chartPalette } from "@/lib/design-system";
import { Chart } from "./Chart";

export function RollingCharts({ risk }: { risk: RiskResponse }) {
  const volatility = risk.rolling?.volatility ?? [];
  const sharpe = risk.rolling?.sharpe ?? [];
  const beta = risk.rolling?.beta ?? [];
  const dates = volatility.map((point) => point.date);

  return (
    <Card className="chart-panel rounded-[24px]">
      <CardHeader className="block">
        <div className="section-kicker text-[var(--accent)]">Rolling Analytics</div>
        <CardTitle className="mt-2 text-2xl tracking-[-0.04em]">Portfolio behavior through time</CardTitle>
        <CardDescription>
          Layer volatility, Sharpe, and beta in a more terminal-style frame so changes in risk regime are easier to spot.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-[22px] border border-[var(--border)] bg-[linear-gradient(180deg,rgba(248,251,253,0.98),rgba(243,247,250,0.98))] p-3">
          <Chart
            style={{ height: 370 }}
            option={{
              animationDuration: 600,
              grid: { top: 28, left: 58, right: 20, bottom: 54 },
              legend: {
                bottom: 4,
                icon: "roundRect",
                itemWidth: 18,
                itemHeight: 8,
                textStyle: { color: chartPalette.axis, fontSize: 11, fontWeight: 600 },
              },
              tooltip: {
                trigger: "axis",
                backgroundColor: chartPalette.tooltipBg,
                borderColor: "rgba(255,255,255,0.08)",
                borderWidth: 1,
                padding: 12,
                textStyle: { color: chartPalette.tooltipText },
                axisPointer: {
                  type: "line",
                  lineStyle: { color: chartPalette.primarySoft, width: 1, opacity: 0.5 },
                },
              },
              xAxis: {
                type: "category",
                data: dates,
                boundaryGap: false,
                axisLabel: { color: chartPalette.axis, margin: 14 },
                axisLine: { lineStyle: { color: chartPalette.border } },
                axisTick: { show: false },
              },
              yAxis: {
                type: "value",
                axisLabel: { color: chartPalette.axis },
                splitLine: { lineStyle: { color: chartPalette.grid, type: "dashed" } },
              },
              series: [
                {
                  name: "Volatility",
                  type: "line",
                  data: volatility.map((point) => point.value),
                  smooth: true,
                  showSymbol: false,
                  lineStyle: { width: 2.4, color: chartPalette.primary },
                  areaStyle: { color: chartPalette.areaPrimary },
                },
                {
                  name: "Sharpe",
                  type: "line",
                  data: sharpe.map((point) => point.value),
                  smooth: true,
                  showSymbol: false,
                  lineStyle: { width: 2.2, color: chartPalette.teal },
                  areaStyle: { color: chartPalette.areaTeal },
                },
                {
                  name: "Beta",
                  type: "line",
                  data: beta.map((point) => point.value),
                  smooth: true,
                  showSymbol: false,
                  lineStyle: { width: 2.2, color: chartPalette.amber },
                  areaStyle: { color: chartPalette.areaAmber },
                },
              ],
            }}
          />
        </div>

        <div className="grid gap-3">
          <div className="surface-dark rounded-[22px] p-5">
            <div className="section-kicker text-slate-400">Reading Guide</div>
            <p className="mt-3 text-sm leading-7 text-slate-300">
              Volatility tracks realized portfolio turbulence, Sharpe captures quality of return, and beta highlights benchmark sensitivity.
            </p>
          </div>
          <div className="rounded-[22px] border border-[var(--border)] bg-[var(--bg-muted)] p-5">
            <div className="section-kicker text-[var(--accent)]">Window</div>
            <div className="metric-value mt-3 text-4xl font-semibold tracking-[-0.05em] text-[var(--text)]">
              {risk.rolling?.window ?? 0}d
            </div>
            <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">
              Shorter windows react faster to market shifts. Longer windows smooth noise but can mask changes in regime.
            </p>
          </div>
          <div className="rounded-[22px] border border-[var(--border)] bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(245,248,250,0.98))] p-5">
            <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
              {[
                ["Volatility points", volatility.length],
                ["Sharpe points", sharpe.length],
                ["Beta points", beta.length],
              ].map(([label, value]) => (
                <div key={label} className="data-cell rounded-[16px] p-4">
                  <div className="section-kicker text-[var(--text-faint)]">{label}</div>
                  <div className="metric-value mt-2 text-2xl font-semibold tracking-[-0.04em] text-[var(--text)]">{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
