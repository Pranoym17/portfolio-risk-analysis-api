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
    <Card className="rounded-[20px]">
      <CardHeader className="block">
        <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Rolling Analytics</div>
        <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Trend the evolving risk profile</CardTitle>
        <CardDescription>
          Rolling volatility, Sharpe, and beta help show whether the portfolio is stabilizing or becoming more sensitive over time.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Chart
          style={{ height: 360 }}
          option={{
            animation: false,
            grid: { top: 24, left: 56, right: 24, bottom: 50 },
            legend: {
              bottom: 4,
              textStyle: { color: chartPalette.axis },
            },
            tooltip: {
              trigger: "axis",
              backgroundColor: chartPalette.tooltipBg,
              borderWidth: 0,
              textStyle: { color: chartPalette.tooltipText },
            },
            xAxis: {
              type: "category",
              data: dates,
              axisLabel: { color: chartPalette.axis },
              axisLine: { lineStyle: { color: chartPalette.border } },
            },
            yAxis: {
              type: "value",
              axisLabel: { color: chartPalette.axis },
              splitLine: { lineStyle: { color: chartPalette.grid } },
            },
            series: [
              {
                name: "Volatility",
                type: "line",
                data: volatility.map((point) => point.value),
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 2, color: chartPalette.primary },
              },
              {
                name: "Sharpe",
                type: "line",
                data: sharpe.map((point) => point.value),
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 2, color: chartPalette.teal },
              },
              {
                name: "Beta",
                type: "line",
                data: beta.map((point) => point.value),
                smooth: true,
                showSymbol: false,
                lineStyle: { width: 2, color: chartPalette.amber },
              },
            ],
          }}
        />
      </CardContent>
    </Card>
  );
}
