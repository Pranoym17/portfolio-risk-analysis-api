import type { AttributionResponse } from "@/lib/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { chartPalette } from "@/lib/design-system";
import { fmtPct } from "@/lib/utils";
import { Chart } from "./Chart";

export function SectorChart({ attribution }: { attribution: AttributionResponse }) {
  return (
    <Card className="rounded-[20px]">
      <CardHeader className="block">
        <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Sector Contribution</div>
        <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Where volatility is concentrated</CardTitle>
        <CardDescription>Sector-level component risk contribution helps surface concentration beyond simple weight allocation.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
        <Chart
          style={{ height: 280 }}
          option={{
            animation: false,
            tooltip: {
              trigger: "item",
              backgroundColor: chartPalette.tooltipBg,
              borderWidth: 0,
              textStyle: { color: chartPalette.tooltipText },
              formatter: (params: { name: string; value: number }) => `${params.name}<br/>${fmtPct(params.value)}`,
            },
            series: [
              {
                type: "pie",
                radius: ["52%", "76%"],
                center: ["50%", "50%"],
                label: {
                  color: chartPalette.axis,
                  formatter: "{b}\n{d}%",
                },
                labelLine: { length: 14, length2: 10 },
                itemStyle: { borderColor: "#fff", borderWidth: 2 },
                data: attribution.sector_attribution.map((item, index) => ({
                  name: item.sector,
                  value: item.trc_pct,
                  itemStyle: {
                    color: [chartPalette.primary, chartPalette.teal, chartPalette.amber, chartPalette.red, chartPalette.slate][index % 5],
                  },
                })),
              },
            ],
          }}
        />

        <div className="space-y-3">
          {attribution.sector_attribution.map((item) => (
            <div key={item.sector} className="rounded-[16px] border border-[var(--border)] bg-[var(--bg-muted)] px-4 py-3">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm font-semibold text-[var(--text)]">{item.sector}</div>
                  <div className="mt-1 text-sm text-[var(--text-soft)]">{item.tickers.join(", ")}</div>
                </div>
                <div className="metric-value text-right text-base font-semibold text-[var(--text)]">{fmtPct(item.trc_pct)}</div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
