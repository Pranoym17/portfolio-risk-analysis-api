import type { AttributionResponse } from "@/lib/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { chartPalette } from "@/lib/design-system";
import { fmtPct } from "@/lib/utils";
import { Chart } from "./Chart";

export function SectorChart({ attribution }: { attribution: AttributionResponse }) {
  const top = attribution.sector_attribution[0];

  return (
    <Card className="chart-panel rounded-[24px]">
      <CardHeader className="block">
        <div className="section-kicker text-[var(--accent)]">Sector Contribution</div>
        <CardTitle className="mt-2 text-2xl tracking-[-0.04em]">Concentration by sector</CardTitle>
        <CardDescription>Volatility attribution should show where risk lives, not just how the pie looks on paper.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5 xl:grid-cols-[0.94fr_1.06fr]">
        <div className="space-y-4">
          <div className="surface-dark rounded-[22px] p-5">
            <div className="section-kicker text-slate-400">Primary driver</div>
            <div className="mt-3 text-3xl font-semibold tracking-[-0.05em] text-white">{top?.sector ?? "Unknown"}</div>
            <div className="mt-2 text-sm text-slate-300">{top ? fmtPct(top.trc_pct) : "--"} of total volatility contribution</div>
          </div>
          <div className="rounded-[22px] border border-[var(--border)] bg-[linear-gradient(180deg,rgba(248,251,253,0.98),rgba(244,248,250,0.98))] p-3">
            <Chart
              style={{ height: 300 }}
              option={{
                animationDuration: 600,
                tooltip: {
                  trigger: "item",
                  backgroundColor: chartPalette.tooltipBg,
                  borderColor: "rgba(255,255,255,0.08)",
                  borderWidth: 1,
                  padding: 12,
                  textStyle: { color: chartPalette.tooltipText },
                  formatter: (params: { name: string; value: number }) => `${params.name}<br/>${fmtPct(params.value)}`,
                },
                series: [
                  {
                    type: "pie",
                    radius: ["48%", "74%"],
                    center: ["50%", "48%"],
                    label: { show: false },
                    labelLine: { show: false },
                    itemStyle: { borderColor: "#fff", borderWidth: 2.5 },
                    data: attribution.sector_attribution.map((item, index) => ({
                      name: item.sector,
                      value: item.trc_pct,
                      itemStyle: {
                        color: [
                          chartPalette.primary,
                          chartPalette.teal,
                          chartPalette.amber,
                          chartPalette.red,
                          chartPalette.slate,
                        ][index % 5],
                      },
                    })),
                  },
                ],
                graphic: [
                  {
                    type: "text",
                    left: "center",
                    top: "42%",
                    style: {
                      text: "Top Sector",
                      fill: chartPalette.axis,
                      fontSize: 11,
                      fontWeight: 700,
                    },
                  },
                  {
                    type: "text",
                    left: "center",
                    top: "50%",
                    style: {
                      text: top?.sector ?? "--",
                      fill: "#0f1f33",
                      fontSize: 20,
                      fontWeight: 700,
                    },
                  },
                ],
              }}
            />
          </div>
        </div>

        <div className="space-y-3">
          {attribution.sector_attribution.map((item) => (
            <div key={item.sector} className="table-shell rounded-[18px] px-4 py-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-sm font-semibold text-[var(--text)]">{item.sector}</div>
                  <div className="mt-1 text-sm leading-6 text-[var(--text-soft)]">{item.tickers.join(", ")}</div>
                </div>
                <div className="metric-value text-right text-lg font-semibold tracking-[-0.03em] text-[var(--text)]">
                  {fmtPct(item.trc_pct)}
                </div>
              </div>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-[var(--bg-subtle)]">
                <div className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),var(--teal))]" style={{ width: `${Math.min(item.trc_pct * 100, 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
