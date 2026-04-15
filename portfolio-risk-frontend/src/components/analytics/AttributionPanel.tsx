import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import type { AttributionResponse } from "@/lib/types";
import { fmtPct, fmtRatio } from "@/lib/utils";

export function AttributionPanel({ attribution }: { attribution: AttributionResponse }) {
  return (
    <Card className="rounded-[24px]">
      <CardHeader className="block">
        <div className="section-kicker text-[var(--accent)]">Risk Attribution</div>
        <CardTitle className="mt-2 text-2xl tracking-[-0.04em]">Asset-level contribution to total volatility</CardTitle>
        <CardDescription>{attribution.summary}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-3">
          {attribution.attribution.slice(0, 3).map((item, index) => (
            <div key={item.ticker} className={index === 0 ? "surface-dark rounded-[20px] p-5" : "data-cell rounded-[20px] p-5"}>
              <div className={`section-kicker ${index === 0 ? "text-slate-400" : "text-[var(--text-faint)]"}`}>Top driver {index + 1}</div>
              <div className={`mt-3 text-2xl font-semibold tracking-[-0.04em] ${index === 0 ? "text-white" : "text-[var(--text)]"}`}>{item.ticker}</div>
              <div className={`mt-2 text-sm ${index === 0 ? "text-slate-300" : "text-[var(--text-soft)]"}`}>{fmtPct(item.trc_pct)} of portfolio volatility</div>
            </div>
          ))}
        </div>

        <div className="table-shell overflow-hidden rounded-[22px]">
          <div className="table-header grid grid-cols-[1.25fr_0.8fr_0.8fr_0.8fr_0.8fr] gap-4 border-b border-[var(--border)] px-5 py-4 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-faint)]">
            <div>Asset</div>
            <div className="text-right">Weight</div>
            <div className="text-right">MRC</div>
            <div className="text-right">TRC</div>
            <div className="text-right">% Total</div>
          </div>
          <div className="scrollbar-thin max-h-[460px] overflow-auto">
            {attribution.attribution.map((item) => (
              <div
                key={item.ticker}
                className="table-row grid grid-cols-[1.25fr_0.8fr_0.8fr_0.8fr_0.8fr] gap-4 border-b border-[var(--border)] px-5 py-4 text-sm last:border-b-0"
              >
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="font-semibold tracking-[-0.01em] text-[var(--text)]">{item.ticker}</div>
                    <Badge tone="accent">{item.sector}</Badge>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-[var(--bg-subtle)]">
                    <div className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),var(--teal))]" style={{ width: `${Math.min(item.trc_pct * 100, 100)}%` }} />
                  </div>
                </div>
                <div className="num text-right text-[var(--text-soft)]">{fmtPct(item.weight)}</div>
                <div className="num text-right text-[var(--text-soft)]">{fmtRatio(item.mrc, 4)}</div>
                <div className="num text-right text-[var(--text-soft)]">{fmtRatio(item.trc, 4)}</div>
                <div className="num text-right font-semibold text-[var(--text)]">{fmtPct(item.trc_pct)}</div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
