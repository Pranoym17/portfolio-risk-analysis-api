import type { AttributionResponse } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { fmtPct, fmtRatio } from "@/lib/utils";

export function AttributionPanel({ attribution }: { attribution: AttributionResponse }) {
  return (
    <Card className="rounded-[20px]">
      <CardHeader className="block">
        <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Risk Attribution</div>
        <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Asset-level contribution to total volatility</CardTitle>
        <CardDescription>{attribution.summary}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-[18px] border border-[var(--border)]">
          <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr_0.8fr_0.8fr] gap-4 border-b border-[var(--border)] bg-[var(--bg-muted)] px-4 py-3 text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--text-faint)]">
            <div>Asset</div>
            <div className="text-right">Weight</div>
            <div className="text-right">MRC</div>
            <div className="text-right">TRC</div>
            <div className="text-right">% Total</div>
          </div>
          <div className="scrollbar-thin max-h-[420px] overflow-auto">
            {attribution.attribution.map((item) => (
              <div
                key={item.ticker}
                className="grid grid-cols-[1.2fr_0.8fr_0.8fr_0.8fr_0.8fr] gap-4 border-b border-[var(--border)] px-4 py-4 text-sm last:border-b-0"
              >
                <div className="space-y-2">
                  <div className="font-semibold text-[var(--text)]">{item.ticker}</div>
                  <Badge tone="accent">{item.sector}</Badge>
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
