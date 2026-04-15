import type { RiskResponse } from "@/lib/types";
import { fmtPct, fmtRatio, fmtSignedPct } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/Card";

const metricsConfig = [
  ["Annual Return", (risk: RiskResponse) => fmtSignedPct(risk.metrics.annual_return)],
  ["Volatility", (risk: RiskResponse) => fmtPct(risk.metrics.volatility)],
  ["Sharpe", (risk: RiskResponse) => fmtRatio(risk.metrics.sharpe_ratio)],
  ["Sortino", (risk: RiskResponse) => fmtRatio(risk.metrics.sortino_ratio)],
  ["Value at Risk", (risk: RiskResponse) => fmtSignedPct(risk.metrics.value_at_risk)],
  ["Max Drawdown", (risk: RiskResponse) => fmtPct(risk.metrics.max_drawdown)],
  ["Worst Day", (risk: RiskResponse) => fmtPct(risk.metrics.worst_day)],
  ["Beta vs Benchmark", (risk: RiskResponse) => fmtRatio(risk.metrics.beta_vs_benchmark)],
] as const;

export function RiskKpis({ risk }: { risk: RiskResponse }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {metricsConfig.map(([label, formatter]) => (
        <Card key={label} className="rounded-[18px]">
          <CardContent className="space-y-2 py-5">
            <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">{label}</div>
            <div className="metric-value text-[28px] font-semibold tracking-[-0.04em] text-[var(--text)]">{formatter(risk)}</div>
            <div className="text-sm text-[var(--text-soft)]">Benchmark: {risk.metrics.benchmark_ticker}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
