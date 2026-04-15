import { TrendingDown, TrendingUp } from "lucide-react";
import type { RiskResponse } from "@/lib/types";
import { fmtPct, fmtRatio, fmtSignedPct } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/Card";

type MetricTone = "good" | "bad" | "warn" | "accent" | "neutral";

const metricsConfig = [
  {
    label: "Annual Return",
    value: (risk: RiskResponse) => fmtSignedPct(risk.metrics.annual_return),
    tone: (risk: RiskResponse): MetricTone => (risk.metrics.annual_return >= 0 ? "good" : "bad"),
  },
  {
    label: "Volatility",
    value: (risk: RiskResponse) => fmtPct(risk.metrics.volatility),
    tone: (): MetricTone => "neutral",
  },
  {
    label: "Sharpe",
    value: (risk: RiskResponse) => fmtRatio(risk.metrics.sharpe_ratio),
    tone: (risk: RiskResponse): MetricTone => ((risk.metrics.sharpe_ratio ?? 0) >= 1 ? "good" : "warn"),
  },
  {
    label: "Sortino",
    value: (risk: RiskResponse) => fmtRatio(risk.metrics.sortino_ratio),
    tone: (risk: RiskResponse): MetricTone => ((risk.metrics.sortino_ratio ?? 0) >= 1 ? "good" : "warn"),
  },
  {
    label: "Value at Risk",
    value: (risk: RiskResponse) => fmtSignedPct(risk.metrics.value_at_risk),
    tone: (): MetricTone => "bad",
  },
  {
    label: "Max Drawdown",
    value: (risk: RiskResponse) => fmtPct(risk.metrics.max_drawdown),
    tone: (): MetricTone => "bad",
  },
  {
    label: "Worst Day",
    value: (risk: RiskResponse) => fmtPct(risk.metrics.worst_day),
    tone: (): MetricTone => "bad",
  },
  {
    label: "Beta vs Benchmark",
    value: (risk: RiskResponse) => fmtRatio(risk.metrics.beta_vs_benchmark),
    tone: (): MetricTone => "accent",
  },
] as const;

const toneClass = {
  good: "text-[var(--teal)] bg-[var(--teal-soft)]",
  bad: "text-[var(--red)] bg-[var(--red-soft)]",
  warn: "text-[var(--amber)] bg-[var(--amber-soft)]",
  accent: "text-[var(--accent)] bg-[var(--accent-soft)]",
  neutral: "text-[var(--text-soft)] bg-[var(--bg-muted)]",
};

export function RiskKpis({ risk }: { risk: RiskResponse }) {
  return (
    <div className="dashboard-grid xl:grid-cols-[1.2fr_0.8fr_0.8fr]">
      <Card className="surface-dark rounded-[24px] xl:col-span-1">
        <CardContent className="relative z-10 grid gap-4 p-6 sm:grid-cols-[1fr_0.85fr]">
          <div>
            <div className="section-kicker text-slate-400">Headline Risk</div>
            <div className="mt-3 text-4xl font-semibold tracking-[-0.05em] text-white">{fmtPct(risk.metrics.volatility)}</div>
            <p className="mt-3 max-w-sm text-sm leading-7 text-slate-300">
              Current annualized portfolio volatility against {risk.metrics.benchmark_ticker}, with drawdown and downside metrics shown alongside.
            </p>
          </div>
          <div className="grid gap-3">
            {[
              ["Return", fmtSignedPct(risk.metrics.annual_return)],
              ["Sharpe", fmtRatio(risk.metrics.sharpe_ratio)],
              ["Drawdown", fmtPct(risk.metrics.max_drawdown)],
              ["Worst Day", fmtPct(risk.metrics.worst_day)],
            ].map(([label, value]) => (
              <div key={label} className="data-cell-dark rounded-[18px] px-4 py-3">
                <div className="section-kicker text-slate-500">{label}</div>
                <div className="metric-value mt-2 text-2xl font-semibold tracking-[-0.04em] text-white">{value}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:col-span-2 xl:grid-cols-2">
        {metricsConfig.map((metric) => {
          const tone = metric.tone(risk);
          const positive = tone === "good" || tone === "accent";
          const Icon = positive ? TrendingUp : TrendingDown;

          return (
            <Card key={metric.label} className="rounded-[20px]">
              <CardContent className="flex items-start justify-between gap-4 py-5">
                <div>
                  <div className="section-kicker text-[var(--text-faint)]">{metric.label}</div>
                  <div className="metric-value mt-3 text-[28px] font-semibold tracking-[-0.04em] text-[var(--text)]">
                    {metric.value(risk)}
                  </div>
                  <div className="mt-2 text-sm text-[var(--text-soft)]">Benchmark: {risk.metrics.benchmark_ticker}</div>
                </div>
                <div className={`flex h-11 w-11 items-center justify-center rounded-[16px] ${toneClass[tone]}`}>
                  <Icon size={18} />
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
