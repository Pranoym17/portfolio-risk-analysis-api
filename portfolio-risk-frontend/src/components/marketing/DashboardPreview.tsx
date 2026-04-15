"use client";

import { motion } from "framer-motion";
import { Chart } from "@/components/analytics/Chart";
import { chartPalette } from "@/lib/design-system";

const days = Array.from({ length: 20 }, (_, index) => `D${index + 1}`);
const returns = [2, 4, 5, 4, 7, 9, 11, 10, 13, 14, 13, 15, 17, 18, 17, 20, 22, 24, 23, 25];
const benchmark = [1, 2, 4, 5, 5, 7, 8, 9, 9, 10, 11, 12, 13, 13, 14, 15, 16, 17, 18, 18];

export function DashboardPreview() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay: 0.12 }}
      className="panel hero-panel rounded-[30px] p-4 sm:p-5"
    >
      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(7,13,24,0.55)] p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="eyebrow text-[var(--text-faint)]">Live Workspace</div>
              <h3 className="mt-2 text-3xl font-semibold tracking-[-0.05em]">Growth Strategy</h3>
            </div>
            <div className="grid gap-2 text-right">
              <span className="eyebrow text-[var(--text-faint)]">Net Value</span>
              <span className="mono text-xl text-[var(--text)]">$1.28M</span>
            </div>
          </div>
          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            {[
              ["Sharpe", "1.87"],
              ["Volatility", "14.2%"],
              ["VaR", "-2.8%"],
            ].map(([label, value]) => (
              <div key={label} className="rounded-[18px] border border-[var(--line)] bg-[rgba(255,255,255,0.03)] px-4 py-3">
                <div className="eyebrow text-[var(--text-faint)]">{label}</div>
                <div className="metric-value mt-2 text-2xl font-semibold">{value}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 rounded-[22px] border border-[var(--line)] bg-[rgba(255,255,255,0.02)] p-3">
            <Chart
              style={{ height: 260 }}
              option={{
                animationDuration: 700,
                grid: { top: 18, left: 36, right: 14, bottom: 28 },
                tooltip: {
                  trigger: "axis",
                  backgroundColor: chartPalette.tooltipBg,
                  borderColor: chartPalette.border,
                  borderWidth: 1,
                  textStyle: { color: chartPalette.tooltipText },
                },
                xAxis: {
                  type: "category",
                  data: days,
                  boundaryGap: false,
                  axisLine: { lineStyle: { color: chartPalette.border } },
                  axisLabel: { color: chartPalette.axis, fontSize: 10 },
                },
                yAxis: {
                  type: "value",
                  axisLabel: { color: chartPalette.axis, fontSize: 10 },
                  splitLine: { lineStyle: { color: chartPalette.grid } },
                },
                series: [
                  {
                    data: returns,
                    type: "line",
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { color: chartPalette.primary, width: 3 },
                    areaStyle: { color: chartPalette.areaPrimary },
                  },
                  {
                    data: benchmark,
                    type: "line",
                    smooth: true,
                    showSymbol: false,
                    lineStyle: { color: chartPalette.teal, width: 2 },
                    areaStyle: { color: chartPalette.areaTeal },
                  },
                ],
              }}
            />
          </div>
        </div>

        <div className="grid gap-4">
          <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.04)] p-4">
            <div className="eyebrow text-[var(--text-faint)]">Allocation</div>
            <div className="mt-3 space-y-3">
              {[
                ["Technology", "56%"],
                ["Healthcare", "18%"],
                ["Financials", "14%"],
                ["Other", "12%"],
              ].map(([label, value], index) => (
                <div key={label}>
                  <div className="mb-1 flex items-center justify-between text-sm text-[var(--text-soft)]">
                    <span>{label}</span>
                    <span className="mono text-[var(--text)]">{value}</span>
                  </div>
                  <div className="h-2 rounded-full bg-[rgba(255,255,255,0.06)]">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: value,
                        background: [chartPalette.primary, chartPalette.teal, chartPalette.amber, chartPalette.red][index],
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.04)] p-4">
            <div className="eyebrow text-[var(--text-faint)]">Signals</div>
            <div className="mt-3 grid gap-3">
              {[
                ["Concentration risk elevated", "Technology exposure remains above target policy band."],
                ["Benchmark beta contained", "Beta remains under 1.0 versus the selected benchmark."],
              ].map(([title, body]) => (
                <div key={title} className="rounded-[18px] border border-[var(--line)] bg-[rgba(7,13,24,0.48)] p-4">
                  <p className="text-sm font-semibold">{title}</p>
                  <p className="mt-1 text-sm leading-6 text-[var(--text-soft)]">{body}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
