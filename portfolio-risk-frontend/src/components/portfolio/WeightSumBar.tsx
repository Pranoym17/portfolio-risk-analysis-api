import { fmtPct } from "@/lib/utils";

export function WeightSumBar({ total }: { total: number }) {
  const difference = Math.abs(total - 1);
  const status = difference < 0.0001 ? "balanced" : total < 1 ? "under" : "over";
  const tone =
    status === "balanced"
      ? "bg-[var(--success)]"
      : status === "under"
      ? "bg-[var(--accent-3)]"
      : "bg-[var(--danger)]";

  return (
    <div className="rounded-[16px] border border-[var(--line)] bg-[rgba(255,255,255,0.03)] p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text-faint)]">Weight Discipline</div>
          <div className="mt-1 text-sm text-[var(--text-soft)]">
            Portfolios must sum to 100% before holdings can be saved.
          </div>
        </div>
        <div className="metric-value text-right text-lg font-semibold text-[var(--text)]">{fmtPct(total, 2)}</div>
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-[rgba(255,255,255,0.06)]">
        <div className={`h-full rounded-full transition-all ${tone}`} style={{ width: `${Math.min(total, 1) * 100}%` }} />
      </div>
    </div>
  );
}
