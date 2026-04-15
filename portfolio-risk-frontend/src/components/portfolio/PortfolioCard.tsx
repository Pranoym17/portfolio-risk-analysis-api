import Link from "next/link";
import { ArrowRight, Layers3 } from "lucide-react";
import type { PortfolioOut } from "@/lib/types";

export function PortfolioCard({
  portfolio,
  featured = false,
}: {
  portfolio: PortfolioOut;
  featured?: boolean;
}) {
  return (
    <div className={`panel rounded-[24px] p-5 ${featured ? "hero-panel" : "panel-muted"}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="eyebrow text-[var(--text-faint)]">Portfolio #{portfolio.id}</div>
          <h3 className="mt-2 text-2xl font-semibold tracking-[-0.05em] text-[var(--text)]">{portfolio.name}</h3>
          <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
            {portfolio.holdings.length} holdings ready for validation, analytics, and benchmark comparison.
          </p>
        </div>
        <div className="grid h-12 w-12 place-items-center rounded-[18px] border border-[var(--line)] bg-[rgba(255,255,255,0.04)] text-[var(--text)]">
          <Layers3 size={18} />
        </div>
      </div>

      <div className="mt-5 flex items-center justify-between">
        <div className="glass-strip rounded-full px-3 py-1.5 text-xs text-[var(--text-soft)]">
          Holdings: <span className="mono text-[var(--text)]">{portfolio.holdings.length}</span>
        </div>
        <Link
          href={`/portfolios/${portfolio.id}`}
          className="focus-ring inline-flex items-center gap-2 rounded-[14px] border border-[var(--line)] bg-[rgba(255,255,255,0.04)] px-4 py-2 text-sm font-semibold text-[var(--text)] transition hover:bg-[rgba(255,255,255,0.08)]"
        >
          Open Workspace
          <ArrowRight size={15} />
        </Link>
      </div>
    </div>
  );
}
