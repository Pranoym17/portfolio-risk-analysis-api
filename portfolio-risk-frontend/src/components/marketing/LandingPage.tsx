import Link from "next/link";
import { ArrowRight, CheckCircle2, ShieldAlert, WalletCards } from "lucide-react";
import { DashboardPreview } from "@/components/marketing/DashboardPreview";
import { Button } from "@/components/ui/Button";

const featureCards = [
  {
    title: "Portfolio creation with holdings discipline",
    body: "Build portfolios, structure weights, and prepare holdings for validation before running analytics.",
  },
  {
    title: "Benchmark-aware risk views",
    body: "Compare volatility, Sharpe, beta, VaR, and drawdown with rolling metrics and benchmark context.",
  },
  {
    title: "Diagnostics built for review",
    body: "See dropped tickers, allocation concentration, and analyst-style summaries in one premium workspace.",
  },
];

export function LandingPage() {
  return (
    <main className="site-container py-8 pb-16">
      <section className="grid gap-8 xl:grid-cols-[0.92fr_1.08fr] xl:items-end">
        <div className="pt-8">
          <div className="eyebrow text-[var(--accent)]">Modern Portfolio Risk Analysis</div>
          <h1 className="mt-4 max-w-4xl text-5xl font-semibold tracking-[-0.07em] text-[var(--text)] sm:text-6xl xl:text-7xl">
            A sharper web terminal for portfolio risk, holdings diagnostics, and benchmark comparison.
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-8 text-[var(--text-soft)]">
            Axiom Risk helps investors, analysts, and technically literate users build portfolios, validate holdings, inspect rolling analytics, and review concentration signals in one coherent product.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/signup">
              <Button className="min-w-[180px]">
                Open Workspace
                <ArrowRight size={16} />
              </Button>
            </Link>
            <Link href="/features">
              <Button variant="secondary" className="min-w-[180px]">
                Explore Platform
              </Button>
            </Link>
          </div>

          <div className="mt-10 grid gap-4 sm:grid-cols-3">
            {[
              ["Rolling metrics", "Sharpe, volatility, beta, drawdown, VaR"],
              ["Holdings workflow", "Validation, weighting, and ticker hygiene"],
              ["Benchmark context", "Compare portfolio sensitivity and return structure"],
            ].map(([title, body]) => (
              <div key={title} className="glass-strip rounded-[22px] p-4">
                <p className="text-sm font-semibold">{title}</p>
                <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">{body}</p>
              </div>
            ))}
          </div>
        </div>

        <DashboardPreview />
      </section>

      <section className="mt-10 grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
        <div className="panel rounded-[28px] p-6">
          <div className="eyebrow text-[var(--text-faint)]">Why it feels better</div>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-0.05em]">A product-shaped workflow instead of disconnected finance widgets.</h2>
          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            {featureCards.map((item) => (
              <div key={item.title} className="rounded-[22px] border border-[var(--line)] bg-[rgba(255,255,255,0.03)] p-4">
                <p className="text-sm font-semibold text-[var(--text)]">{item.title}</p>
                <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">{item.body}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-5">
          <div className="panel hero-panel rounded-[28px] p-6">
            <ShieldAlert className="text-[var(--accent)]" size={20} />
            <h3 className="mt-4 text-2xl font-semibold tracking-[-0.05em]">See what breaks before it reaches the risk engine.</h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">
              Ticker validation, dropped symbol feedback, and allocation checks keep analysis outputs more reliable.
            </p>
          </div>
          <div className="panel rounded-[28px] p-6">
            <WalletCards className="text-[var(--accent-2)]" size={20} />
            <h3 className="mt-4 text-2xl font-semibold tracking-[-0.05em]">Built for academic rigor and practical investing.</h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">
              Use it to learn portfolio risk or to manage a real analytical workflow. The interface is designed to support both.
            </p>
          </div>
        </div>
      </section>

      <section className="mt-10 panel rounded-[30px] p-6 sm:p-8">
        <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <div>
            <div className="eyebrow text-[var(--text-faint)]">What you can do</div>
            <h2 className="mt-3 text-3xl font-semibold tracking-[-0.05em]">Run a disciplined portfolio workflow from creation to diagnostics.</h2>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {[
              "Create a portfolio and manage holdings",
              "Validate tickers before they break analytics",
              "Review rolling volatility, Sharpe, beta, and VaR",
              "Compare allocation and benchmark sensitivity",
            ].map((item) => (
              <div key={item} className="flex items-start gap-3 rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.03)] p-4">
                <CheckCircle2 className="mt-0.5 shrink-0 text-[var(--accent)]" size={18} />
                <p className="text-sm leading-6 text-[var(--text-soft)]">{item}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
