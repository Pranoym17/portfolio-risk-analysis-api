const sections = [
  ["Portfolio workflows", "Create portfolios, structure weights, manage holdings, and review diagnostics from one environment."],
  ["Risk analytics", "Inspect annual return, volatility, Sharpe, Sortino, beta, VaR, worst day, and max drawdown."],
  ["Benchmark comparison", "Use benchmark-aware analytics to understand sensitivity and performance context."],
  ["Review surfaces", "Tables, charts, and analyst-style summaries are designed for reading, not just displaying data."],
];

export default function FeaturesPage() {
  return (
    <main className="site-container py-10">
      <section className="panel hero-panel rounded-[30px] p-8">
        <div className="eyebrow text-[var(--accent)]">Platform Overview</div>
        <h1 className="mt-3 text-5xl font-semibold tracking-[-0.06em]">Portfolio analytics built around real review workflows.</h1>
        <p className="mt-4 max-w-3xl text-base leading-8 text-[var(--text-soft)]">
          Axiom Risk is not just a page of cards. It is a portfolio analysis product organized around holdings quality, benchmark context, and readable quantitative surfaces.
        </p>
      </section>
      <section className="mt-8 grid gap-5 md:grid-cols-2">
        {sections.map(([title, body]) => (
          <div key={title} className="panel rounded-[28px] p-6">
            <h2 className="text-2xl font-semibold tracking-[-0.05em]">{title}</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">{body}</p>
          </div>
        ))}
      </section>
    </main>
  );
}
