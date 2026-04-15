const plans = [
  { name: "Explorer", price: "$0", body: "Perfect for learning the product and validating the analysis workflow." },
  { name: "Analyst", price: "$24", body: "For active users managing several portfolios and regular benchmark review." },
  { name: "Studio", price: "$79", body: "For power users who want richer workflows, team context, and more persistent analysis." },
];

export default function PricingPage() {
  return (
    <main className="site-container py-10">
      <section className="panel hero-panel rounded-[30px] p-8">
        <div className="eyebrow text-[var(--accent)]">Pricing</div>
        <h1 className="mt-3 text-5xl font-semibold tracking-[-0.06em]">Simple pricing for a serious portfolio analytics workspace.</h1>
        <p className="mt-4 max-w-2xl text-base leading-8 text-[var(--text-soft)]">
          Start with the core workflow, then move into heavier analysis and collaboration when you need it.
        </p>
      </section>
      <section className="mt-8 grid gap-5 lg:grid-cols-3">
        {plans.map((plan) => (
          <div key={plan.name} className="panel rounded-[28px] p-6">
            <div className="eyebrow text-[var(--text-faint)]">{plan.name}</div>
            <div className="mt-3 text-5xl font-semibold tracking-[-0.06em]">{plan.price}<span className="text-lg text-[var(--text-faint)]">/mo</span></div>
            <p className="mt-4 text-sm leading-7 text-[var(--text-soft)]">{plan.body}</p>
          </div>
        ))}
      </section>
    </main>
  );
}
