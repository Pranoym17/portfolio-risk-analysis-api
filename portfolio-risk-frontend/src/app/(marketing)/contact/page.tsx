export default function ContactPage() {
  return (
    <main className="site-container py-10">
      <section className="panel rounded-[30px] p-8">
        <div className="eyebrow text-[var(--accent)]">Support</div>
        <h1 className="mt-3 text-5xl font-semibold tracking-[-0.06em]">Need help with the workflow?</h1>
        <p className="mt-4 max-w-2xl text-base leading-8 text-[var(--text-soft)]">
          Reach out if you need help understanding ticker validation, portfolio construction, or how to interpret the analytics surfaces.
        </p>
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          <div className="glass-strip rounded-[22px] p-5">
            <div className="eyebrow text-[var(--text-faint)]">Email</div>
            <p className="mt-2 text-lg font-semibold">support@axiomrisk.app</p>
          </div>
          <div className="glass-strip rounded-[22px] p-5">
            <div className="eyebrow text-[var(--text-faint)]">Response</div>
            <p className="mt-2 text-lg font-semibold">Usually within one business day</p>
          </div>
        </div>
      </section>
    </main>
  );
}
