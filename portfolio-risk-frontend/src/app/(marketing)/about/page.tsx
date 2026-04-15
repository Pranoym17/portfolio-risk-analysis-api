export default function AboutPage() {
  return (
    <main className="site-container py-10">
      <section className="panel hero-panel rounded-[30px] p-8">
        <div className="eyebrow text-[var(--accent)]">About</div>
        <h1 className="mt-3 text-5xl font-semibold tracking-[-0.06em]">A cleaner interface for understanding portfolio risk.</h1>
        <p className="mt-4 max-w-3xl text-base leading-8 text-[var(--text-soft)]">
          This product exists to make portfolio diagnostics easier to trust and easier to read. It brings together holdings construction, ticker validation, benchmark comparison, and rolling metrics in one coherent workspace.
        </p>
      </section>
    </main>
  );
}
