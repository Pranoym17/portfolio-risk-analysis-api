import Link from "next/link";

const links = [
  { href: "/features", label: "Platform" },
  { href: "/pricing", label: "Pricing" },
  { href: "/about", label: "About" },
  { href: "/contact", label: "Support" },
];

export function SiteFooter() {
  return (
    <footer className="site-container py-10">
      <div className="grid gap-6 rounded-[28px] border border-[var(--line)] bg-[rgba(255,255,255,0.03)] px-6 py-8 lg:grid-cols-[1.4fr_0.6fr]">
        <div>
          <div className="eyebrow text-[var(--text-faint)]">Axiom Risk</div>
          <h2 className="mt-2 text-2xl font-semibold tracking-[-0.05em]">Portfolio analytics with benchmark context, holdings discipline, and risk clarity.</h2>
          <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
            Built for investors and analysts who need a cleaner view of rolling metrics, allocation concentration, and portfolio diagnostics.
          </p>
        </div>
        <div className="grid gap-2 text-sm text-[var(--text-soft)]">
          {links.map((item) => (
            <Link key={item.href} href={item.href} className="transition hover:text-[var(--text)]">
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </footer>
  );
}
