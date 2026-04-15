export function ControlPanel({
  title,
  description,
  children,
  className,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel rounded-[26px] ${className ?? ""}`}>
      <div className="border-b border-[var(--line)] px-5 py-4">
        <div className="eyebrow text-[var(--text-faint)]">Controls</div>
        <h3 className="mt-1 text-2xl font-semibold tracking-[-0.05em] text-[var(--text)]">{title}</h3>
        {description ? <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">{description}</p> : null}
      </div>
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}
