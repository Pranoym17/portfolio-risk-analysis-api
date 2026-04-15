import { cn } from "@/lib/utils";

type BadgeTone = "neutral" | "good" | "bad" | "accent" | "warn";

export function Badge({
  className,
  tone = "neutral",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  const styles = {
    neutral: "border-[var(--border)] bg-[var(--bg-muted)] text-[var(--text-soft)]",
    good: "border-[#b9dbd8] bg-[var(--teal-soft)] text-[var(--teal)]",
    bad: "border-[#e4bbbb] bg-[var(--red-soft)] text-[var(--red)]",
    accent: "border-[#bfd3e0] bg-[var(--accent-soft)] text-[var(--accent-strong)]",
    warn: "border-[#e7d3a6] bg-[var(--amber-soft)] text-[var(--amber)]",
  } satisfies Record<BadgeTone, string>;

  return (
    <span
      className={cn("inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.08em]", styles[tone], className)}
      {...props}
    />
  );
}
