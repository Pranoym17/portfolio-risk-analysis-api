import { cn } from "@/lib/utils";

type BadgeTone = "neutral" | "good" | "bad" | "accent" | "warn";

export function Badge({
  className,
  tone = "neutral",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  const styles = {
    neutral: "border-[var(--line)] bg-[rgba(255,255,255,0.04)] text-[var(--text-soft)]",
    good: "border-[rgba(88,199,152,0.18)] bg-[rgba(88,199,152,0.12)] text-[var(--success)]",
    bad: "border-[rgba(255,143,152,0.18)] bg-[rgba(255,143,152,0.12)] text-[var(--danger)]",
    accent: "border-[rgba(123,162,255,0.18)] bg-[rgba(123,162,255,0.12)] text-[var(--accent)]",
    warn: "border-[rgba(255,183,111,0.18)] bg-[rgba(255,183,111,0.12)] text-[var(--accent-3)]",
  } satisfies Record<BadgeTone, string>;

  return (
    <span
      className={cn("inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.08em]", styles[tone], className)}
      {...props}
    />
  );
}
