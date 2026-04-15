"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type MetricCardProps = {
  label: string;
  value: string | number;
  detail?: string;
  points?: number[];
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
  className?: string;
};

export function MetricCard({
  label,
  value,
  detail,
  points = [],
  tone = "neutral",
  className,
}: MetricCardProps) {
  const badgeStyles = {
    accent: "bg-[rgba(123,162,255,0.12)] text-[var(--accent)]",
    good: "bg-[rgba(88,199,152,0.12)] text-[var(--success)]",
    warn: "bg-[rgba(255,183,111,0.12)] text-[var(--accent-3)]",
    bad: "bg-[rgba(255,143,152,0.12)] text-[var(--danger)]",
    neutral: "bg-[rgba(255,255,255,0.06)] text-[var(--text-soft)]",
  }[tone];

  const barStyles = {
    accent: "bg-[var(--accent)]",
    good: "bg-[var(--success)]",
    warn: "bg-[var(--accent-3)]",
    bad: "bg-[var(--danger)]",
    neutral: "bg-[var(--text-faint)]",
  }[tone];

  const max = Math.max(...points, 1);

  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
      className={cn("panel panel-muted kpi-glow rounded-[22px] px-4 py-4", className)}
    >
      <div className="relative z-10 flex items-start justify-between gap-4">
        <div>
          <div className="eyebrow text-[var(--text-faint)]">{label}</div>
          <div className="metric-value mt-3 text-[34px] font-semibold tracking-[-0.06em] text-[var(--text)]">{value}</div>
          {detail ? <div className="mt-2 text-sm text-[var(--text-soft)]">{detail}</div> : null}
        </div>
        <div className={cn("rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.12em]", badgeStyles)}>Live</div>
      </div>

      {points.length ? (
        <div className="mt-5 flex h-14 items-end gap-1">
          {points.map((point, index) => (
            <div
              key={`${label}-${index}`}
              className={cn("flex-1 rounded-full opacity-90", barStyles)}
              style={{ height: `${Math.max(12, (point / max) * 100)}%` }}
            />
          ))}
        </div>
      ) : null}
    </motion.div>
  );
}
