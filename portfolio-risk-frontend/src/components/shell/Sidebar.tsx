"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { usePathname } from "next/navigation";
import {
  ActivitySquare,
  BadgePercent,
  BriefcaseBusiness,
  ChartColumnBig,
  CircleUserRound,
  Plus,
  Search,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";

const primary = [
  { href: "/dashboard", label: "Overview", icon: ChartColumnBig },
  { href: "/portfolios", label: "Portfolios", icon: BriefcaseBusiness },
  { href: "/analytics", label: "Analytics", icon: ActivitySquare },
  { href: "/tickers", label: "Ticker Validation", icon: Search },
];

const secondary = [
  { href: "/settings", label: "Settings", icon: Settings },
  { href: "/profile", label: "Profile", icon: CircleUserRound },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden lg:block">
      <div className="panel sticky top-4 flex min-h-[calc(100vh-32px)] flex-col rounded-[30px] p-4">
        <Link href="/dashboard" className="flex items-center gap-3 border-b border-[var(--line)] pb-5">
          <div className="grid h-12 w-12 place-items-center rounded-[18px] bg-[rgba(123,162,255,0.14)] text-[var(--text)]">
            <BadgePercent size={20} />
          </div>
          <div>
            <div className="eyebrow text-[var(--text-faint)]">Portfolio Intelligence</div>
            <div className="text-lg font-semibold tracking-[-0.04em]">Axiom Risk</div>
          </div>
        </Link>

        <Link
          href="/portfolios/create"
          className="focus-ring mt-5 inline-flex items-center justify-center gap-2 rounded-[18px] border border-[rgba(149,177,255,0.18)] bg-[rgba(123,162,255,0.12)] px-4 py-3 text-sm font-semibold text-[var(--text)] transition hover:bg-[rgba(123,162,255,0.18)]"
        >
          <Plus size={16} />
          New Portfolio
        </Link>

        <nav className="mt-6 grid gap-2">
          {primary.map((item) => {
            const active = pathname.startsWith(item.href);
            const Icon = item.icon;
            return (
              <motion.div key={item.href} whileHover={{ x: 3 }} transition={{ duration: 0.16 }}>
                <Link
                  href={item.href}
                  className={cn(
                    "relative flex items-center gap-3 rounded-[18px] px-4 py-3 text-sm transition",
                    active
                      ? "bg-[rgba(255,255,255,0.08)] text-[var(--text)]"
                      : "text-[var(--text-soft)] hover:bg-[rgba(255,255,255,0.04)] hover:text-[var(--text)]",
                  )}
                >
                  {active ? <span className="absolute left-0 top-3 bottom-3 w-[3px] rounded-full bg-[var(--accent)]" /> : null}
                  <Icon size={17} />
                  <span>{item.label}</span>
                </Link>
              </motion.div>
            );
          })}
        </nav>

        <div className="mt-6 rounded-[22px] border border-[var(--line)] bg-[rgba(255,255,255,0.03)] p-4">
          <div className="eyebrow text-[var(--text-faint)]">Workspace Focus</div>
          <div className="mt-3 space-y-3 text-sm text-[var(--text-soft)]">
            <p>Benchmark-aware review with rolling metrics, allocation context, and holdings quality signals.</p>
            <div className="grid gap-2">
              {[
                ["Sharpe", "1.87"],
                ["Beta", "0.92"],
                ["Max DD", "-8.4%"],
              ].map(([label, value]) => (
                <div key={label} className="glass-strip flex items-center justify-between rounded-[14px] px-3 py-2">
                  <span className="mono text-xs">{label}</span>
                  <span className="mono text-xs text-[var(--text)]">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <nav className="mt-auto grid gap-2 pt-6">
          {secondary.map((item) => {
            const active = pathname.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-[16px] px-4 py-3 text-sm transition",
                  active
                    ? "bg-[rgba(255,255,255,0.08)] text-[var(--text)]"
                    : "text-[var(--text-soft)] hover:bg-[rgba(255,255,255,0.04)] hover:text-[var(--text)]",
                )}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
