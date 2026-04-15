"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BriefcaseBusiness, ChartColumnIncreasing, ShieldCheck } from "lucide-react";
import { useAuth } from "@/components/providers/AuthProvider";
import { cn } from "@/lib/utils";

const items = [
  { href: "/portfolios", label: "Overview", icon: ChartColumnIncreasing },
];

export function Sidebar() {
  const pathname = usePathname();
  const { user } = useAuth();

  return (
    <aside className="hidden w-[280px] shrink-0 lg:block">
      <div className="surface sticky top-5 rounded-[22px] p-5">
        <div className="flex items-center gap-3 border-b border-[var(--border)] pb-5">
          <div className="grid h-11 w-11 place-items-center rounded-[14px] bg-[var(--text)] text-white">
            <BriefcaseBusiness size={20} />
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Portfolio Risk</div>
            <div className="mt-1 text-lg font-semibold text-[var(--text)]">Analyst Workspace</div>
          </div>
        </div>

        <nav className="space-y-1 py-5">
          {items.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-[14px] px-3 py-3 text-sm transition-colors",
                  active
                    ? "bg-[var(--text)] text-white"
                    : "text-[var(--text-soft)] hover:bg-[var(--bg-muted)] hover:text-[var(--text)]",
                )}
              >
                <Icon size={18} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="rounded-[18px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
          <div className="flex items-center gap-2 text-[var(--accent)]">
            <ShieldCheck size={16} />
            <span className="text-xs font-semibold uppercase tracking-[0.08em]">Authenticated Session</span>
          </div>
          <p className="mt-3 text-sm leading-6 text-[var(--text-soft)]">
            Signed in as <span className="font-medium text-[var(--text)]">{user?.email ?? "Loading"}</span>.
          </p>
          <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
            Use this workspace to create portfolios, refine holdings, and inspect risk concentration with benchmark context.
          </p>
        </div>
      </div>
    </aside>
  );
}
