"use client";

import { LogOut } from "lucide-react";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/components/providers/AuthProvider";
import { titleFromPath } from "@/lib/utils";

export function Topbar() {
  const pathname = usePathname();
  const { user, signOut } = useAuth();

  return (
    <header className="surface rounded-[20px] px-4 py-4 sm:px-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Portfolio Risk Analysis API</div>
          <h1 className="mt-1 text-[26px] font-semibold tracking-[-0.03em] text-[var(--text)]">{titleFromPath(pathname)}</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-soft)]">
            Professional portfolio construction, holdings management, and risk diagnostics with clean benchmark-aware analytics.
          </p>
        </div>

        <div className="flex items-center justify-between gap-3 lg:justify-end">
          <div className="rounded-[14px] border border-[var(--border)] bg-[var(--bg-muted)] px-3 py-2">
            <div className="text-[11px] uppercase tracking-[0.1em] text-[var(--text-faint)]">Account</div>
            <div className="mt-1 text-sm font-medium text-[var(--text)]">{user?.email ?? "Loading..."}</div>
          </div>
          <Button variant="secondary" onClick={signOut}>
            <LogOut size={16} />
            Sign Out
          </Button>
        </div>
      </div>
    </header>
  );
}
