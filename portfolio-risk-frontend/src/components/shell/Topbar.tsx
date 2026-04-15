"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Bell, ChevronDown, LogOut, Search, Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { listPortfolios } from "@/lib/api";
import type { PortfolioOut } from "@/lib/types";
import { useAuth } from "@/components/providers/AuthProvider";

export function Topbar({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  const router = useRouter();
  const { user, signOut } = useAuth();
  const [portfolios, setPortfolios] = useState<PortfolioOut[]>([]);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    listPortfolios()
      .then((data) => {
        setPortfolios(data);
        if (data[0]) setSelected(String(data[0].id));
      })
      .catch(() => setPortfolios([]));
  }, []);

  return (
    <header className="panel rounded-[28px] px-4 py-4 sm:px-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <div className="eyebrow text-[var(--accent)]">Authenticated Workspace</div>
          <h1 className="mt-1 text-[30px] font-semibold tracking-[-0.06em] text-[var(--text)]">{title}</h1>
          <p className="mt-1 max-w-2xl text-sm text-[var(--text-soft)]">{subtitle}</p>
        </div>

        <div className="flex flex-wrap items-center gap-3 xl:justify-end">
          <div className="glass-strip relative min-w-[240px] rounded-[18px] px-3 py-2.5">
            <div className="eyebrow text-[var(--text-faint)]">Portfolio</div>
            <select
              value={selected}
              onChange={(event) => {
                setSelected(event.target.value);
                if (event.target.value) router.push(`/portfolios/${event.target.value}`);
              }}
              className="mt-1 h-8 w-full appearance-none bg-transparent pr-6 text-sm font-semibold text-[var(--text)] outline-none"
            >
              <option value="">Select portfolio</option>
              {portfolios.map((portfolio) => (
                <option key={portfolio.id} value={portfolio.id}>
                  {portfolio.name}
                </option>
              ))}
            </select>
            <ChevronDown size={16} className="pointer-events-none absolute right-3 top-[31px] text-[var(--text-faint)]" />
          </div>

          <div className="relative min-w-[240px] flex-1 sm:flex-none">
            <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)]" />
            <Input placeholder="Search holdings or pages" className="pl-10" />
          </div>

          <Link href="/portfolios/create">
            <Button variant="secondary">
              <Sparkles size={15} />
              Create
            </Button>
          </Link>

          <Button variant="quiet" className="min-w-11 px-3" title="Alerts">
            <Bell size={16} />
          </Button>

          <div className="glass-strip rounded-[18px] px-3 py-2.5">
            <div className="eyebrow text-[var(--text-faint)]">Account</div>
            <div className="mt-1 text-sm font-semibold text-[var(--text)]">{user?.email ?? "Loading..."}</div>
          </div>

          <Button variant="quiet" onClick={signOut}>
            <LogOut size={16} />
          </Button>
        </div>
      </div>
    </header>
  );
}
