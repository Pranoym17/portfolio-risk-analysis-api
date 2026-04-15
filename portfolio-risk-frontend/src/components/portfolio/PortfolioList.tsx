"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowRight, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { deletePortfolio } from "@/lib/api";
import type { PortfolioOut } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { EmptyState, SkeletonBlock } from "@/components/ui/StatePanel";
import { getErrorMessage } from "@/lib/utils";

export function PortfolioList({
  items,
  loading,
  onChanged,
}: {
  items: PortfolioOut[];
  loading: boolean;
  onChanged: () => void;
}) {
  const [query, setQuery] = useState("");

  const visible = items.filter((item) => {
    const term = query.trim().toLowerCase();
    if (!term) return true;
    return item.name.toLowerCase().includes(term) || String(item.id).includes(term);
  });

  async function onDelete(id: number) {
    if (!window.confirm("Delete this portfolio and its holdings?")) return;
    try {
      await deletePortfolio(id);
      toast.success("Portfolio removed");
      onChanged();
    } catch (error: unknown) {
      toast.error(getErrorMessage(error, "Delete failed"));
    }
  }

  return (
    <Card className="rounded-[20px]">
      <CardHeader>
        <div>
          <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Portfolio Directory</div>
          <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Your portfolios</CardTitle>
          <CardDescription>Browse, open, and retire user-owned portfolios.</CardDescription>
        </div>
        <div className="w-full max-w-[240px]">
          <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by name or id" />
        </div>
      </CardHeader>

      <CardContent>
        {loading ? (
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, index) => (
              <SkeletonBlock key={index} className="h-[76px]" />
            ))}
          </div>
        ) : visible.length === 0 ? (
          <EmptyState
            title="No portfolios available"
            body="Create a portfolio to begin constructing holdings and reviewing risk metrics."
          />
        ) : (
          <div className="space-y-3">
            {visible.map((portfolio) => (
              <div
                key={portfolio.id}
                className="group rounded-[18px] border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-4 transition hover:border-[var(--border-strong)] hover:shadow-[var(--shadow-sm)]"
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-base font-semibold text-[var(--text)]">{portfolio.name}</div>
                    <div className="mt-1 text-sm text-[var(--text-soft)]">
                      Portfolio #{portfolio.id} · {portfolio.holdings.length} holdings staged
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button variant="ghost" onClick={() => onDelete(portfolio.id)}>
                      <Trash2 size={16} />
                      Delete
                    </Button>
                    <Link
                      href={`/portfolios/${portfolio.id}`}
                      className="focus-ring inline-flex min-h-10 items-center gap-2 rounded-[12px] border border-[var(--border)] bg-[var(--bg-muted)] px-3.5 text-sm font-medium text-[var(--text)] transition hover:bg-[var(--bg-subtle)]"
                    >
                      Open workspace
                      <ArrowRight size={16} />
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
