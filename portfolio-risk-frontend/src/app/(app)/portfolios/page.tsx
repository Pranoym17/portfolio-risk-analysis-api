"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { FolderPlus } from "lucide-react";
import { PortfolioCard } from "@/components/portfolio/PortfolioCard";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/StatePanel";
import { listPortfolios } from "@/lib/api";
import type { PortfolioOut } from "@/lib/types";
import { getErrorMessage } from "@/lib/utils";

export default function PortfoliosPage() {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<PortfolioOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        setItems(await listPortfolios());
      } catch (nextError) {
        setError(getErrorMessage(nextError, "Unable to load portfolios."));
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  const filtered = useMemo(
    () => items.filter((item) => item.name.toLowerCase().includes(query.toLowerCase())),
    [items, query],
  );

  if (loading) return <LoadingState title="Loading portfolios..." />;

  return (
    <div className="space-y-5">
      {error ? <ErrorState title="Portfolio library unavailable" body={error} /> : null}

      <section className="panel hero-panel rounded-[30px] p-6">
        <div className="grid gap-5 xl:grid-cols-[0.92fr_1.08fr]">
          <div>
            <div className="eyebrow text-[var(--accent)]">Portfolio Library</div>
            <h2 className="mt-3 text-4xl font-semibold tracking-[-0.06em]">Search, compare, and open portfolio workspaces faster.</h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
              This view is designed to feel like a curated library rather than a plain CRUD list, so it is easier to move from overview to analysis.
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-end">
            <label className="block min-w-[260px] flex-1 sm:flex-none">
              <span className="eyebrow text-[var(--text-faint)]">Search</span>
              <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find by portfolio name" className="mt-2" />
            </label>
            <Link href="/portfolios/create">
              <Button className="w-full sm:w-auto">
                <FolderPlus size={16} />
                New Portfolio
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {!filtered.length ? (
        <EmptyState
          title="No portfolios found"
          body={items.length ? "Try a different search term or create a new portfolio." : "Create your first portfolio to begin the analytics workflow."}
          actionLabel="Create Portfolio"
          onAction={() => {
            window.location.href = "/portfolios/create";
          }}
        />
      ) : (
        <section className="grid gap-5 xl:grid-cols-2">
          {filtered.map((portfolio, index) => (
            <PortfolioCard key={portfolio.id} portfolio={portfolio} featured={index === 0} />
          ))}
        </section>
      )}
    </div>
  );
}
