"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { FolderPlus, ListChecks } from "lucide-react";
import { toast } from "sonner";
import { createPortfolio } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { getErrorMessage } from "@/lib/utils";

export default function CreatePortfolioPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleCreate() {
    if (!name.trim()) {
      toast.error("Enter a portfolio name first.");
      return;
    }

    setLoading(true);
    try {
      const created = await createPortfolio(name.trim());
      toast.success("Portfolio created");
      router.push(`/portfolios/${created.id}`);
    } catch (error) {
      toast.error(getErrorMessage(error, "Unable to create the portfolio."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[0.88fr_1.12fr]">
      <section className="panel hero-panel rounded-[30px] p-6">
        <div className="eyebrow text-[var(--accent)]">Portfolio Creation</div>
        <h2 className="mt-3 text-4xl font-semibold tracking-[-0.06em]">Create the portfolio shell, then move straight into holdings and analytics.</h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
          The creation flow stays lightweight on purpose. Name the portfolio, enter the workspace, then build the holdings table where the actual risk workflow begins.
        </p>
      </section>

      <section className="panel rounded-[30px] p-6">
        <div className="eyebrow text-[var(--text-faint)]">Create Portfolio</div>
        <div className="mt-6 grid gap-6">
          <label className="block space-y-2">
            <span className="text-sm font-medium text-[var(--text)]">Portfolio Name</span>
            <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Growth Strategy" />
          </label>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="glass-strip rounded-[22px] p-4">
              <FolderPlus className="text-[var(--accent)]" size={18} />
              <p className="mt-3 text-sm font-semibold">Step 1</p>
              <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">Create the portfolio shell with a clear internal name.</p>
            </div>
            <div className="glass-strip rounded-[22px] p-4">
              <ListChecks className="text-[var(--accent-2)]" size={18} />
              <p className="mt-3 text-sm font-semibold">Step 2</p>
              <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">Move into the workspace, add holdings, validate tickers, and run analytics.</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button onClick={handleCreate} loading={loading}>
              <FolderPlus size={16} />
              Create Portfolio
            </Button>
            <Button variant="secondary" onClick={() => router.push("/portfolios")}>
              Back to Library
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
