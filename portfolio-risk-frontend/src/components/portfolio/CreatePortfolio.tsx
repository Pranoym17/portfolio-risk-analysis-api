"use client";

import { useState } from "react";
import { toast } from "sonner";
import { createPortfolio } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { getErrorMessage } from "@/lib/utils";

export function CreatePortfolio({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed) {
      toast.error("Portfolio name is required");
      return;
    }

    setLoading(true);
    try {
      await createPortfolio(trimmed);
      toast.success("Portfolio created");
      setName("");
      onCreated();
    } catch (error: unknown) {
      toast.error(getErrorMessage(error, "Unable to create portfolio"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="rounded-[24px]">
      <CardHeader className="block">
        <div className="section-kicker text-[var(--accent)]">Portfolio Creation</div>
        <CardTitle className="mt-2 text-2xl tracking-[-0.04em]">Launch a new portfolio shell</CardTitle>
        <CardDescription>Start with a name, then move straight into holdings construction, validation, and analytics.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <label className="block space-y-2">
          <span className="text-sm font-semibold text-[var(--text)]">Portfolio name</span>
          <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Global Core Equity" />
        </label>

        <div className="rounded-[18px] border border-[var(--border)] bg-[var(--bg-muted)] p-4 text-sm leading-7 text-[var(--text-soft)]">
          Good portfolio names read like real sleeves or mandates, for example <span className="font-semibold text-[var(--text)]">US Quality Compounders</span> or <span className="font-semibold text-[var(--text)]">Balanced Multi-Asset Core</span>.
        </div>

        <Button className="w-full" loading={loading} onClick={submit}>
          Create Portfolio
        </Button>
      </CardContent>
    </Card>
  );
}
