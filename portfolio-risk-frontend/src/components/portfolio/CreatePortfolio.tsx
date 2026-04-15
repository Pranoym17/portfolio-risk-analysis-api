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
    <Card className="rounded-[20px]">
      <CardHeader className="block">
        <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Portfolio Creation</div>
        <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Create a new portfolio shell</CardTitle>
        <CardDescription>Start with a portfolio name, then move into holdings construction and analytics.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <label className="block space-y-2">
          <span className="text-sm font-medium text-[var(--text)]">Portfolio name</span>
          <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Global Core Equity" />
        </label>

        <Button className="w-full" loading={loading} onClick={submit}>
          Create Portfolio
        </Button>
      </CardContent>
    </Card>
  );
}
