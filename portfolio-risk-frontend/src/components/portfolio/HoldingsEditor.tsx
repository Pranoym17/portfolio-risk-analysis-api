"use client";

import { useEffect, useState } from "react";
import { LoaderCircle, Plus, SearchCheck, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { replaceHoldings, validateTicker } from "@/lib/api";
import type { HoldingIn, PortfolioOut, TickerValidationResult } from "@/lib/types";
import { getErrorMessage, sumWeights, toWeightInput } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { WeightSumBar } from "./WeightSumBar";

type EditableHolding = {
  ticker: string;
  weight: number;
  validation?: TickerValidationResult | null;
  validating?: boolean;
};

export function HoldingsEditor({
  portfolio,
  onUpdated,
}: {
  portfolio: PortfolioOut;
  onUpdated: (portfolio: PortfolioOut) => void;
}) {
  const [rows, setRows] = useState<EditableHolding[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setRows(
      portfolio.holdings.length
        ? portfolio.holdings.map((holding) => ({
            ticker: holding.ticker,
            weight: holding.weight,
            validation: null,
            validating: false,
          }))
        : [
            { ticker: "AAPL", weight: 0.4, validation: null, validating: false },
            { ticker: "MSFT", weight: 0.35, validation: null, validating: false },
            { ticker: "SPY", weight: 0.25, validation: null, validating: false },
          ],
    );
  }, [portfolio.holdings]);

  function updateRow(index: number, patch: Partial<EditableHolding>) {
    setRows((current) => current.map((row, rowIndex) => (rowIndex === index ? { ...row, ...patch } : row)));
  }

  function removeRow(index: number) {
    setRows((current) => current.filter((_, rowIndex) => rowIndex !== index));
  }

  function addRow() {
    setRows((current) => [...current, { ticker: "", weight: 0, validation: null, validating: false }]);
  }

  async function runValidation(index: number) {
    const ticker = rows[index]?.ticker.trim().toUpperCase();
    if (!ticker) return;

    updateRow(index, { validating: true });
    try {
      const result = await validateTicker(ticker);
      updateRow(index, {
        ticker,
        validation: result,
        validating: false,
      });
    } catch {
      updateRow(index, {
        validation: {
          ticker,
          is_valid: false,
          rows_returned: 0,
          error: "Validation unavailable",
        },
        validating: false,
      });
    }
  }

  async function save() {
    const payload: HoldingIn[] = rows
      .filter((row) => row.ticker.trim())
      .map((row) => ({
        ticker: row.ticker.trim().toUpperCase(),
        weight: Number(row.weight),
      }));

    if (payload.length === 0) {
      toast.error("Add at least one holding");
      return;
    }

    setSaving(true);
    try {
      const updated = await replaceHoldings(portfolio.id, payload);
      toast.success("Holdings updated");
      onUpdated(updated);
    } catch (error: unknown) {
      toast.error(getErrorMessage(error, "Unable to update holdings"));
    } finally {
      setSaving(false);
    }
  }

  const totalWeight = sumWeights(rows);

  return (
    <Card className="rounded-[20px]">
      <CardHeader className="block">
        <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Holdings Editor</div>
        <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Construct portfolio weights</CardTitle>
        <CardDescription>Manage holdings precisely, validate tickers, and save only when the allocation is fully balanced.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <WeightSumBar total={totalWeight} />

        <div className="overflow-hidden rounded-[18px] border border-[var(--border)]">
          <div className="grid grid-cols-[1.4fr_0.8fr_0.8fr] gap-4 border-b border-[var(--border)] bg-[var(--bg-muted)] px-4 py-3 text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--text-faint)]">
            <div>Ticker</div>
            <div>Weight</div>
            <div className="text-right">Actions</div>
          </div>

          <div className="scrollbar-thin max-h-[420px] overflow-auto">
            {rows.map((row, index) => (
              <div
                key={`${portfolio.id}-${index}`}
                className="grid grid-cols-1 gap-3 border-b border-[var(--border)] px-4 py-4 last:border-b-0 md:grid-cols-[1.4fr_0.8fr_0.8fr]"
              >
                <div className="space-y-2">
                  <Input
                    value={row.ticker}
                    onChange={(event) => updateRow(index, { ticker: event.target.value.toUpperCase(), validation: null })}
                    placeholder="AAPL"
                  />
                  {row.validation ? (
                    <Badge tone={row.validation.is_valid ? "good" : "bad"}>
                      {row.validation.is_valid ? `Validated · ${row.validation.rows_returned} rows` : row.validation.error ?? "Invalid ticker"}
                    </Badge>
                  ) : (
                    <span className="text-xs text-[var(--text-faint)]">Validate if you want an early data-quality check.</span>
                  )}
                </div>

                <div className="space-y-2">
                  <Input
                    value={toWeightInput(row.weight)}
                    onChange={(event) => updateRow(index, { weight: Number(event.target.value) || 0 })}
                    inputMode="decimal"
                    placeholder="0.2500"
                  />
                  <span className="text-xs text-[var(--text-faint)]">Use decimal weights that sum to 1.00.</span>
                </div>

                <div className="flex items-start justify-end gap-2">
                  <Button variant="secondary" onClick={() => runValidation(index)} type="button">
                    {row.validating ? <LoaderCircle className="animate-spin" size={16} /> : <SearchCheck size={16} />}
                    Validate
                  </Button>
                  <Button variant="ghost" onClick={() => removeRow(index)} type="button">
                    <Trash2 size={16} />
                    Remove
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:justify-between">
          <Button variant="secondary" onClick={addRow} type="button">
            <Plus size={16} />
            Add Holding
          </Button>
          <Button loading={saving} onClick={save}>
            Save Holdings
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
