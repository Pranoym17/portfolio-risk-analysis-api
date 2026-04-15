"use client";

import { useState } from "react";
import { SearchCheck, XCircle } from "lucide-react";
import { validateTicker } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import type { TickerValidationResult } from "@/lib/types";
import { EmptyState, ErrorState } from "@/components/ui/StatePanel";
import { getErrorMessage } from "@/lib/utils";

export default function TickersPage() {
  const [ticker, setTicker] = useState("");
  const [result, setResult] = useState<TickerValidationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleValidate() {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await validateTicker(ticker.trim().toUpperCase()));
    } catch (nextError) {
      setError(getErrorMessage(nextError, "Unable to validate ticker."));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
      <section className="panel hero-panel rounded-[30px] p-6">
        <div className="eyebrow text-[var(--accent)]">Ticker Validation</div>
        <h2 className="mt-3 text-4xl font-semibold tracking-[-0.06em]">Check a symbol before it reaches the portfolio workflow.</h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--text-soft)]">
          Use validation to confirm whether market data is available, catch obvious ticker issues early, and keep the risk engine cleaner.
        </p>
      </section>

      <section className="panel rounded-[30px] p-6">
        <div className="eyebrow text-[var(--text-faint)]">Validate Symbol</div>
        <div className="mt-6 flex flex-col gap-3 sm:flex-row">
          <Input value={ticker} onChange={(event) => setTicker(event.target.value.toUpperCase())} placeholder="AAPL" />
          <Button onClick={handleValidate} loading={loading}>
            <SearchCheck size={16} />
            Validate
          </Button>
        </div>

        <div className="mt-6">
          {error ? <ErrorState title="Validation failed" body={error} /> : null}
          {!error && !result ? (
            <EmptyState title="No ticker checked yet" body="Enter a symbol to see whether price history is available for analytics." />
          ) : null}
          {result ? (
            <div className={`rounded-[22px] border p-5 ${result.is_valid ? "border-[rgba(88,199,152,0.18)] bg-[rgba(88,199,152,0.08)]" : "border-[rgba(255,143,152,0.18)] bg-[rgba(255,143,152,0.08)]"}`}>
              <div className="flex items-center gap-2">
                {result.is_valid ? <SearchCheck size={18} className="text-[var(--success)]" /> : <XCircle size={18} className="text-[var(--danger)]" />}
                <p className="text-sm font-semibold text-[var(--text)]">{result.ticker}</p>
              </div>
              <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">
                {result.is_valid
                  ? `Validation passed with ${result.rows_returned} rows returned for the requested period and interval.`
                  : result.error ?? "Ticker validation failed."}
              </p>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
