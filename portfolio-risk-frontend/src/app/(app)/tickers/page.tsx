"use client"

import { useState } from "react"
import { Search, ShieldCheck } from "lucide-react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export default function TickersPage() {
  const router = useRouter()
  const [symbol, setSymbol] = useState("")

  return (
    <div className="min-h-[70vh] flex items-center justify-center">
      <div className="w-full max-w-2xl rounded-2xl border border-border/60 bg-card p-8">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
            <ShieldCheck className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold text-foreground">Ticker Validation</h1>
            <p className="text-sm text-muted-foreground">
              Check whether your backend can retrieve pricing history for a ticker before adding it to a portfolio.
            </p>
          </div>
        </div>

        <div className="mt-8 flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="Enter ticker, e.g. AAPL"
              className="h-12 pl-9 bg-surface-1"
            />
          </div>
          <Button
            className="h-12 px-5"
            onClick={() => router.push(`/ticker/${encodeURIComponent(symbol.trim() || "AAPL")}`)}
          >
            Validate
          </Button>
        </div>
      </div>
    </div>
  )
}
