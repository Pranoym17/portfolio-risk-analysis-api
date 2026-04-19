"use client"

import { useState } from "react"
import { 
  Search,
  Download,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  Calendar,
  Receipt
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

// Mock transaction data
const transactions = [
  {
    id: "1",
    type: "buy",
    ticker: "AAPL",
    name: "Apple Inc.",
    shares: 25,
    price: 175.50,
    total: 4387.50,
    date: "2026-04-14",
    time: "09:32:15",
    portfolio: "Growth Portfolio",
    status: "completed",
  },
  {
    id: "2",
    type: "sell",
    ticker: "GOOGL",
    name: "Alphabet Inc.",
    shares: 10,
    price: 141.20,
    total: 1412.00,
    date: "2026-04-13",
    time: "14:45:22",
    portfolio: "Growth Portfolio",
    status: "completed",
  },
  {
    id: "3",
    type: "dividend",
    ticker: "JNJ",
    name: "Johnson & Johnson",
    shares: 200,
    price: 1.19,
    total: 238.00,
    date: "2026-04-12",
    time: "00:00:00",
    portfolio: "Dividend Income",
    status: "completed",
  },
  {
    id: "4",
    type: "buy",
    ticker: "NVDA",
    name: "NVIDIA Corporation",
    shares: 5,
    price: 872.50,
    total: 4362.50,
    date: "2026-04-11",
    time: "10:15:33",
    portfolio: "Growth Portfolio",
    status: "completed",
  },
  {
    id: "5",
    type: "transfer",
    ticker: "VTI",
    name: "Vanguard Total Stock Market ETF",
    shares: 50,
    price: 224.89,
    total: 11244.50,
    date: "2026-04-10",
    time: "16:00:00",
    portfolio: "Balanced ETF",
    status: "completed",
  },
  {
    id: "6",
    type: "buy",
    ticker: "BTC",
    name: "Bitcoin",
    shares: 0.15,
    price: 67500,
    total: 10125.00,
    date: "2026-04-09",
    time: "22:14:08",
    portfolio: "Speculative",
    status: "completed",
  },
  {
    id: "7",
    type: "sell",
    ticker: "MSFT",
    name: "Microsoft Corporation",
    shares: 15,
    price: 378.00,
    total: 5670.00,
    date: "2026-04-08",
    time: "11:30:45",
    portfolio: "Growth Portfolio",
    status: "completed",
  },
  {
    id: "8",
    type: "dividend",
    ticker: "O",
    name: "Realty Income Corporation",
    shares: 400,
    price: 0.256,
    total: 102.40,
    date: "2026-04-05",
    time: "00:00:00",
    portfolio: "Dividend Income",
    status: "completed",
  },
]

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  })
}

function getTypeIcon(type: string) {
  switch (type) {
    case "buy":
      return <ArrowDownRight className="h-4 w-4" />
    case "sell":
      return <ArrowUpRight className="h-4 w-4" />
    case "dividend":
      return <Receipt className="h-4 w-4" />
    case "transfer":
      return <RefreshCw className="h-4 w-4" />
    default:
      return null
  }
}

function getTypeColor(type: string) {
  switch (type) {
    case "buy":
      return "bg-primary/10 text-primary"
    case "sell":
      return "bg-negative/10 text-negative"
    case "dividend":
      return "bg-positive/10 text-positive"
    case "transfer":
      return "bg-chart-3/10 text-chart-3"
    default:
      return "bg-muted text-muted-foreground"
  }
}

export default function TransactionsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [typeFilter, setTypeFilter] = useState("all")
  const [portfolioFilter, setPortfolioFilter] = useState("all")

  const filteredTransactions = transactions.filter(t => {
    const matchesSearch = 
      t.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesType = typeFilter === "all" || t.type === typeFilter
    const matchesPortfolio = portfolioFilter === "all" || t.portfolio === portfolioFilter
    return matchesSearch && matchesType && matchesPortfolio
  })

  const portfolios = [...new Set(transactions.map(t => t.portfolio))]

  // Calculate summary stats
  const totalBuys = transactions.filter(t => t.type === "buy").reduce((sum, t) => sum + t.total, 0)
  const totalSells = transactions.filter(t => t.type === "sell").reduce((sum, t) => sum + t.total, 0)
  const totalDividends = transactions.filter(t => t.type === "dividend").reduce((sum, t) => sum + t.total, 0)

  return (
    <div className="min-h-screen">
      {/* Page Header */}
      <div className="border-b border-border/60 bg-card/50">
        <div className="px-6 lg:px-8 py-6">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <Receipt className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight">Transactions</h1>
                  <p className="text-sm text-muted-foreground">
                    View and manage your transaction history
                  </p>
                </div>
              </div>
            </div>
            
            {/* Summary Stats */}
            <div className="flex items-center gap-6">
              <div className="text-right">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Buys
                </p>
                <p className="text-lg font-semibold tabular-nums text-primary">
                  {formatCurrency(totalBuys)}
                </p>
              </div>
              <div className="h-8 w-px bg-border" />
              <div className="text-right">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Sells
                </p>
                <p className="text-lg font-semibold tabular-nums text-negative">
                  {formatCurrency(totalSells)}
                </p>
              </div>
              <div className="h-8 w-px bg-border" />
              <div className="text-right">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Dividends
                </p>
                <p className="text-lg font-semibold tabular-nums text-positive">
                  {formatCurrency(totalDividends)}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="border-b border-border/40 bg-surface-1/50">
        <div className="px-6 lg:px-8 py-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search transactions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-64 pl-9 h-9 bg-background"
                />
              </div>
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="w-36 h-9">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="buy">Buys</SelectItem>
                  <SelectItem value="sell">Sells</SelectItem>
                  <SelectItem value="dividend">Dividends</SelectItem>
                  <SelectItem value="transfer">Transfers</SelectItem>
                </SelectContent>
              </Select>
              <Select value={portfolioFilter} onValueChange={setPortfolioFilter}>
                <SelectTrigger className="w-44 h-9">
                  <SelectValue placeholder="Portfolio" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Portfolios</SelectItem>
                  {portfolios.map(p => (
                    <SelectItem key={p} value={p}>{p}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" className="h-9">
                <Calendar className="h-4 w-4 mr-2" />
                Date Range
              </Button>
            </div>
            
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" className="h-9">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Transactions List */}
      <div className="px-6 lg:px-8 py-6">
        <div className="bg-card border border-border/60 rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border bg-surface-2/50">
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Asset
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Portfolio
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Shares
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Price
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Total
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Date
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {filteredTransactions.map((transaction) => (
                  <tr 
                    key={transaction.id}
                    className="transition-colors hover:bg-surface-1"
                  >
                    <td className="px-4 py-4">
                      <div className={cn(
                        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium capitalize",
                        getTypeColor(transaction.type)
                      )}>
                        {getTypeIcon(transaction.type)}
                        {transaction.type}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-surface-2 font-mono text-xs font-semibold">
                          {transaction.ticker.slice(0, 2)}
                        </div>
                        <div>
                          <p className="font-medium text-sm">{transaction.ticker}</p>
                          <p className="text-xs text-muted-foreground">{transaction.name}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <span className="text-sm text-muted-foreground">{transaction.portfolio}</span>
                    </td>
                    <td className="px-4 py-4 text-right tabular-nums text-sm">
                      {transaction.shares.toLocaleString()}
                    </td>
                    <td className="px-4 py-4 text-right tabular-nums text-sm">
                      {formatCurrency(transaction.price)}
                    </td>
                    <td className="px-4 py-4 text-right font-medium tabular-nums">
                      <span className={cn(
                        transaction.type === "sell" || transaction.type === "dividend" 
                          ? "text-positive" 
                          : ""
                      )}>
                        {transaction.type === "sell" || transaction.type === "dividend" ? "+" : "-"}
                        {formatCurrency(transaction.total)}
                      </span>
                    </td>
                    <td className="px-4 py-4 text-right">
                      <div>
                        <p className="text-sm tabular-nums">{formatDate(transaction.date)}</p>
                        <p className="text-xs text-muted-foreground tabular-nums">{transaction.time}</p>
                      </div>
                    </td>
                    <td className="px-4 py-4 text-center">
                      <Badge variant="secondary" className="bg-positive/10 text-positive text-xs">
                        {transaction.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Table Footer */}
          <div className="border-t border-border bg-surface-1/50 px-4 py-3 flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Showing {filteredTransactions.length} of {transactions.length} transactions
            </p>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled>
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled>
                Next
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
