"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"
import {
  createPortfolio as createPortfolioRequest,
  deletePortfolio as deletePortfolioRequest,
  listPortfolios,
  replaceHoldings as replaceHoldingsRequest,
} from "@/lib/api"
import type { HoldingIn, PortfolioOut } from "@/lib/types"
import { useAuth } from "./auth-provider"

const ACTIVE_PORTFOLIO_STORAGE_KEY = "portfolio-risk-active-portfolio"

type PortfolioContextValue = {
  portfolios: PortfolioOut[]
  activePortfolioId: number | null
  activePortfolio: PortfolioOut | null
  isLoading: boolean
  refreshPortfolios: () => Promise<void>
  selectPortfolio: (portfolioId: number | null) => void
  createPortfolio: (name: string) => Promise<PortfolioOut>
  deletePortfolio: (portfolioId: number) => Promise<void>
  replaceHoldings: (portfolioId: number, holdings: HoldingIn[]) => Promise<PortfolioOut>
}

const PortfolioContext = createContext<PortfolioContextValue | null>(null)

function sortPortfolios(portfolios: PortfolioOut[]) {
  return [...portfolios].sort((a, b) => a.name.localeCompare(b.name))
}

export function PortfolioProvider({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading: authLoading } = useAuth()
  const [portfolios, setPortfolios] = useState<PortfolioOut[]>([])
  const [activePortfolioId, setActivePortfolioId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const applyActiveSelection = useCallback((items: PortfolioOut[], requestedId?: number | null) => {
    if (items.length === 0) {
      setActivePortfolioId(null)
      if (typeof window !== "undefined") {
        window.localStorage.removeItem(ACTIVE_PORTFOLIO_STORAGE_KEY)
      }
      return
    }

    const storedId =
      requestedId ??
      (typeof window !== "undefined"
        ? Number(window.localStorage.getItem(ACTIVE_PORTFOLIO_STORAGE_KEY))
        : NaN)

    const nextActive = items.find((portfolio) => portfolio.id === storedId) ?? items[0]
    setActivePortfolioId(nextActive.id)
    if (typeof window !== "undefined") {
      window.localStorage.setItem(ACTIVE_PORTFOLIO_STORAGE_KEY, String(nextActive.id))
    }
  }, [])

  const refreshPortfolios = useCallback(async () => {
    if (!isAuthenticated) {
      setPortfolios([])
      setActivePortfolioId(null)
      setIsLoading(false)
      return
    }

    setIsLoading(true)
    try {
      const nextPortfolios = sortPortfolios(await listPortfolios())
      setPortfolios(nextPortfolios)
      applyActiveSelection(nextPortfolios)
    } finally {
      setIsLoading(false)
    }
  }, [applyActiveSelection, isAuthenticated])

  useEffect(() => {
    if (authLoading) return
    void refreshPortfolios()
  }, [authLoading, refreshPortfolios])

  const selectPortfolio = useCallback((portfolioId: number | null) => {
    setActivePortfolioId(portfolioId)
    if (typeof window === "undefined") return
    if (portfolioId === null) {
      window.localStorage.removeItem(ACTIVE_PORTFOLIO_STORAGE_KEY)
      return
    }
    window.localStorage.setItem(ACTIVE_PORTFOLIO_STORAGE_KEY, String(portfolioId))
  }, [])

  const createPortfolio = useCallback(async (name: string) => {
    const created = await createPortfolioRequest(name)
    const nextPortfolios = sortPortfolios([...portfolios, created])
    setPortfolios(nextPortfolios)
    applyActiveSelection(nextPortfolios, created.id)
    return created
  }, [applyActiveSelection, portfolios])

  const deletePortfolio = useCallback(async (portfolioId: number) => {
    await deletePortfolioRequest(portfolioId)
    const nextPortfolios = portfolios.filter((portfolio) => portfolio.id !== portfolioId)
    setPortfolios(nextPortfolios)
    applyActiveSelection(nextPortfolios)
  }, [applyActiveSelection, portfolios])

  const replaceHoldings = useCallback(async (portfolioId: number, holdings: HoldingIn[]) => {
    const updated = await replaceHoldingsRequest(portfolioId, holdings)
    const nextPortfolios = sortPortfolios(
      portfolios.map((portfolio) => portfolio.id === portfolioId ? updated : portfolio),
    )
    setPortfolios(nextPortfolios)
    applyActiveSelection(nextPortfolios, portfolioId)
    return updated
  }, [applyActiveSelection, portfolios])

  const activePortfolio = useMemo(
    () => portfolios.find((portfolio) => portfolio.id === activePortfolioId) ?? null,
    [activePortfolioId, portfolios],
  )

  const value = useMemo<PortfolioContextValue>(() => ({
    portfolios,
    activePortfolioId,
    activePortfolio,
    isLoading,
    refreshPortfolios,
    selectPortfolio,
    createPortfolio,
    deletePortfolio,
    replaceHoldings,
  }), [
    portfolios,
    activePortfolioId,
    activePortfolio,
    isLoading,
    refreshPortfolios,
    selectPortfolio,
    createPortfolio,
    deletePortfolio,
    replaceHoldings,
  ])

  return <PortfolioContext.Provider value={value}>{children}</PortfolioContext.Provider>
}

export function usePortfolioData() {
  const context = useContext(PortfolioContext)
  if (!context) {
    throw new Error("usePortfolioData must be used within a PortfolioProvider")
  }
  return context
}
