"use client"

import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/sonner"
import { AuthProvider } from "./auth-provider"
import { PortfolioProvider } from "./portfolio-provider"

export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem
      disableTransitionOnChange
    >
      <AuthProvider>
        <PortfolioProvider>
          {children}
          <Toaster richColors position="top-right" />
        </PortfolioProvider>
      </AuthProvider>
    </ThemeProvider>
  )
}
