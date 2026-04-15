"use client";

import { useMemo } from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

function titleFromPath(pathname: string) {
  if (pathname.startsWith("/dashboard")) return "Command Center";
  if (pathname.startsWith("/portfolios/create")) return "New Portfolio";
  if (pathname.startsWith("/portfolios/")) return "Portfolio Workspace";
  if (pathname.startsWith("/portfolios")) return "Portfolio Library";
  if (pathname.startsWith("/analytics")) return "Risk Analytics";
  if (pathname.startsWith("/tickers")) return "Ticker Validation";
  if (pathname.startsWith("/settings")) return "Workspace Settings";
  if (pathname.startsWith("/profile")) return "Account Profile";
  return "Axiom Risk";
}

function subtitleFromPath(pathname: string) {
  if (pathname.startsWith("/dashboard")) return "Portfolio-level signals, benchmark context, and holdings diagnostics in one view.";
  if (pathname.startsWith("/portfolios/create")) return "Create a portfolio shell and prepare holdings for analysis.";
  if (pathname.startsWith("/portfolios/")) return "Inspect rolling metrics, allocation, and holdings quality for the selected book.";
  if (pathname.startsWith("/portfolios")) return "Search, compare, and launch into portfolio workspaces.";
  if (pathname.startsWith("/analytics")) return "A deeper quantitative view across risk metrics and concentration structure.";
  if (pathname.startsWith("/tickers")) return "Validate symbols before they enter the risk engine.";
  if (pathname.startsWith("/settings")) return "Preferences, notifications, and workspace defaults.";
  if (pathname.startsWith("/profile")) return "Your account and access details.";
  return "Portfolio intelligence workspace.";
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const header = useMemo(
    () => ({
      title: titleFromPath(pathname),
      subtitle: subtitleFromPath(pathname),
    }),
    [pathname],
  );

  return (
    <div className="page-shell analytics-grid min-h-screen">
      <div className="mx-auto grid min-h-screen max-w-[1680px] gap-4 px-3 py-3 lg:grid-cols-[260px_minmax(0,1fr)] lg:px-4 lg:py-4">
        <Sidebar />
        <div className="min-w-0">
          <Topbar title={header.title} subtitle={header.subtitle} />
          <main className="pt-4 lg:pt-5">{children}</main>
        </div>
      </div>
    </div>
  );
}
