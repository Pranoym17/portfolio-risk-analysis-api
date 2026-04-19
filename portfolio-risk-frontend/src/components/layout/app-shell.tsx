"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { AppSidebar } from "./app-sidebar"
import { AppTopbar } from "./app-topbar"
import { Sheet, SheetContent } from "@/components/ui/sheet"

interface AppShellProps {
  children: React.ReactNode
  title?: string
  subtitle?: string
  actions?: React.ReactNode
  className?: string
}

export function AppShell({ 
  children, 
  title, 
  subtitle, 
  actions,
  className 
}: AppShellProps) {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="min-h-screen bg-background flex">
      {/* Desktop sidebar */}
      <div className="hidden lg:block">
        <div className="fixed inset-y-0 left-0 z-50">
          <AppSidebar />
        </div>
      </div>

      {/* Mobile sidebar */}
      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetContent side="left" className="p-0 w-64">
          <AppSidebar />
        </SheetContent>
      </Sheet>

      {/* Main content area */}
      <div className="flex-1 lg:pl-64 flex flex-col min-h-screen">
        <AppTopbar 
          title={title} 
          subtitle={subtitle} 
          actions={actions}
          onMenuClick={() => setMobileOpen(true)} 
        />
        
        <main className={cn("flex-1 p-6", className)}>
          {children}
        </main>
      </div>
    </div>
  )
}
