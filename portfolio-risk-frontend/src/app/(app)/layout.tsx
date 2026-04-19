import { AppShell } from "@/components/layout/app-shell"
import { AppAuthGuard } from "@/components/auth/app-auth-guard"

export default function AppLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <AppAuthGuard>
      <AppShell>{children}</AppShell>
    </AppAuthGuard>
  )
}
