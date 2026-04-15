"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/components/providers/AuthProvider";

export function AuthGate({ children }: { children: React.ReactNode }) {
  const { token, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !token && pathname !== "/login") {
      router.replace("/login");
    }
  }, [loading, pathname, router, token]);

  if (loading) {
    return (
      <div className="panel mx-auto flex min-h-[320px] w-full max-w-3xl items-center justify-center rounded-[24px]">
        <div className="space-y-3 text-center">
          <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-[var(--line)] border-t-[var(--accent)]" />
          <p className="text-sm text-[var(--text-soft)]">Preparing your workspace...</p>
        </div>
      </div>
    );
  }

  if (!token) return null;

  return <>{children}</>;
}
