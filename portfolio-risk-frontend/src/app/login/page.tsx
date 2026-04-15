"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { AuthPanel } from "@/components/auth/AuthPanel";
import { useAuth } from "@/components/providers/AuthProvider";

export default function LoginPage() {
  const { token, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && token) {
      router.replace("/portfolios");
    }
  }, [loading, router, token]);

  return <AuthPanel />;
}
