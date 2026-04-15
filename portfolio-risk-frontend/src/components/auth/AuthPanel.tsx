"use client";

import { useState } from "react";
import { useAuth } from "@/components/providers/AuthProvider";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";

export function AuthPanel() {
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    try {
      if (mode === "signin") {
        await signIn(email, password);
      } else {
        await signUp(email, password);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[1.05fr_0.95fr]">
      <section className="flex items-center border-b border-[var(--border)] bg-[var(--bg-elevated)] px-6 py-10 lg:border-b-0 lg:border-r lg:px-14">
        <div className="mx-auto max-w-xl">
          <p className="text-[11px] uppercase tracking-[0.16em] text-[var(--text-faint)]">Premium Financial Interface</p>
          <h1 className="mt-4 text-4xl font-semibold tracking-[-0.04em] text-[var(--text)] sm:text-5xl">
            Portfolio risk analytics with institutional calm.
          </h1>
          <p className="mt-6 max-w-lg text-base leading-8 text-[var(--text-soft)]">
            Build user-owned portfolios, validate holdings, and review volatility, drawdown, beta, rolling metrics,
            and concentration drivers in one deliberate workspace.
          </p>
          <div className="mt-10 grid gap-4 sm:grid-cols-3">
            {[
              ["Sharpe and drawdown", "Fast KPI review for return, volatility, downside, and benchmark sensitivity."],
              ["Attribution clarity", "See asset and sector contribution instead of a single headline score."],
              ["Analyst-friendly editing", "Precise holdings management with validation and weight discipline."],
            ].map(([title, copy]) => (
              <div key={title} className="rounded-[18px] border border-[var(--border)] bg-[var(--bg-muted)] p-4">
                <div className="text-sm font-semibold text-[var(--text)]">{title}</div>
                <div className="mt-2 text-sm leading-6 text-[var(--text-soft)]">{copy}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="flex items-center justify-center px-6 py-10 lg:px-10">
        <Card className="w-full max-w-md">
          <CardHeader className="block">
            <div className="text-[11px] uppercase tracking-[0.16em] text-[var(--text-faint)]">Secure Access</div>
            <CardTitle className="mt-2 text-2xl tracking-[-0.03em]">
              {mode === "signin" ? "Sign in to your workspace" : "Create your analyst account"}
            </CardTitle>
            <CardDescription>
              Use the backend auth flow you just added. Portfolios and analytics remain scoped to the signed-in user.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 rounded-[14px] bg-[var(--bg-muted)] p-1">
              <button
                className={`rounded-[12px] px-3 py-2 text-sm font-medium transition ${
                  mode === "signin" ? "bg-[var(--bg-elevated)] text-[var(--text)] shadow-[var(--shadow-sm)]" : "text-[var(--text-soft)]"
                }`}
                onClick={() => setMode("signin")}
                type="button"
              >
                Sign In
              </button>
              <button
                className={`rounded-[12px] px-3 py-2 text-sm font-medium transition ${
                  mode === "signup" ? "bg-[var(--bg-elevated)] text-[var(--text)] shadow-[var(--shadow-sm)]" : "text-[var(--text-soft)]"
                }`}
                onClick={() => setMode("signup")}
                type="button"
              >
                Sign Up
              </button>
            </div>

            <label className="block space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Email</span>
              <Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="analyst@firm.com" type="email" />
            </label>

            <label className="block space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Password</span>
              <Input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 8 characters"
                type="password"
              />
            </label>

            <Button className="w-full" loading={loading} onClick={submit}>
              {mode === "signin" ? "Enter Workspace" : "Create Account"}
            </Button>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
