"use client";

import { useState } from "react";
import { LockKeyhole, ShieldCheck, TrendingUp } from "lucide-react";
import { useAuth } from "@/components/providers/AuthProvider";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Reveal } from "@/components/ui/Reveal";

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
    <div className="grid min-h-screen grid-cols-1 bg-[linear-gradient(135deg,#f7fbfd_0%,#eef3f6_52%,#e8f0f4_100%)] lg:grid-cols-[1.12fr_0.88fr]">
      <section className="flex items-center px-6 py-10 lg:px-14">
        <Reveal className="mx-auto max-w-2xl" delay={0.02}>
          <div className="panel-dark rounded-[30px] p-8 lg:p-10">
            <div className="section-kicker text-sky-300">Secure Analyst Access</div>
            <h1 className="mt-5 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
              A sharper frontend for portfolio construction and risk intelligence.
            </h1>
            <p className="mt-5 max-w-xl text-base leading-8 text-slate-300">
              Sign in to a user-scoped workspace built for holdings precision, benchmark-aware analytics, and premium
              data review rather than generic dashboard fluff.
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              {[
                { Icon: ShieldCheck, title: "Scoped portfolios", copy: "Each account sees only its own portfolios and risk surfaces." },
                { Icon: TrendingUp, title: "Live analytics", copy: "Refresh rolling metrics, attribution, and covariance from one cockpit." },
                { Icon: LockKeyhole, title: "Serious workflow", copy: "A tighter fintech UI built for users, not demos." },
              ].map(({ Icon, title, copy }) => (
                <div key={title} className="rounded-[20px] border border-white/10 bg-white/6 p-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-[14px] bg-white/10 text-white">
                    <Icon size={18} />
                  </div>
                  <div className="mt-4 text-sm font-semibold text-white">{title}</div>
                  <div className="mt-2 text-sm leading-6 text-slate-300">{copy}</div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </section>

      <section className="flex items-center justify-center px-6 py-10 lg:px-10">
        <Reveal delay={0.08}>
        <Card className="w-full max-w-md rounded-[28px]">
          <CardHeader className="block">
            <div className="section-kicker text-[var(--accent)]">Secure Access</div>
            <CardTitle className="mt-2 text-3xl tracking-[-0.04em]">
              {mode === "signin" ? "Enter the workspace" : "Create your account"}
            </CardTitle>
            <CardDescription>
              Use the authentication layer from your backend and step directly into the portfolio risk terminal.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="grid grid-cols-2 rounded-[16px] bg-[var(--bg-muted)] p-1">
              <button
                className={`rounded-[14px] px-3 py-3 text-sm font-semibold transition ${
                  mode === "signin" ? "bg-white text-[var(--text)] shadow-[var(--shadow-sm)]" : "text-[var(--text-soft)]"
                }`}
                onClick={() => setMode("signin")}
                type="button"
              >
                Sign In
              </button>
              <button
                className={`rounded-[14px] px-3 py-3 text-sm font-semibold transition ${
                  mode === "signup" ? "bg-white text-[var(--text)] shadow-[var(--shadow-sm)]" : "text-[var(--text-soft)]"
                }`}
                onClick={() => setMode("signup")}
                type="button"
              >
                Sign Up
              </button>
            </div>

            <label className="block space-y-2">
              <span className="text-sm font-semibold text-[var(--text)]">Email</span>
              <Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="analyst@firm.com" type="email" />
            </label>

            <label className="block space-y-2">
              <span className="text-sm font-semibold text-[var(--text)]">Password</span>
              <Input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 8 characters" type="password" />
            </label>

            <Button className="w-full" loading={loading} onClick={submit}>
              {mode === "signin" ? "Access Portfolio Terminal" : "Create Analyst Account"}
            </Button>
          </CardContent>
        </Card>
        </Reveal>
      </section>
    </div>
  );
}
