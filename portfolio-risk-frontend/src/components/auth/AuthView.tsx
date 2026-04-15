"use client";

import Link from "next/link";
import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, LockKeyhole, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useAuth } from "@/components/providers/AuthProvider";

type Mode = "login" | "signup" | "forgot" | "reset";

const copy = {
  login: {
    eyebrow: "Secure Access",
    title: "Sign in to the analytics workspace",
    subtitle: "Review portfolio diagnostics, holdings edits, and benchmark-aware risk metrics from one terminal.",
    cta: "Sign In",
  },
  signup: {
    eyebrow: "Create Account",
    title: "Open your portfolio intelligence workspace",
    subtitle: "Start building portfolios, validating holdings, and reviewing rolling analytics with production-ready workflows.",
    cta: "Create Account",
  },
  forgot: {
    eyebrow: "Account Recovery",
    title: "Request a password reset link",
    subtitle: "We’ll help you regain access to your portfolio workspace and restore your analysis flow.",
    cta: "Send Reset Link",
  },
  reset: {
    eyebrow: "Reset Password",
    title: "Set a new workspace password",
    subtitle: "Choose a new password so you can return to your portfolio analytics environment.",
    cta: "Update Password",
  },
} satisfies Record<Mode, { eyebrow: string; title: string; subtitle: string; cta: string }>;

export function AuthView({ mode }: { mode: Mode }) {
  const { signIn, signUp } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);

    try {
      if (mode === "login") {
        await signIn(email, password);
        return;
      }

      if (mode === "signup") {
        if (password.length < 8) {
          throw new Error("Use at least 8 characters for the account password.");
        }
        await signUp(email, password);
        return;
      }

      if (mode === "forgot") {
        toast.success("Reset instructions UI completed. Wire the backend email flow when ready.");
        router.push("/login");
        return;
      }

      if (password !== confirm) {
        throw new Error("Passwords do not match.");
      }
      toast.success("Password reset UI completed. Connect the reset token flow when the backend endpoint is ready.");
      router.push("/login");
    } catch (error) {
      if (error instanceof Error) {
        toast.error(error.message);
      }
    } finally {
      setLoading(false);
    }
  }

  const active = copy[mode];

  return (
    <div className="page-shell flex min-h-screen items-center justify-center px-4 py-8">
      <div className="grid w-full max-w-[1360px] gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <motion.section
          initial={{ opacity: 0, x: -18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.55 }}
          className="panel hero-panel hidden rounded-[30px] p-8 lg:flex lg:flex-col lg:justify-between"
        >
          <div>
            <div className="eyebrow text-[var(--text-faint)]">Axiom Risk Platform</div>
            <h1 className="mt-4 max-w-lg text-5xl font-semibold tracking-[-0.06em]">Portfolio analytics with stronger market context.</h1>
            <p className="mt-4 max-w-xl text-base leading-8 text-[var(--text-soft)]">
              Build portfolios, validate holdings, inspect risk signals, and compare benchmark sensitivity in a workspace designed for disciplined review.
            </p>
          </div>

          <div className="grid gap-4">
            {[
              ["Risk analytics", "Rolling volatility, Sharpe, beta, VaR, and drawdown in one environment."],
              ["Benchmark context", "Keep benchmark selection and portfolio sensitivity visible across workflows."],
              ["Holdings discipline", "Manage weights, validate tickers, and review dropped positions before analysis."],
            ].map(([title, body]) => (
              <div key={title} className="glass-strip rounded-[22px] p-4">
                <p className="text-sm font-semibold">{title}</p>
                <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">{body}</p>
              </div>
            ))}
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, x: 18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.55, delay: 0.06 }}
          className="panel rounded-[30px] p-6 sm:p-8"
        >
          <div className="flex items-center justify-between">
            <Link href="/" className="inline-flex items-center gap-2 text-sm text-[var(--text-soft)] transition hover:text-[var(--text)]">
              <ArrowRight size={14} className="rotate-180" />
              Back to site
            </Link>
            <div className="glass-strip inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs text-[var(--text-soft)]">
              <ShieldCheck size={13} />
              Secure product preview
            </div>
          </div>

          <div className="mt-10 max-w-xl">
            <div className="eyebrow text-[var(--accent)]">{active.eyebrow}</div>
            <h2 className="mt-3 text-4xl font-semibold tracking-[-0.06em]">{active.title}</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--text-soft)]">{active.subtitle}</p>
          </div>

          <form onSubmit={handleSubmit} className="mt-10 space-y-5">
            <label className="block space-y-2">
              <span className="text-sm font-medium text-[var(--text)]">Email</span>
              <Input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="analyst@firm.com" />
            </label>

            {mode !== "forgot" ? (
              <label className="block space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Password</span>
                <Input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Enter password" />
              </label>
            ) : null}

            {mode === "reset" ? (
              <label className="block space-y-2">
                <span className="text-sm font-medium text-[var(--text)]">Confirm Password</span>
                <Input type="password" value={confirm} onChange={(event) => setConfirm(event.target.value)} placeholder="Confirm password" />
              </label>
            ) : null}

            <Button type="submit" loading={loading} className="w-full">
              <LockKeyhole size={16} />
              {active.cta}
            </Button>
          </form>

          <div className="mt-6 flex flex-wrap gap-4 text-sm text-[var(--text-soft)]">
            {mode !== "login" ? <Link href="/login" className="transition hover:text-[var(--text)]">Already have an account?</Link> : null}
            {mode !== "signup" ? <Link href="/signup" className="transition hover:text-[var(--text)]">Create account</Link> : null}
            {mode !== "forgot" ? <Link href="/forgot-password" className="transition hover:text-[var(--text)]">Forgot password</Link> : null}
          </div>
        </motion.section>
      </div>
    </div>
  );
}
