"use client";

import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "quiet";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  loading?: boolean;
};

export function Button({
  className,
  variant = "primary",
  loading = false,
  disabled,
  children,
  ...props
}: Props) {
  const styles = {
    primary:
      "border border-[rgba(149,177,255,0.24)] bg-[linear-gradient(135deg,rgba(123,162,255,0.92),rgba(87,136,255,0.92))] text-[#09111d] shadow-[0_18px_40px_-20px_rgba(123,162,255,0.75)] hover:brightness-110",
    secondary:
      "border border-[var(--line-strong)] bg-[rgba(255,255,255,0.05)] text-[var(--text)] hover:bg-[rgba(255,255,255,0.08)]",
    ghost: "text-[var(--text-soft)] hover:bg-[rgba(255,255,255,0.05)] hover:text-[var(--text)]",
    danger: "border border-[rgba(255,143,152,0.22)] bg-[rgba(255,143,152,0.12)] text-[var(--text)] hover:bg-[rgba(255,143,152,0.18)]",
    quiet: "border border-transparent bg-transparent text-[var(--text-faint)] hover:border-[var(--line)] hover:bg-[rgba(255,255,255,0.03)] hover:text-[var(--text)]",
  } satisfies Record<ButtonVariant, string>;

  return (
    <button
      className={cn(
        "focus-ring inline-flex min-h-11 items-center justify-center gap-2 rounded-[14px] px-4 text-sm font-semibold transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-60",
        styles[variant],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-current/30 border-t-current" />
      ) : null}
      {children}
    </button>
  );
}
