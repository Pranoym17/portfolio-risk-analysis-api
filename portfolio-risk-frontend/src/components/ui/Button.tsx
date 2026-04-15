"use client";

import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

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
      "bg-[var(--accent)] text-white shadow-[0_8px_18px_-12px_rgba(21,71,101,0.72)] hover:bg-[var(--accent-strong)]",
    secondary:
      "border border-[var(--border-strong)] bg-[var(--bg-elevated)] text-[var(--text)] hover:bg-[var(--bg-muted)]",
    ghost: "text-[var(--text-soft)] hover:bg-[var(--bg-muted)] hover:text-[var(--text)]",
    danger: "bg-[var(--red)] text-white hover:bg-[#8f4040]",
  } satisfies Record<ButtonVariant, string>;

  return (
    <button
      className={cn(
        "focus-ring inline-flex min-h-10 items-center justify-center gap-2 rounded-[12px] px-4 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
        styles[variant],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/35 border-t-white" /> : null}
      {children}
    </button>
  );
}
