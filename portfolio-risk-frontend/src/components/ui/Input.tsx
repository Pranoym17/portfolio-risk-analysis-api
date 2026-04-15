import { cn } from "@/lib/utils";

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "focus-ring h-11 w-full rounded-[12px] border border-[var(--border)] bg-[var(--bg-elevated)] px-3.5 text-sm text-[var(--text)] placeholder:text-[var(--text-faint)]",
        className,
      )}
      {...props}
    />
  );
}

export function Select({ className, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "focus-ring h-11 w-full rounded-[12px] border border-[var(--border)] bg-[var(--bg-elevated)] px-3.5 text-sm text-[var(--text)]",
        className,
      )}
      {...props}
    />
  );
}
