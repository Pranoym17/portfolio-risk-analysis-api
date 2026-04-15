import { cn } from "@/lib/utils";

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "focus-ring h-11 w-full rounded-[14px] border border-[var(--line)] bg-[rgba(255,255,255,0.04)] px-3.5 text-sm text-[var(--text)] placeholder:text-[var(--text-faint)]",
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
        "focus-ring h-11 w-full rounded-[14px] border border-[var(--line)] bg-[rgba(255,255,255,0.04)] px-3.5 text-sm text-[var(--text)]",
        className,
      )}
      {...props}
    />
  );
}
