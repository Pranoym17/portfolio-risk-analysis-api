import { cn } from "@/lib/utils";

export function Panel({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  return <section className={cn("panel rounded-[24px]", className)}>{children}</section>;
}

export function PanelHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="relative flex flex-col gap-4 border-b border-[var(--line)] px-5 py-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        {eyebrow ? <div className="eyebrow text-[var(--text-faint)]">{eyebrow}</div> : null}
        <h3 className="mt-1 text-[20px] font-semibold tracking-[-0.04em] text-[var(--text)]">{title}</h3>
        {description ? <p className="mt-1 max-w-2xl text-sm leading-6 text-[var(--text-soft)]">{description}</p> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}
