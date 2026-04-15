import { Button } from "./Button";

export function EmptyState({
  title,
  body,
  actionLabel,
  onAction,
}: {
  title: string;
  body: string;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="surface-grid rounded-[18px] border border-dashed border-[var(--border-strong)] px-6 py-10 text-center">
      <div className="mx-auto max-w-md space-y-2">
        <p className="text-base font-semibold text-[var(--text)]">{title}</p>
        <p className="text-sm leading-6 text-[var(--text-soft)]">{body}</p>
        {actionLabel && onAction ? (
          <div className="pt-2">
            <Button variant="secondary" onClick={onAction}>
              {actionLabel}
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function ErrorState({
  title,
  body,
  onRetry,
}: {
  title: string;
  body: string;
  onRetry?: () => void;
}) {
  return (
    <div className="rounded-[18px] border border-[#e4bbbb] bg-[var(--red-soft)] px-5 py-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-[var(--red)]">{title}</p>
          <p className="mt-1 text-sm text-[#8a5151]">{body}</p>
        </div>
        {onRetry ? (
          <Button variant="secondary" onClick={onRetry}>
            Retry
          </Button>
        ) : null}
      </div>
    </div>
  );
}

export function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-[14px] bg-[var(--bg-subtle)] ${className}`} />;
}
