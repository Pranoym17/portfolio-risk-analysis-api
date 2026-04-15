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
    <div className="panel panel-muted rounded-[24px] border-dashed px-6 py-12 text-center">
      <div className="mx-auto max-w-md space-y-2">
        <p className="text-lg font-semibold tracking-[-0.02em] text-[var(--text)]">{title}</p>
        <p className="text-sm leading-7 text-[var(--text-soft)]">{body}</p>
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
    <div className="rounded-[20px] border border-[rgba(255,143,152,0.24)] bg-[linear-gradient(180deg,rgba(255,143,152,0.12),rgba(255,143,152,0.05))] px-5 py-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold tracking-[0.01em] text-[var(--danger)]">{title}</p>
          <p className="mt-1 text-sm leading-7 text-[var(--text-soft)]">{body}</p>
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
  return <div className={`animate-pulse rounded-[14px] bg-[rgba(255,255,255,0.06)] ${className}`} />;
}

export function LoadingState({ title = "Loading workspace..." }: { title?: string }) {
  return (
    <div className="panel rounded-[24px] px-6 py-16 text-center">
      <div className="mx-auto flex max-w-sm flex-col items-center gap-4">
        <div className="h-12 w-12 animate-spin rounded-full border-2 border-[var(--line)] border-t-[var(--accent)]" />
        <div>
          <p className="text-base font-semibold text-[var(--text)]">{title}</p>
          <p className="mt-1 text-sm text-[var(--text-soft)]">Preparing charts, holdings, and benchmark context.</p>
        </div>
      </div>
    </div>
  );
}
