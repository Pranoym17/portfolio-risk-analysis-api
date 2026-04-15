export function LoadingState({
  lines = 4,
  className = "",
}: {
  lines?: number;
  className?: string;
}) {
  return (
    <div className={`overflow-hidden rounded-[22px] border border-[var(--border)] bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(246,250,252,0.98))] p-5 ${className}`}>
      <div className="space-y-3">
        {Array.from({ length: lines }).map((_, index) => (
          <div
            key={index}
            className="h-4 animate-pulse rounded-full bg-[linear-gradient(90deg,rgba(216,226,233,0.5),rgba(231,238,243,0.95),rgba(216,226,233,0.5))]"
          />
        ))}
      </div>
    </div>
  );
}
