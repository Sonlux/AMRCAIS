import { SIGNAL_COLORS } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface SignalCardProps {
  module: string;
  signal: string;
  strength: number;
  explanation?: string;
  className?: string;
}

export default function SignalCard({
  module,
  signal,
  strength,
  explanation,
  className,
}: SignalCardProps) {
  const color = SIGNAL_COLORS[signal] ?? "#6b7280";

  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-surface p-4 transition-colors hover:border-border-light",
        className,
      )}
    >
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wider text-text-muted">
          {module}
        </p>
        <span
          className="rounded px-2 py-0.5 text-xs font-semibold"
          style={{
            backgroundColor: `${color}18`,
            color,
          }}
        >
          {signal}
        </span>
      </div>

      {/* Strength bar */}
      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between">
          <span className="text-xs text-text-secondary">Strength</span>
          <span className="font-mono text-xs text-text-secondary">
            {(strength * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-elevated">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${Math.min(strength * 100, 100)}%`,
              backgroundColor: color,
            }}
          />
        </div>
      </div>

      {explanation && (
        <p className="mt-2 line-clamp-2 text-xs text-text-secondary">
          {explanation}
        </p>
      )}
    </div>
  );
}
