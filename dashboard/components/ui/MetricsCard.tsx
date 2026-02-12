import { cn } from "@/lib/utils";

interface MetricsCardProps {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
  className?: string;
}

export default function MetricsCard({
  label,
  value,
  sub,
  color,
  className,
}: MetricsCardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-surface p-4 transition-colors hover:border-border-light",
        className,
      )}
    >
      <p className="text-xs font-medium uppercase tracking-wider text-text-muted">
        {label}
      </p>
      <p
        className="mt-1 font-mono text-2xl font-semibold"
        style={color ? { color } : undefined}
      >
        {value}
      </p>
      {sub && <p className="mt-0.5 text-xs text-text-secondary">{sub}</p>}
    </div>
  );
}
