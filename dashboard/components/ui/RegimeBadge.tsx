import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";

interface RegimeBadgeProps {
  regime: number;
  confidence?: number;
  size?: "sm" | "md" | "lg";
}

export default function RegimeBadge({
  regime,
  confidence,
  size = "md",
}: RegimeBadgeProps) {
  const color = REGIME_COLORS[regime] ?? "#6b7280";
  const name = REGIME_NAMES[regime] ?? "Unknown";

  const sizeClasses = {
    sm: "text-xs px-2 py-0.5",
    md: "text-sm px-3 py-1",
    lg: "text-base px-4 py-1.5",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${sizeClasses[size]}`}
      style={{
        backgroundColor: `${color}18`,
        color,
        border: `1px solid ${color}40`,
      }}
    >
      <span
        className="h-2 w-2 rounded-full"
        style={{ backgroundColor: color }}
      />
      {name}
      {confidence !== undefined && (
        <span className="ml-1 opacity-70">
          {(confidence * 100).toFixed(0)}%
        </span>
      )}
    </span>
  );
}
