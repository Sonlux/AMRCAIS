"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchCurrentRegime } from "@/lib/api";
import { REGIME_COLORS, REGIME_NAMES, REFETCH_INTERVAL } from "@/lib/constants";
import { pct } from "@/lib/utils";

const BREADCRUMB_MAP: Record<string, string> = {
  "/": "Overview",
  "/regime": "Regime Explorer",
  "/modules": "Module Deep Dive",
  "/correlations": "Correlation Monitor",
  "/backtest": "Backtest Lab",
  "/meta": "Meta-Learning",
};

export default function Topbar() {
  const pathname = usePathname();
  const pageTitle = BREADCRUMB_MAP[pathname] ?? "AMRCAIS";

  // Render time only on the client to avoid SSR hydration mismatch
  const [time, setTime] = useState<string>("");
  useEffect(() => {
    const tick = () => setTime(new Date().toLocaleTimeString());
    tick();
    const id = setInterval(tick, 1_000);
    return () => clearInterval(id);
  }, []);

  const { data: regime } = useQuery({
    queryKey: ["regime", "current"],
    queryFn: fetchCurrentRegime,
    refetchInterval: REFETCH_INTERVAL,
  });

  const regimeId = regime?.regime ?? 0;
  const regimeColor = REGIME_COLORS[regimeId] ?? "#6b7280";

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-border bg-surface/80 px-6 backdrop-blur-sm">
      {/* Left: breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <span className="text-text-muted">AMRCAIS</span>
        <span className="text-text-muted">/</span>
        <span className="font-medium text-foreground">{pageTitle}</span>
      </div>

      {/* Right: regime pill + last updated */}
      <div className="flex items-center gap-4">
        {regime && (
          <div
            className="flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold"
            style={{
              backgroundColor: `${regimeColor}18`,
              color: regimeColor,
              border: `1px solid ${regimeColor}40`,
            }}
          >
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: regimeColor }}
            />
            {REGIME_NAMES[regimeId] ?? "Unknown"}
            <span className="text-text-muted">|</span>
            <span>{pct(regime.confidence)}</span>
          </div>
        )}

        <span className="text-xs text-text-muted">{time}</span>
      </div>
    </header>
  );
}
