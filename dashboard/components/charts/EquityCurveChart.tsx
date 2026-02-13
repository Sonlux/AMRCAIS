"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";
import type { EquityPoint } from "@/lib/types";
import type { TVDataPoint, TVMarker } from "./LightweightChart";

// Lazy-load the canvas-based chart (no SSR)
const LightweightChart = dynamic(() => import("./LightweightChart"), {
  ssr: false,
  loading: () => (
    <div
      className="flex items-center justify-center bg-surface"
      style={{ height: 340 }}
    >
      <span className="text-xs text-text-muted">Loading chart…</span>
    </div>
  ),
});

interface EquityCurveChartProps {
  equityCurve: EquityPoint[];
  /** Show regime-change markers on the chart. */
  showRegimeMarkers?: boolean;
  height?: number;
}

/**
 * Equity curve rendered with TradingView Lightweight Charts.
 * Much higher performance than Plotly for large backtests.
 * Optionally overlaid with regime-change markers.
 */
export default function EquityCurveChart({
  equityCurve,
  showRegimeMarkers = true,
  height = 340,
}: EquityCurveChartProps) {
  const tvData: TVDataPoint[] = useMemo(
    () => equityCurve.map((p) => ({ time: p.date, value: p.value })),
    [equityCurve],
  );

  const markers: TVMarker[] = useMemo(() => {
    if (!showRegimeMarkers) return [];
    const m: TVMarker[] = [];
    for (let i = 1; i < equityCurve.length; i++) {
      const prev = equityCurve[i - 1];
      const curr = equityCurve[i];
      if (
        curr.regime !== null &&
        prev.regime !== null &&
        curr.regime !== prev.regime
      ) {
        m.push({
          time: curr.date,
          position: "belowBar",
          color: REGIME_COLORS[curr.regime] ?? "#555568",
          shape: "circle",
          text: REGIME_NAMES[curr.regime] ?? `R${curr.regime}`,
        });
      }
    }
    return m;
  }, [equityCurve, showRegimeMarkers]);

  return (
    <LightweightChart
      data={tvData}
      markers={markers}
      height={height}
      seriesType="area"
      color="#3b82f6"
      precision={0}
    />
  );
}
