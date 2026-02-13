"use client";

import PlotlyChart from "./PlotlyChart";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";
import type { RegimeReturnEntry } from "@/lib/types";

interface RegimeReturnsChartProps {
  returns: RegimeReturnEntry[];
  height?: number;
}

/**
 * Grouped bar chart comparing strategy vs benchmark returns by regime.
 * Used in the Backtest Lab page.
 */
export default function RegimeReturnsChart({
  returns,
  height = 300,
}: RegimeReturnsChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          type: "bar",
          x: returns.map((r) => r.regime_name),
          y: returns.map((r) => r.strategy_return * 100),
          name: "Strategy",
          marker: {
            color: returns.map((r) => REGIME_COLORS[r.regime] ?? "#3b82f6"),
          },
          hovertemplate: "%{x}<br>Strategy: %{y:.1f}%<extra></extra>",
        },
        {
          type: "bar",
          x: returns.map((r) => r.regime_name),
          y: returns.map((r) => r.benchmark_return * 100),
          name: "Benchmark",
          marker: { color: "#555568" },
          hovertemplate: "%{x}<br>Benchmark: %{y:.1f}%<extra></extra>",
        },
      ]}
      layout={{
        barmode: "group",
        showlegend: true,
        legend: { orientation: "h", y: -0.2, font: { size: 10 } },
        yaxis: {
          title: { text: "Return (%)", font: { size: 10 } },
        },
        margin: { t: 10, l: 50, r: 20, b: 60 },
      }}
    />
  );
}
