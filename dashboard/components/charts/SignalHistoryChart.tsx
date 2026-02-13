"use client";

import PlotlyChart from "./PlotlyChart";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";

interface SignalHistoryChartProps {
  history: { date: string; signal: string; strength: number; regime: number }[];
  height?: number;
}

/**
 * Dual-axis chart: signal strength over time with regime colour coding.
 * Used in the Module Deep Dive page.
 */
export default function SignalHistoryChart({
  history,
  height = 300,
}: SignalHistoryChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          x: history.map((h) => h.date),
          y: history.map((h) => h.strength),
          type: "scatter" as const,
          mode: "lines" as const,
          name: "Signal Strength",
          line: { color: "#3b82f6", width: 1.5 },
          hovertemplate: "%{x}<br>Strength: %{y:.2f}<extra></extra>",
        },
        {
          x: history.map((h) => h.date),
          y: history.map((h) => h.regime),
          type: "scatter" as const,
          mode: "lines" as const,
          name: "Regime",
          yaxis: "y2",
          line: { color: "#555568", width: 1, dash: "dot" as const },
          marker: {
            color: history.map((h) => REGIME_COLORS[h.regime] ?? "#555568"),
            size: 4,
          },
          hovertemplate: history.map(
            (h) =>
              `${h.date}<br>${REGIME_NAMES[h.regime] ?? `Regime ${h.regime}`}<extra></extra>`,
          ),
        },
      ]}
      layout={{
        showlegend: true,
        legend: {
          orientation: "h",
          y: -0.15,
          font: { size: 10 },
        },
        yaxis: {
          title: { text: "Signal Strength", font: { size: 10 } },
          range: [-1, 1],
        },
        yaxis2: {
          title: { text: "Regime", font: { size: 10 } },
          overlaying: "y",
          side: "right",
          dtick: 1,
          range: [0.5, 4.5],
        },
        xaxis: { type: "date" },
        margin: { t: 10, r: 50, b: 50, l: 50 },
      }}
    />
  );
}
