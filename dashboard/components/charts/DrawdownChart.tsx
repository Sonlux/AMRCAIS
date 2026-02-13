"use client";

import PlotlyChart from "./PlotlyChart";
import type { DrawdownPoint } from "@/lib/types";

interface DrawdownChartProps {
  series: DrawdownPoint[];
  height?: number;
}

/**
 * Inverted area chart showing portfolio drawdown percentage over time.
 * Drawdowns are shown as negative red-filled areas below zero.
 */
export default function DrawdownChart({
  series,
  height = 200,
}: DrawdownChartProps) {
  if (!series || series.length === 0) return null;

  return (
    <PlotlyChart
      height={height}
      data={[
        {
          x: series.map((d) => d.date),
          y: series.map((d) => d.drawdown),
          type: "scatter" as const,
          mode: "lines" as const,
          fill: "tozeroy",
          fillcolor: "rgba(239,68,68,0.12)",
          line: { color: "#ef4444", width: 1.5 },
          hovertemplate: "%{x}<br>Drawdown: %{y:.2%}<extra></extra>",
        },
      ]}
      layout={{
        showlegend: false,
        yaxis: {
          title: { text: "Drawdown", font: { size: 10 } },
          tickformat: ".0%",
          autorange: true,
        },
        xaxis: { type: "date" },
        margin: { t: 10, l: 50, r: 20, b: 40 },
      }}
    />
  );
}
