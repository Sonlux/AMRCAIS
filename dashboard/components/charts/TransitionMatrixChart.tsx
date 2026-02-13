"use client";

import PlotlyChart from "./PlotlyChart";
import { REGIME_NAMES, REGIME_COLORS } from "@/lib/constants";

interface TransitionMatrixChartProps {
  matrix: number[][];
  regimeNames: string[];
  height?: number;
}

/**
 * Heatmap displaying regime transition probabilities.
 * Used in the Regime Explorer page.
 */
export default function TransitionMatrixChart({
  matrix,
  regimeNames,
  height = 280,
}: TransitionMatrixChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          type: "heatmap",
          z: matrix,
          x: regimeNames,
          y: regimeNames,
          colorscale: [
            [0, "#0a0a14"],
            [0.5, "#3b82f6"],
            [1, "#22c55e"],
          ],
          hovertemplate:
            "From: %{y}<br>To: %{x}<br>Prob: %{z:.2%}<extra></extra>",
          showscale: true,
          colorbar: {
            tickfont: { size: 10, color: "#8888a0" },
            len: 0.8,
          },
        } as Plotly.Data,
      ]}
      layout={{
        xaxis: {
          title: { text: "To Regime", font: { size: 10 } },
          tickfont: { size: 9 },
        },
        yaxis: {
          title: { text: "From Regime", font: { size: 10 } },
          tickfont: { size: 9 },
          autorange: "reversed",
        },
        margin: { t: 10, l: 100, r: 20, b: 80 },
      }}
    />
  );
}
