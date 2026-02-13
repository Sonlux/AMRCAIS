"use client";

import PlotlyChart from "./PlotlyChart";
import type { DisagreementPoint } from "@/lib/types";

interface DisagreementSeriesChartProps {
  series: DisagreementPoint[];
  threshold: number;
  height?: number;
}

/**
 * Time-series chart of regime disagreement with threshold line.
 * Used in the Regime Explorer page.
 */
export default function DisagreementSeriesChart({
  series,
  threshold,
  height = 260,
}: DisagreementSeriesChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          x: series.map((d) => d.date),
          y: series.map((d) => d.disagreement),
          type: "scatter" as const,
          mode: "lines" as const,
          fill: "tozeroy",
          fillcolor: "rgba(245,158,11,0.08)",
          line: { color: "#f59e0b", width: 1.5 },
          hovertemplate: "%{x}<br>Disagreement: %{y:.2f}<extra></extra>",
        },
        {
          x: [series[0]?.date, series[series.length - 1]?.date],
          y: [threshold, threshold],
          type: "scatter" as const,
          mode: "lines" as const,
          line: { color: "#ef4444", width: 1, dash: "dash" as const },
          name: "Threshold",
          hoverinfo: "skip" as const,
        },
      ]}
      layout={{
        showlegend: false,
        yaxis: {
          title: { text: "Disagreement", font: { size: 10 } },
          range: [0, 1],
        },
        xaxis: { type: "date" },
        margin: { t: 10, l: 50, r: 20, b: 40 },
      }}
    />
  );
}
