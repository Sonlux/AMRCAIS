"use client";

import PlotlyChart from "./PlotlyChart";
import type { DisagreementPoint, PricePoint } from "@/lib/types";

interface DisagreementVsSpxChartProps {
  disagreement: DisagreementPoint[];
  prices: PricePoint[];
  height?: number;
}

/**
 * Dual-axis chart: disagreement index on the left Y-axis, SPX price on the
 * right Y-axis. Bands where disagreement > 0.6 are highlighted.
 */
export default function DisagreementVsSpxChart({
  disagreement,
  prices,
  height = 320,
}: DisagreementVsSpxChartProps) {
  if (!disagreement.length) return null;

  // Build shapes for high-disagreement bands
  const shapes: Plotly.Shape[] = [];
  let bandStart: string | null = null;
  for (let i = 0; i < disagreement.length; i++) {
    const d = disagreement[i];
    if (d.disagreement > 0.6 && !bandStart) {
      bandStart = d.date;
    } else if (
      (d.disagreement <= 0.6 || i === disagreement.length - 1) &&
      bandStart
    ) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: bandStart,
        x1: d.date,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(239,68,68,0.06)",
        line: { width: 0 },
      } as Plotly.Shape);
      bandStart = null;
    }
  }

  return (
    <PlotlyChart
      height={height}
      data={[
        {
          x: disagreement.map((d) => d.date),
          y: disagreement.map((d) => d.disagreement),
          type: "scatter" as const,
          mode: "lines" as const,
          name: "Disagreement",
          yaxis: "y",
          fill: "tozeroy",
          fillcolor: "rgba(245,158,11,0.08)",
          line: { color: "#f59e0b", width: 1.5 },
          hovertemplate: "%{x}<br>Disagreement: %{y:.2f}<extra></extra>",
        },
        {
          x: prices.map((p) => p.date),
          y: prices.map((p) => p.close),
          type: "scatter" as const,
          mode: "lines" as const,
          name: "SPX",
          yaxis: "y2",
          line: { color: "#3b82f6", width: 1.5 },
          hovertemplate: "%{x}<br>SPX: %{y:,.0f}<extra></extra>",
        },
      ]}
      layout={{
        showlegend: true,
        legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center" },
        yaxis: {
          title: { text: "Disagreement", font: { size: 10 } },
          range: [0, 1],
          side: "left",
        },
        yaxis2: {
          title: { text: "SPX Price", font: { size: 10 } },
          overlaying: "y",
          side: "right",
        },
        xaxis: { type: "date" },
        shapes,
        margin: { t: 10, l: 50, r: 60, b: 60 },
      }}
    />
  );
}
