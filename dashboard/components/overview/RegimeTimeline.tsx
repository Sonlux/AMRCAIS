"use client";

import PlotlyChart from "@/components/charts/PlotlyChart";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";
import type { RegimeHistoryPoint } from "@/lib/types";

interface RegimeTimelineProps {
  history: RegimeHistoryPoint[];
}

export default function RegimeTimeline({ history }: RegimeTimelineProps) {
  if (!history.length) return null;

  // Build separate traces per regime for colored segments
  const regimeIds = [1, 2, 3, 4];
  const traces: Plotly.Data[] = regimeIds.map((rid) => ({
    x: history.map((h) => h.date),
    y: history.map((h) => (h.regime === rid ? h.confidence : null)),
    type: "scatter" as const,
    mode: "lines" as const,
    name: REGIME_NAMES[rid],
    line: { color: REGIME_COLORS[rid], width: 2 },
    fill: "tozeroy",
    fillcolor: `${REGIME_COLORS[rid]}12`,
    connectgaps: false,
    hovertemplate: `%{x}<br>Confidence: %{y:.1%}<extra>${REGIME_NAMES[rid]}</extra>`,
  }));

  // Disagreement trace on secondary axis
  traces.push({
    x: history.map((h) => h.date),
    y: history.map((h) => h.disagreement),
    type: "scatter",
    mode: "lines",
    name: "Disagreement",
    yaxis: "y2",
    line: { color: "#6b7280", width: 1, dash: "dot" },
    hovertemplate: "%{x}<br>Disagreement: %{y:.2f}<extra></extra>",
  });

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
        Regime Timeline
      </p>
      <PlotlyChart
        data={traces}
        height={240}
        layout={{
          showlegend: true,
          legend: {
            orientation: "h",
            y: -0.15,
            font: { size: 10, color: "#8888a0" },
          },
          xaxis: { type: "date" },
          yaxis: {
            title: { text: "Confidence", font: { size: 10 } },
            range: [0, 1],
          },
          yaxis2: {
            title: { text: "Disagreement", font: { size: 10 } },
            overlaying: "y",
            side: "right",
            range: [0, 1],
          },
          margin: { t: 10, r: 50, b: 50, l: 50 },
        }}
      />
    </div>
  );
}
