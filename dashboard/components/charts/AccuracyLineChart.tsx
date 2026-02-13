"use client";

import PlotlyChart from "./PlotlyChart";
import type { ClassifierAccuracyResponse } from "@/lib/types";

const CLASSIFIER_COLORS: Record<string, string> = {
  hmm: "#3b82f6",
  ml: "#22c55e",
  correlation: "#f59e0b",
  volatility: "#ef4444",
  ensemble: "#a855f7",
};

interface AccuracyLineChartProps {
  data: ClassifierAccuracyResponse;
  height?: number;
}

/**
 * Multi-line chart showing rolling classification accuracy per classifier.
 * A dashed 75 % threshold line highlights the acceptable-accuracy boundary.
 */
export default function AccuracyLineChart({
  data,
  height = 300,
}: AccuracyLineChartProps) {
  const traces = data.classifiers.map((clf) => {
    const pts = data.series.filter((p) => p.classifier === clf);
    return {
      x: pts.map((p) => p.date),
      y: pts.map((p) => p.accuracy),
      type: "scatter" as const,
      mode: "lines" as const,
      name: clf.toUpperCase(),
      line: {
        color: CLASSIFIER_COLORS[clf] ?? "#6b7280",
        width: clf === "ensemble" ? 2.5 : 1.5,
      },
      hovertemplate: `${clf.toUpperCase()}<br>%{x}<br>Accuracy: %{y:.1%}<extra></extra>`,
    };
  });

  // Threshold line at 75 %
  const allDates = data.series.map((p) => p.date);
  if (allDates.length > 0) {
    traces.push({
      x: [allDates[0], allDates[allDates.length - 1]],
      y: [0.75, 0.75],
      type: "scatter" as const,
      mode: "lines" as const,
      name: "75 % threshold",
      line: { color: "#ef4444", width: 1, dash: "dash" as const } as never,
      hoverinfo: "skip" as never,
    } as never);
  }

  return (
    <PlotlyChart
      height={height}
      data={traces}
      layout={{
        showlegend: true,
        legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center" },
        yaxis: {
          title: { text: "Accuracy", font: { size: 10 } },
          range: [0, 1],
          tickformat: ".0%",
        },
        xaxis: { type: "date" },
        margin: { t: 10, l: 50, r: 20, b: 60 },
      }}
    />
  );
}
