"use client";

import PlotlyChart from "./PlotlyChart";

interface WeightEvolutionChartProps {
  history: { date: string; weights: Record<string, number> }[];
  height?: number;
}

const WEIGHT_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7"];

/**
 * Stacked area chart showing classifier weight evolution over time.
 * Used in the Meta-Learning page.
 */
export default function WeightEvolutionChart({
  history,
  height = 300,
}: WeightEvolutionChartProps) {
  const classifiers = Object.keys(history[0]?.weights ?? {});

  return (
    <PlotlyChart
      height={height}
      data={classifiers.map((clf, idx) => ({
        x: history.map((h) => h.date),
        y: history.map((h) => h.weights[clf] ?? 0),
        type: "scatter" as const,
        mode: "lines" as const,
        name: clf.replace(/_/g, " "),
        line: { color: WEIGHT_COLORS[idx % WEIGHT_COLORS.length], width: 1.5 },
        stackgroup: "one",
      }))}
      layout={{
        showlegend: true,
        legend: { orientation: "h", y: -0.2, font: { size: 10 } },
        yaxis: {
          title: { text: "Weight", font: { size: 10 } },
          range: [0, 1],
        },
        xaxis: { type: "date" },
      }}
    />
  );
}
