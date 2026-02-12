"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchCurrentRegime,
  fetchRegimeHistory,
  fetchClassifierVotes,
  fetchTransitionMatrix,
  fetchDisagreement,
} from "@/lib/api";
import {
  REGIME_COLORS,
  REGIME_NAMES,
  STALE_TIME,
  REFETCH_INTERVAL,
} from "@/lib/constants";
import { pct, num } from "@/lib/utils";

import RegimeBadge from "@/components/ui/RegimeBadge";
import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function RegimePage() {
  const regimeQ = useQuery({
    queryKey: ["regime", "current"],
    queryFn: fetchCurrentRegime,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const historyQ = useQuery({
    queryKey: ["regime", "history"],
    queryFn: () => fetchRegimeHistory(),
    staleTime: STALE_TIME,
  });

  const classifiersQ = useQuery({
    queryKey: ["regime", "classifiers"],
    queryFn: fetchClassifierVotes,
    staleTime: STALE_TIME,
  });

  const transitionQ = useQuery({
    queryKey: ["regime", "transitions"],
    queryFn: fetchTransitionMatrix,
    staleTime: STALE_TIME,
  });

  const disagreementQ = useQuery({
    queryKey: ["regime", "disagreement"],
    queryFn: () => fetchDisagreement(),
    staleTime: STALE_TIME,
  });

  const regime = regimeQ.data;
  const classifiers = classifiersQ.data;
  const transitions = transitionQ.data;
  const disagreement = disagreementQ.data;
  const history = historyQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Regime Explorer
        </h1>
        <p className="text-sm text-text-secondary">
          Deep dive into regime detection, classifier votes, and transition
          dynamics
        </p>
      </div>

      {/* Current regime detail */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {regime ? (
          <>
            <MetricsCard
              label="Current Regime"
              value={REGIME_NAMES[regime.regime] ?? "Unknown"}
              color={REGIME_COLORS[regime.regime]}
            />
            <MetricsCard
              label="Confidence"
              value={pct(regime.confidence)}
              color={REGIME_COLORS[regime.regime]}
            />
            <MetricsCard
              label="Disagreement"
              value={pct(regime.disagreement)}
              sub={regime.transition_warning ? "⚠ Warning" : "Stable"}
              color={regime.disagreement > 0.6 ? "#ef4444" : undefined}
            />
            <MetricsCard
              label="Avg Disagreement"
              value={disagreement ? pct(disagreement.avg_disagreement) : "—"}
            />
            <MetricsCard
              label="Total Transitions"
              value={transitions?.total_transitions ?? "—"}
            />
          </>
        ) : regimeQ.isError ? (
          <div className="col-span-5">
            <ErrorState onRetry={() => regimeQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Middle row: Classifier breakdown + Transition matrix */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Classifier radar / bar */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Classifier Weights & Votes
          </p>
          {classifiers ? (
            <PlotlyChart
              height={300}
              data={classifiers.votes.map((v) => ({
                type: "bar" as const,
                x: [v.classifier.replace(/_/g, " ")],
                y: [v.weight],
                marker: { color: REGIME_COLORS[v.regime] ?? "#6b7280" },
                name: REGIME_NAMES[v.regime] ?? "",
                text: [REGIME_NAMES[v.regime]],
                textposition: "inside" as const,
                hovertemplate: `${v.classifier}<br>Weight: %{y:.2f}<br>Vote: ${REGIME_NAMES[v.regime]}<br>Confidence: ${(v.confidence * 100).toFixed(1)}%<extra></extra>`,
              }))}
              layout={{
                barmode: "group",
                showlegend: false,
                yaxis: { title: { text: "Weight", font: { size: 10 } } },
              }}
            />
          ) : classifiersQ.isError ? (
            <ErrorState onRetry={() => classifiersQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-64" />
          )}
        </div>

        {/* Transition matrix heatmap */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Transition Matrix
          </p>
          {transitions ? (
            <PlotlyChart
              height={300}
              data={[
                {
                  type: "heatmap",
                  z: transitions.matrix,
                  x: transitions.regime_names,
                  y: transitions.regime_names,
                  colorscale: [
                    [0, "#0a0a0f"],
                    [0.5, "#3b82f6"],
                    [1, "#22c55e"],
                  ],
                  showscale: true,
                  colorbar: {
                    tickfont: { color: "#8888a0", size: 10 },
                    len: 0.8,
                  },
                  hovertemplate:
                    "From: %{y}<br>To: %{x}<br>Probability: %{z:.2%}<extra></extra>",
                } as Plotly.Data,
              ]}
              layout={{
                xaxis: { side: "bottom" as const, tickfont: { size: 10 } },
                yaxis: {
                  autorange: "reversed" as const,
                  tickfont: { size: 10 },
                },
                margin: { t: 10, l: 100, r: 20, b: 80 },
              }}
            />
          ) : transitionQ.isError ? (
            <ErrorState onRetry={() => transitionQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-64" />
          )}
        </div>
      </div>

      {/* Disagreement time series */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Disagreement Over Time
        </p>
        {disagreement ? (
          <PlotlyChart
            height={250}
            data={[
              {
                x: disagreement.series.map((d) => d.date),
                y: disagreement.series.map((d) => d.disagreement),
                type: "scatter",
                mode: "lines",
                line: { color: "#8888a0", width: 1.5 },
                fill: "tozeroy",
                fillcolor: "rgba(136,136,160,0.06)",
                name: "Disagreement",
              },
              {
                x: [
                  disagreement.series[0]?.date,
                  disagreement.series[disagreement.series.length - 1]?.date,
                ],
                y: [disagreement.threshold, disagreement.threshold],
                type: "scatter",
                mode: "lines",
                line: { color: "#ef4444", dash: "dash", width: 1 },
                name: "Threshold",
              },
            ]}
            layout={{
              showlegend: true,
              legend: { orientation: "h", y: -0.2, font: { size: 10 } },
              yaxis: {
                range: [0, 1],
                title: { text: "Disagreement", font: { size: 10 } },
              },
              xaxis: { type: "date" },
            }}
          />
        ) : disagreementQ.isError ? (
          <ErrorState onRetry={() => disagreementQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-56" />
        )}
      </div>

      {/* Regime history strip chart */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Regime History
        </p>
        {history ? (
          <PlotlyChart
            height={200}
            data={[1, 2, 3, 4].map((rid) => ({
              x: history.history.map((h) => h.date),
              y: history.history.map((h) => (h.regime === rid ? rid : null)),
              type: "scatter" as const,
              mode: "markers" as const,
              marker: { color: REGIME_COLORS[rid], size: 4 },
              name: REGIME_NAMES[rid],
              connectgaps: false,
            }))}
            layout={{
              showlegend: true,
              legend: { orientation: "h", y: -0.25, font: { size: 10 } },
              yaxis: {
                tickvals: [1, 2, 3, 4],
                ticktext: Object.values(REGIME_NAMES),
                tickfont: { size: 10 },
              },
              xaxis: { type: "date" },
              margin: { t: 10, l: 120, r: 20, b: 50 },
            }}
          />
        ) : historyQ.isError ? (
          <ErrorState onRetry={() => historyQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-44" />
        )}
      </div>
    </div>
  );
}
