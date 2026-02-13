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
import { pct } from "@/lib/utils";
import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";
import {
  TransitionMatrixChart,
  DisagreementSeriesChart,
  RegimeStripChart,
} from "@/components/charts";

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
            <TransitionMatrixChart
              matrix={transitions.matrix}
              regimeNames={transitions.regime_names}
              height={300}
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
          <DisagreementSeriesChart
            series={disagreement.series}
            threshold={disagreement.threshold}
            height={250}
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
          <RegimeStripChart history={history.history} height={200} />
        ) : historyQ.isError ? (
          <ErrorState onRetry={() => historyQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-44" />
        )}
      </div>
    </div>
  );
}
