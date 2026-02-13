"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchSystemHealth,
  fetchPerformance,
  fetchWeights,
  fetchWeightHistory,
  fetchRecalibrations,
} from "@/lib/api";
import { STALE_TIME, REFETCH_INTERVAL } from "@/lib/constants";
import { pct, num, cn } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import {
  ClassifierWeightsChart,
  RegimeDistributionChart,
  WeightEvolutionChart,
} from "@/components/charts";

export default function MetaPage() {
  const healthQ = useQuery({
    queryKey: ["meta", "health"],
    queryFn: fetchSystemHealth,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const perfQ = useQuery({
    queryKey: ["meta", "performance"],
    queryFn: fetchPerformance,
    staleTime: STALE_TIME,
  });

  const weightsQ = useQuery({
    queryKey: ["meta", "weights"],
    queryFn: fetchWeights,
    staleTime: STALE_TIME,
  });

  const weightHistQ = useQuery({
    queryKey: ["meta", "weight-history"],
    queryFn: fetchWeightHistory,
    staleTime: STALE_TIME,
  });

  const recalQ = useQuery({
    queryKey: ["meta", "recalibrations"],
    queryFn: fetchRecalibrations,
    staleTime: STALE_TIME,
  });

  const health = healthQ.data;
  const perf = perfQ.data;
  const weights = weightsQ.data;
  const weightHist = weightHistQ.data;
  const recalibrations = recalQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">Meta-Learning</h1>
        <p className="text-sm text-text-secondary">
          Classifier performance tracking, adaptive weights, and recalibration
          history
        </p>
      </div>

      {/* System health banner */}
      {health && (
        <div
          className={cn(
            "rounded-lg border p-4",
            health.needs_recalibration
              ? "border-regime-3/40 bg-regime-3/5"
              : "border-regime-1/40 bg-regime-1/5",
          )}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-foreground">
                System Status:{" "}
                <span
                  className={
                    health.needs_recalibration
                      ? "text-regime-3"
                      : "text-regime-1"
                  }
                >
                  {health.system_status}
                </span>
              </p>
              {health.needs_recalibration && (
                <p className="mt-1 text-xs text-text-secondary">
                  Urgency: {health.urgency} — Severity: {num(health.severity)}
                </p>
              )}
            </div>
            {health.reasons.length > 0 && (
              <div className="text-right">
                {health.reasons.map((r, i) => (
                  <p key={i} className="text-xs text-text-secondary">
                    {r}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {perf ? (
          <>
            <MetricsCard
              label="Stability Score"
              value={num(perf.stability_score)}
              sub={perf.stability_rating}
              color={
                perf.stability_score > 0.7
                  ? "#22c55e"
                  : perf.stability_score > 0.4
                    ? "#f59e0b"
                    : "#ef4444"
              }
            />
            <MetricsCard
              label="Transitions"
              value={perf.transition_count}
              sub="Total regime changes"
            />
            <MetricsCard
              label="Avg Disagreement"
              value={pct(perf.avg_disagreement)}
              color={perf.avg_disagreement > 0.5 ? "#f59e0b" : undefined}
            />
            <MetricsCard
              label="High Disagree Days"
              value={perf.high_disagreement_days}
              sub={`of ${perf.total_classifications} total`}
            />
            <MetricsCard
              label="Recalibrations"
              value={recalibrations?.total_recalibrations ?? "—"}
              sub={
                recalibrations?.last_recalibration
                  ? `Last: ${recalibrations.last_recalibration}`
                  : "None yet"
              }
            />
          </>
        ) : perfQ.isError ? (
          <div className="col-span-5">
            <ErrorState onRetry={() => perfQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Middle row: Weights + Regime distribution */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Current weights */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Classifier Weights
            {weights?.is_adaptive && (
              <span className="ml-2 rounded bg-accent/10 px-1.5 py-0.5 text-accent">
                Adaptive
              </span>
            )}
          </p>
          {weights ? (
            <ClassifierWeightsChart weights={weights.weights} height={260} />
          ) : weightsQ.isError ? (
            <ErrorState onRetry={() => weightsQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-56" />
          )}
        </div>

        {/* Regime distribution pie */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Regime Distribution
          </p>
          {perf ? (
            <RegimeDistributionChart
              distribution={perf.regime_distribution}
              height={260}
            />
          ) : (
            <ChartSkeleton height="h-56" />
          )}
        </div>
      </div>

      {/* Weight history over time */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Weight Evolution Over Time
        </p>
        {weightHist && weightHist.history.length > 0 ? (
          <WeightEvolutionChart history={weightHist.history} height={300} />
        ) : weightHistQ.isError ? (
          <ErrorState onRetry={() => weightHistQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-64" />
        )}
      </div>

      {/* Recalibration events */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Recalibration Events
        </p>
        {recalibrations && recalibrations.events.length > 0 ? (
          <div className="space-y-3">
            {recalibrations.events.map((evt, i) => (
              <div
                key={i}
                className="rounded-md border border-border/50 bg-surface-elevated p-3"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-text-muted">{evt.date}</span>
                    <span
                      className={cn(
                        "rounded px-1.5 py-0.5 text-xs font-medium",
                        evt.urgency === "high"
                          ? "bg-regime-2/10 text-regime-2"
                          : evt.urgency === "medium"
                            ? "bg-regime-3/10 text-regime-3"
                            : "bg-regime-4/10 text-regime-4",
                      )}
                    >
                      {evt.urgency}
                    </span>
                  </div>
                  <span className="font-mono text-xs text-text-secondary">
                    Severity: {num(evt.severity)}
                  </span>
                </div>
                <p className="mt-1 text-sm text-text-secondary">
                  {evt.trigger_reason}
                </p>
                {evt.recommendations.length > 0 && (
                  <ul className="mt-2 space-y-0.5">
                    {evt.recommendations.map((rec, j) => (
                      <li key={j} className="text-xs text-text-muted">
                        • {rec}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        ) : recalQ.isError ? (
          <ErrorState onRetry={() => recalQ.refetch()} />
        ) : recalibrations ? (
          <p className="text-sm text-text-secondary">
            No recalibration events recorded yet.
          </p>
        ) : (
          <ChartSkeleton height="h-32" />
        )}
      </div>
    </div>
  );
}
