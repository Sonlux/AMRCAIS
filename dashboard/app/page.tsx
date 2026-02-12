"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchCurrentRegime,
  fetchModuleSummary,
  fetchRegimeHistory,
  fetchDisagreement,
  fetchStatus,
} from "@/lib/api";
import { REFETCH_INTERVAL, STALE_TIME, REGIME_COLORS } from "@/lib/constants";
import { pct } from "@/lib/utils";

import RegimeBadge from "@/components/ui/RegimeBadge";
import MetricsCard from "@/components/ui/MetricsCard";
import SignalCard from "@/components/ui/SignalCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import DisagreementGauge from "@/components/overview/DisagreementGauge";
import RegimeTimeline from "@/components/overview/RegimeTimeline";

export default function OverviewPage() {
  const regimeQ = useQuery({
    queryKey: ["regime", "current"],
    queryFn: fetchCurrentRegime,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const modulesQ = useQuery({
    queryKey: ["modules", "summary"],
    queryFn: fetchModuleSummary,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const historyQ = useQuery({
    queryKey: ["regime", "history"],
    queryFn: () => fetchRegimeHistory(),
    staleTime: STALE_TIME,
  });

  const disagreementQ = useQuery({
    queryKey: ["regime", "disagreement"],
    queryFn: () => fetchDisagreement(),
    staleTime: STALE_TIME,
  });

  const statusQ = useQuery({
    queryKey: ["system", "status"],
    queryFn: fetchStatus,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const regime = regimeQ.data;
  const modules = modulesQ.data;
  const history = historyQ.data;
  const disagreement = disagreementQ.data;
  const status = statusQ.data;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-foreground">Overview</h1>
          <p className="text-sm text-text-secondary">
            Real-time regime-aware market analytics
          </p>
        </div>
        {regime && (
          <RegimeBadge
            regime={regime.regime}
            confidence={regime.confidence}
            size="lg"
          />
        )}
      </div>

      {/* Top metrics row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {regime ? (
          <>
            <MetricsCard
              label="Confidence"
              value={pct(regime.confidence)}
              sub="Ensemble confidence"
              color={REGIME_COLORS[regime.regime]}
            />
            <MetricsCard
              label="Disagreement"
              value={pct(regime.disagreement)}
              sub={
                regime.transition_warning ? "⚠ Transition warning" : "Stable"
              }
              color={regime.disagreement > 0.6 ? "#ef4444" : undefined}
            />
            <MetricsCard
              label="Classifiers"
              value={Object.keys(regime.classifier_votes).length}
              sub="Active voter count"
            />
            <MetricsCard
              label="Uptime"
              value={
                status?.uptime_seconds
                  ? `${Math.floor(status.uptime_seconds / 60)}m`
                  : "—"
              }
              sub={status?.is_initialized ? "System ready" : "Initializing"}
            />
          </>
        ) : regimeQ.isError ? (
          <div className="col-span-4">
            <ErrorState
              message="Failed to load regime data"
              onRetry={() => regimeQ.refetch()}
            />
          </div>
        ) : (
          <>
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
          </>
        )}
      </div>

      {/* Middle row: Disagreement gauge + Classifier votes */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          {disagreement ? (
            <DisagreementGauge
              value={disagreement.avg_disagreement}
              threshold={disagreement.threshold}
            />
          ) : disagreementQ.isError ? (
            <ErrorState
              message="Disagreement data unavailable"
              onRetry={() => disagreementQ.refetch()}
            />
          ) : (
            <ChartSkeleton height="h-48" />
          )}
        </div>

        {/* Classifier vote breakdown */}
        <div className="rounded-lg border border-border bg-surface p-4 lg:col-span-2">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Classifier Votes
          </p>
          {regime ? (
            <div className="space-y-3">
              {Object.entries(regime.classifier_votes).map(
                ([classifier, vote]) => (
                  <div key={classifier} className="flex items-center gap-3">
                    <span className="w-28 truncate text-sm text-text-secondary">
                      {classifier.replace(/_/g, " ")}
                    </span>
                    <div className="flex-1">
                      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-elevated">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${(regime.probabilities[classifier] ?? 0.5) * 100}%`,
                            backgroundColor: REGIME_COLORS[vote] ?? "#6b7280",
                          }}
                        />
                      </div>
                    </div>
                    <RegimeBadge regime={vote} size="sm" />
                  </div>
                ),
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="skeleton h-6 w-full rounded" />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Signal cards */}
      <div>
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Module Signals
        </p>
        {modules ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {modules.signals.map((sig) => (
              <SignalCard
                key={sig.module}
                module={sig.module.replace(/_/g, " ")}
                signal={sig.signal}
                strength={sig.strength}
                explanation={sig.explanation}
              />
            ))}
          </div>
        ) : modulesQ.isError ? (
          <ErrorState
            message="Module signals unavailable"
            onRetry={() => modulesQ.refetch()}
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {[1, 2, 3, 4, 5].map((i) => (
              <CardSkeleton key={i} />
            ))}
          </div>
        )}
      </div>

      {/* Regime timeline */}
      {history ? (
        <RegimeTimeline history={history.history} />
      ) : historyQ.isError ? (
        <ErrorState
          message="Regime history unavailable"
          onRetry={() => historyQ.refetch()}
        />
      ) : (
        <ChartSkeleton height="h-60" />
      )}
    </div>
  );
}
