"use client";

import { Suspense } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchModuleSummary,
  fetchModuleAnalysis,
  fetchModuleHistory,
} from "@/lib/api";
import {
  MODULE_NAMES,
  SIGNAL_COLORS,
  REGIME_NAMES,
  STALE_TIME,
  REFETCH_INTERVAL,
} from "@/lib/constants";
import { pct, cn } from "@/lib/utils";
import { useQueryState } from "@/lib/hooks";

import SignalCard from "@/components/ui/SignalCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import { SignalHistoryChart } from "@/components/charts";

const MODULE_KEYS = [
  "macro",
  "yield_curve",
  "options",
  "factors",
  "correlations",
] as const;

export default function ModulesPage() {
  return (
    <Suspense>
      <ModulesContent />
    </Suspense>
  );
}

function ModulesContent() {
  const [activeTab, setActiveTab] = useQueryState("tab", MODULE_KEYS[0]);

  const summaryQ = useQuery({
    queryKey: ["modules", "summary"],
    queryFn: fetchModuleSummary,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const analysisQ = useQuery({
    queryKey: ["modules", "analysis", activeTab],
    queryFn: () => fetchModuleAnalysis(activeTab),
    staleTime: STALE_TIME,
  });

  const historyQ = useQuery({
    queryKey: ["modules", "history", activeTab],
    queryFn: () => fetchModuleHistory(activeTab),
    staleTime: STALE_TIME,
  });

  const summary = summaryQ.data;
  const analysis = analysisQ.data;
  const history = historyQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Module Deep Dive
        </h1>
        <p className="text-sm text-text-secondary">
          Regime-adaptive analytical modules — signal analysis and history
        </p>
      </div>

      {/* Signal summary row */}
      <div>
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Current Signals
          {summary && (
            <span className="ml-2 text-text-secondary">
              — Regime: {REGIME_NAMES[summary.current_regime] ?? "Unknown"}
            </span>
          )}
        </p>
        {summary ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {summary.signals.map((sig) => (
              <SignalCard
                key={sig.module}
                module={MODULE_NAMES[sig.module] ?? sig.module}
                signal={sig.signal}
                strength={sig.strength}
                explanation={sig.explanation}
              />
            ))}
          </div>
        ) : summaryQ.isError ? (
          <ErrorState onRetry={() => summaryQ.refetch()} />
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {MODULE_KEYS.map((k) => (
              <CardSkeleton key={k} />
            ))}
          </div>
        )}
      </div>

      {/* Module tabs */}
      <div className="flex gap-1 overflow-x-auto border-b border-border pb-px">
        {MODULE_KEYS.map((key) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={cn(
              "shrink-0 border-b-2 px-4 py-2 text-sm font-medium transition-colors",
              activeTab === key
                ? "border-accent text-foreground"
                : "border-transparent text-text-secondary hover:text-foreground",
            )}
          >
            {MODULE_NAMES[key] ?? key}
          </button>
        ))}
      </div>

      {/* Active module detail */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Left column: signal detail + regime parameters */}
        <div className="space-y-4">
          {analysis ? (
            <>
              <div className="rounded-lg border border-border bg-surface p-4">
                <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
                  Signal Detail
                </p>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Signal</span>
                    <span
                      className="font-semibold"
                      style={{
                        color:
                          SIGNAL_COLORS[analysis.signal.signal] ?? "#6b7280",
                      }}
                    >
                      {analysis.signal.signal}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Strength</span>
                    <span className="font-mono">
                      {pct(analysis.signal.strength)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Confidence</span>
                    <span className="font-mono">
                      {pct(analysis.signal.confidence)}
                    </span>
                  </div>
                </div>
                <p className="mt-3 text-xs text-text-secondary">
                  {analysis.signal.explanation}
                </p>
                <p className="mt-1 text-xs italic text-text-muted">
                  {analysis.signal.regime_context}
                </p>
              </div>

              {/* Regime parameters */}
              <div className="rounded-lg border border-border bg-surface p-4">
                <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
                  Regime Parameters
                </p>
                <div className="space-y-1.5">
                  {Object.entries(analysis.regime_parameters).map(
                    ([key, val]) => (
                      <div key={key} className="flex justify-between text-xs">
                        <span className="text-text-secondary">
                          {key.replace(/_/g, " ")}
                        </span>
                        <span className="font-mono text-foreground">
                          {typeof val === "number"
                            ? val.toFixed(4)
                            : String(val)}
                        </span>
                      </div>
                    ),
                  )}
                </div>
              </div>

              {/* Raw metrics */}
              <div className="rounded-lg border border-border bg-surface p-4">
                <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
                  Raw Metrics
                </p>
                <pre className="max-h-48 overflow-auto text-xs text-text-secondary">
                  {JSON.stringify(analysis.raw_metrics, null, 2)}
                </pre>
              </div>
            </>
          ) : analysisQ.isError ? (
            <ErrorState onRetry={() => analysisQ.refetch()} />
          ) : (
            <>
              <CardSkeleton />
              <CardSkeleton />
            </>
          )}
        </div>

        {/* Right column: signal history chart */}
        <div className="lg:col-span-2">
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Signal History — {MODULE_NAMES[activeTab] ?? activeTab}
            </p>
            {history && history.history.length > 0 ? (
              <SignalHistoryChart history={history.history} height={400} />
            ) : historyQ.isError ? (
              <ErrorState onRetry={() => historyQ.refetch()} />
            ) : (
              <ChartSkeleton height="h-96" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
