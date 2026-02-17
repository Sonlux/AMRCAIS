"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchResearchSummary,
  fetchResearchReports,
  generateCaseStudy,
} from "@/lib/api";
import { STALE_TIME } from "@/lib/constants";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, TableSkeleton } from "@/components/ui/Skeleton";

const REGIME_OPTIONS = [
  { value: 1, label: "Risk-On Growth" },
  { value: 2, label: "Risk-Off Crisis" },
  { value: 3, label: "Stagflation" },
  { value: 4, label: "Disinflationary Boom" },
];

export default function ResearchPage() {
  const queryClient = useQueryClient();

  const [fromRegime, setFromRegime] = useState<number>(1);
  const [toRegime, setToRegime] = useState<number>(2);
  const [expandedReport, setExpandedReport] = useState<string | null>(null);

  const summaryQ = useQuery({
    queryKey: ["phase5", "research-summary"],
    queryFn: fetchResearchSummary,
    staleTime: STALE_TIME,
  });

  const reportsQ = useQuery({
    queryKey: ["phase5", "research-reports"],
    queryFn: () => fetchResearchReports(30),
    staleTime: STALE_TIME,
  });

  const caseMutation = useMutation({
    mutationFn: () => generateCaseStudy(fromRegime, toRegime),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["phase5", "research-reports"],
      });
      queryClient.invalidateQueries({
        queryKey: ["phase5", "research-summary"],
      });
    },
  });

  const summary = summaryQ.data;
  const reports = reportsQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Research Publisher
        </h1>
        <p className="text-sm text-text-secondary">
          Auto-generated research reports, regime transition case studies, and
          factor analysis
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {summary ? (
          <>
            <MetricsCard label="Total Reports" value={summary.total_reports} />
            <MetricsCard
              label="Report Types"
              value={
                summary.report_types
                  ? Object.keys(summary.report_types).length
                  : 0
              }
            />
            <MetricsCard
              label="Last Report"
              value={summary.last_report ?? "—"}
            />
            <MetricsCard
              label="Publisher Status"
              value={summary.total_reports > 0 ? "Active" : "Idle"}
            />
          </>
        ) : summaryQ.isError ? (
          <div className="col-span-4">
            <ErrorState onRetry={() => summaryQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 4 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Report type breakdown */}
      {summary?.report_types &&
        Object.keys(summary.report_types).length > 0 && (
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Reports by Type
            </p>
            <div className="flex flex-wrap gap-3">
              {Object.entries(summary.report_types).map(([type, count]) => (
                <div
                  key={type}
                  className="rounded-md border border-border/50 bg-surface-elevated px-3 py-2"
                >
                  <p className="text-xs text-text-muted">{type}</p>
                  <p className="font-mono text-lg font-semibold text-foreground">
                    {count as number}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

      {/* Case study generator */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Generate Case Study
        </p>
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              From Regime
            </label>
            <select
              value={fromRegime}
              onChange={(e) => setFromRegime(Number(e.target.value))}
              className="rounded-md border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
            >
              {REGIME_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <span className="pb-1.5 text-text-muted">→</span>
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              To Regime
            </label>
            <select
              value={toRegime}
              onChange={(e) => setToRegime(Number(e.target.value))}
              className="rounded-md border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
            >
              {REGIME_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => caseMutation.mutate()}
            disabled={caseMutation.isPending || fromRegime === toRegime}
            className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent/80 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {caseMutation.isPending ? "Generating..." : "Generate"}
          </button>
        </div>
        {fromRegime === toRegime && (
          <p className="mt-2 text-xs text-amber-400">
            Source and target regimes must differ
          </p>
        )}
        {caseMutation.isError && (
          <p className="mt-2 text-xs text-red-400">
            Failed to generate case study. Try again.
          </p>
        )}
        {caseMutation.isSuccess && (
          <p className="mt-2 text-xs text-green-400">
            Case study generated successfully.
          </p>
        )}
      </div>

      {/* Report list */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Research Reports ({reports?.total ?? 0})
        </p>
        {reports && reports.reports.length > 0 ? (
          <div className="space-y-2">
            {reports.reports.map((r) => (
              <div
                key={r.id}
                className="rounded-md border border-border/50 bg-surface-elevated p-3"
              >
                <div
                  className="flex cursor-pointer items-center justify-between"
                  onClick={() =>
                    setExpandedReport(expandedReport === r.id ? null : r.id)
                  }
                >
                  <div className="flex items-center gap-3">
                    <span className="rounded bg-accent/20 px-2 py-0.5 text-xs font-medium text-accent">
                      {r.report_type}
                    </span>
                    <span className="text-sm font-medium text-foreground">
                      {r.title}
                    </span>
                  </div>
                  <span className="text-xs text-text-muted">
                    {r.created_at}
                  </span>
                </div>
                {expandedReport === r.id && (
                  <div className="mt-3 border-t border-border/50 pt-3">
                    {r.content ? (
                      <pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs text-text-secondary">
                        {typeof r.content === "string"
                          ? r.content
                          : JSON.stringify(r.content, null, 2)}
                      </pre>
                    ) : (
                      <p className="text-xs text-text-muted italic">
                        No content available
                      </p>
                    )}
                    {r.metadata && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {Object.entries(r.metadata).map(([k, v]) => (
                          <span
                            key={k}
                            className="rounded bg-surface px-2 py-0.5 text-xs text-text-muted"
                          >
                            {k}: {String(v)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : reports ? (
          <p className="text-sm text-text-secondary">No reports yet</p>
        ) : reportsQ.isError ? (
          <ErrorState onRetry={() => reportsQ.refetch()} />
        ) : (
          <TableSkeleton rows={5} />
        )}
      </div>
    </div>
  );
}
