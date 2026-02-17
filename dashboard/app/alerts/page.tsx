"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchAlerts,
  fetchEvents,
  fetchAlertConfig,
  acknowledgeAlert,
  fetchPhase4Status,
} from "@/lib/api";
import { STALE_TIME, REFETCH_INTERVAL } from "@/lib/constants";
import { num, cn } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, TableSkeleton } from "@/components/ui/Skeleton";

export default function AlertsPage() {
  const [severityFilter, setSeverityFilter] = useState<string>("");
  const [showUnackOnly, setShowUnackOnly] = useState(false);
  const queryClient = useQueryClient();

  // Phase 4 status pre-fetched for cache
  useQuery({
    queryKey: ["phase4", "status"],
    queryFn: fetchPhase4Status,
    staleTime: STALE_TIME,
  });

  const alertsQ = useQuery({
    queryKey: ["phase4", "alerts", severityFilter, showUnackOnly],
    queryFn: () => fetchAlerts(100, severityFilter || undefined, showUnackOnly),
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const eventsQ = useQuery({
    queryKey: ["phase4", "events"],
    queryFn: () => fetchEvents(50),
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const configQ = useQuery({
    queryKey: ["phase4", "alert-config"],
    queryFn: fetchAlertConfig,
    staleTime: STALE_TIME,
  });

  const ackMut = useMutation({
    mutationFn: acknowledgeAlert,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["phase4", "alerts"] });
    },
  });

  const alerts = alertsQ.data;
  const events = eventsQ.data;
  const config = configQ.data;

  const severityColor = (s: string) => {
    switch (s) {
      case "critical":
        return "#ef4444";
      case "high":
        return "#f59e0b";
      case "medium":
        return "#3b82f6";
      default:
        return "#6b7280";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Alerts &amp; Events
        </h1>
        <p className="text-sm text-text-secondary">
          Real-time alert management, event log, and system monitoring
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {alerts ? (
          <>
            <MetricsCard label="Total Alerts" value={alerts.total} />
            <MetricsCard
              label="Unacknowledged"
              value={alerts.unacknowledged}
              color={alerts.unacknowledged > 0 ? "#ef4444" : "#22c55e"}
            />
            <MetricsCard label="Total Events" value={events?.total ?? "—"} />
            <MetricsCard
              label="Alert Configs"
              value={config ? Object.keys(config.configs).length : "—"}
            />
          </>
        ) : alertsQ.isError ? (
          <div className="col-span-4">
            <ErrorState onRetry={() => alertsQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 4 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div>
          <label className="mb-1 block text-xs text-text-secondary">
            Severity
          </label>
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
          >
            <option value="">All</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>
        <div className="flex items-end gap-2 pt-5">
          <input
            type="checkbox"
            id="unack"
            checked={showUnackOnly}
            onChange={(e) => setShowUnackOnly(e.target.checked)}
            className="accent-accent"
          />
          <label htmlFor="unack" className="text-xs text-text-secondary">
            Unacknowledged only
          </label>
        </div>
      </div>

      {/* Alerts list */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Alerts ({alerts?.alerts.length ?? 0})
        </p>
        {alerts && alerts.alerts.length > 0 ? (
          <div className="space-y-2">
            {alerts.alerts.map((alert) => (
              <div
                key={alert.alert_id}
                className={cn(
                  "flex items-start justify-between rounded-md border p-3",
                  alert.acknowledged
                    ? "border-border/50 bg-surface-elevated"
                    : "border-border bg-surface",
                )}
              >
                <div className="flex items-start gap-3">
                  <span
                    className="mt-0.5 rounded px-2 py-0.5 text-xs font-semibold"
                    style={{
                      backgroundColor: `${severityColor(alert.severity)}18`,
                      color: severityColor(alert.severity),
                    }}
                  >
                    {alert.severity}
                  </span>
                  <div>
                    <p className="text-sm font-medium text-foreground">
                      {alert.alert_type.replace(/_/g, " ")}
                    </p>
                    <p className="mt-0.5 text-xs text-text-secondary">
                      {alert.message}
                    </p>
                    <p className="mt-1 text-xs text-text-muted">
                      {alert.timestamp}
                    </p>
                  </div>
                </div>
                {!alert.acknowledged && (
                  <button
                    onClick={() => ackMut.mutate(alert.alert_id)}
                    disabled={ackMut.isPending}
                    className="shrink-0 rounded bg-surface-elevated px-3 py-1 text-xs font-medium text-text-secondary transition-colors hover:bg-border hover:text-foreground disabled:opacity-50"
                  >
                    Ack
                  </button>
                )}
                {alert.acknowledged && (
                  <span className="text-xs text-text-muted">✓</span>
                )}
              </div>
            ))}
          </div>
        ) : alerts ? (
          <p className="text-sm text-text-secondary">No alerts to display</p>
        ) : (
          <TableSkeleton rows={5} />
        )}
      </div>

      {/* Alert Config */}
      {config && Object.keys(config.configs).length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Alert Configuration
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-muted">
                  <th className="pb-2 pr-4">Alert Type</th>
                  <th className="pb-2 pr-4">Enabled</th>
                  <th className="pb-2 pr-4">Cooldown (s)</th>
                  <th className="pb-2">Threshold</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(config.configs).map(([type, cfg], i) => (
                  <tr
                    key={type}
                    className={cn(
                      "border-b border-border/50",
                      i % 2 === 0 ? "bg-surface" : "bg-surface-elevated",
                    )}
                  >
                    <td className="py-2 pr-4 font-medium text-foreground">
                      {type.replace(/_/g, " ")}
                    </td>
                    <td className="py-2 pr-4">
                      <span
                        className={cn(
                          "rounded px-2 py-0.5 text-xs font-medium",
                          cfg.enabled
                            ? "bg-regime-1/10 text-regime-1"
                            : "bg-regime-2/10 text-regime-2",
                        )}
                      >
                        {cfg.enabled ? "ON" : "OFF"}
                      </span>
                    </td>
                    <td className="py-2 pr-4 font-mono text-text-secondary">
                      {cfg.cooldown_seconds}
                    </td>
                    <td className="py-2 font-mono text-text-secondary">
                      {cfg.threshold != null ? num(cfg.threshold) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent Events */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Recent Events ({events?.events.length ?? 0})
        </p>
        {events && events.events.length > 0 ? (
          <div className="max-h-96 space-y-1.5 overflow-y-auto">
            {events.events.map((evt) => (
              <div
                key={evt.event_id}
                className="flex items-center justify-between rounded-md bg-surface-elevated px-3 py-2 text-xs"
              >
                <div className="flex items-center gap-2">
                  <span className="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-accent">
                    {evt.event_type}
                  </span>
                  <span className="text-text-secondary">{evt.source}</span>
                </div>
                <span className="text-text-muted">{evt.timestamp}</span>
              </div>
            ))}
          </div>
        ) : events ? (
          <p className="text-sm text-text-secondary">No events recorded</p>
        ) : (
          <TableSkeleton rows={5} />
        )}
      </div>
    </div>
  );
}
