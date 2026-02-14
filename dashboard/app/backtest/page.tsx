"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { runBacktest, fetchBacktestResults } from "@/lib/api";
import { REGIME_COLORS, STALE_TIME, TRACKED_ASSETS } from "@/lib/constants";
import { pct, pctRaw, num } from "@/lib/utils";
import type { BacktestRequest, BacktestResultResponse } from "@/lib/types";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import DataTable from "@/components/ui/DataTable";
import {
  EquityCurveChart,
  RegimeReturnsChart,
  DrawdownChart,
} from "@/components/charts";
import type { ColumnDef } from "@tanstack/react-table";
import type { RegimeReturnEntry, TradeLogEntry } from "@/lib/types";

/* ─── TanStack Table column definitions ───────────────────── */

const regimeBreakdownColumns: ColumnDef<RegimeReturnEntry, unknown>[] = [
  {
    accessorKey: "regime_name",
    header: "Regime",
    cell: ({ row }) => (
      <span
        className="font-medium"
        style={{ color: REGIME_COLORS[row.original.regime] ?? "#6b7280" }}
      >
        {row.original.regime_name}
      </span>
    ),
  },
  { accessorKey: "days", header: "Days" },
  {
    accessorKey: "strategy_return",
    header: "Strategy Return",
    cell: ({ getValue }) => {
      const v = getValue() as number;
      return (
        <span
          className="font-mono"
          style={{ color: v >= 0 ? "#22c55e" : "#ef4444" }}
        >
          {pctRaw(v)}
        </span>
      );
    },
  },
  {
    accessorKey: "benchmark_return",
    header: "Benchmark Return",
    cell: ({ getValue }) => {
      const v = getValue() as number;
      return (
        <span
          className="font-mono"
          style={{ color: v >= 0 ? "#22c55e" : "#ef4444" }}
        >
          {pctRaw(v)}
        </span>
      );
    },
  },
  {
    accessorKey: "hit_rate",
    header: "Hit Rate",
    cell: ({ getValue }) => (
      <span className="font-mono text-text-secondary">
        {pct(getValue() as number)}
      </span>
    ),
  },
];

const tradeLogColumns: ColumnDef<TradeLogEntry, unknown>[] = [
  { accessorKey: "date", header: "Date" },
  { accessorKey: "action", header: "Action" },
  {
    accessorKey: "regime_name",
    header: "Regime",
    cell: ({ row }) => (
      <span
        className="font-medium"
        style={{ color: REGIME_COLORS[row.original.regime] ?? "#6b7280" }}
      >
        {row.original.regime_name}
      </span>
    ),
  },
  {
    accessorKey: "allocations",
    header: "Allocations",
    cell: ({ getValue }) => {
      const alloc = getValue() as Record<string, number>;
      return (
        <span className="font-mono text-xs text-text-secondary">
          {Object.entries(alloc)
            .map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`)
            .join(", ")}
        </span>
      );
    },
  },
  {
    accessorKey: "portfolio_value",
    header: "Portfolio Value",
    cell: ({ getValue }) => (
      <span className="font-mono">
        $
        {(getValue() as number).toLocaleString(undefined, {
          maximumFractionDigits: 0,
        })}
      </span>
    ),
  },
  {
    accessorKey: "daily_return",
    header: "Daily Return",
    cell: ({ getValue }) => {
      const v = getValue() as number;
      return (
        <span
          className="font-mono"
          style={{ color: v >= 0 ? "#22c55e" : "#ef4444" }}
        >
          {pctRaw(v)}
        </span>
      );
    },
  },
];

export default function BacktestPage() {
  const [form, setForm] = useState<BacktestRequest>({
    start_date: "2015-01-01",
    end_date: "2024-01-01",
    strategy: "regime_following",
    initial_capital: 100000,
    assets: [...TRACKED_ASSETS],
  });

  const [result, setResult] = useState<BacktestResultResponse | null>(null);

  const backtestMut = useMutation({
    mutationFn: runBacktest,
    onSuccess: (data) => setResult(data),
  });

  const resultsQ = useQuery({
    queryKey: ["backtest", "results"],
    queryFn: fetchBacktestResults,
    staleTime: STALE_TIME,
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">Backtest Lab</h1>
        <p className="text-sm text-text-secondary">
          Walk-forward regime-following strategy backtesting
        </p>
      </div>

      {/* Config form */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Backtest Configuration
        </p>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              Start Date
            </label>
            <input
              type="date"
              value={form.start_date}
              onChange={(e) =>
                setForm((f) => ({ ...f, start_date: e.target.value }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              End Date
            </label>
            <input
              type="date"
              value={form.end_date}
              onChange={(e) =>
                setForm((f) => ({ ...f, end_date: e.target.value }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              Strategy
            </label>
            <select
              value={form.strategy}
              onChange={(e) =>
                setForm((f) => ({ ...f, strategy: e.target.value }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            >
              <option value="regime_following">Regime Following</option>
              <option value="momentum">Momentum</option>
              <option value="mean_reversion">Mean Reversion</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              Initial Capital
            </label>
            <input
              type="number"
              value={form.initial_capital}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  initial_capital: Number(e.target.value),
                }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={() => backtestMut.mutate(form)}
              disabled={backtestMut.isPending}
              className="w-full rounded bg-accent px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-accent/90 disabled:opacity-50"
            >
              {backtestMut.isPending ? "Running…" : "Run Backtest"}
            </button>
          </div>
        </div>
        {backtestMut.isError && (
          <p className="mt-2 text-xs text-regime-2">
            Error: {backtestMut.error.message}
          </p>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
            <MetricsCard
              label="Total Return"
              value={pctRaw(result.total_return)}
              color={result.total_return >= 0 ? "#22c55e" : "#ef4444"}
            />
            <MetricsCard
              label="Sharpe Ratio"
              value={num(result.sharpe_ratio)}
              color={result.sharpe_ratio >= 1 ? "#22c55e" : "#f59e0b"}
            />
            <MetricsCard
              label="Max Drawdown"
              value={pctRaw(result.max_drawdown)}
              color="#ef4444"
            />
            <MetricsCard
              label="Benchmark Return"
              value={pctRaw(result.benchmark_return)}
              color={result.benchmark_return >= 0 ? "#22c55e" : "#ef4444"}
            />
            <MetricsCard
              label="Alpha"
              value={pctRaw(result.total_return - result.benchmark_return)}
              color={
                result.total_return - result.benchmark_return >= 0
                  ? "#22c55e"
                  : "#ef4444"
              }
            />
          </div>

          {/* Equity curve — TradingView Lightweight Charts */}
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Equity Curve
            </p>
            <EquityCurveChart equityCurve={result.equity_curve} height={350} />
          </div>

          {/* Drawdown chart */}
          {result.drawdown_curve && result.drawdown_curve.length > 0 && (
            <div className="rounded-lg border border-border bg-surface p-4">
              <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
                Drawdown
              </p>
              <DrawdownChart series={result.drawdown_curve} height={200} />
            </div>
          )}

          {/* Returns by regime */}
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Returns by Regime
            </p>
            <RegimeReturnsChart returns={result.regime_returns} height={280} />
          </div>

          {/* Regime return table — TanStack Table */}
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Regime Breakdown
            </p>
            <DataTable
              columns={regimeBreakdownColumns}
              data={result.regime_returns}
            />
          </div>

          {/* Trade Log */}
          {result.trade_log && result.trade_log.length > 0 && (
            <div className="rounded-lg border border-border bg-surface p-4">
              <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
                Trade Log ({result.trade_log.length} events)
              </p>
              <DataTable columns={tradeLogColumns} data={result.trade_log} />
            </div>
          )}
        </>
      )}

      {/* Previous results list */}
      {resultsQ.data && resultsQ.data.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Previous Backtest Results
          </p>
          <div className="space-y-2">
            {resultsQ.data.map((r: BacktestResultResponse) => (
              <button
                key={r.id}
                onClick={() => setResult(r)}
                className="flex w-full items-center justify-between rounded-md bg-surface-elevated px-3 py-2 text-sm transition-colors hover:bg-border"
              >
                <span className="text-text-secondary">
                  {r.start_date} → {r.end_date}
                </span>
                <span className="font-mono text-xs">
                  <span
                    style={{
                      color: r.total_return >= 0 ? "#22c55e" : "#ef4444",
                    }}
                  >
                    {pctRaw(r.total_return)}
                  </span>
                  <span className="mx-2 text-text-muted">|</span>
                  <span className="text-text-secondary">
                    SR {num(r.sharpe_ratio)}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
