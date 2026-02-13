"use client";

import { useEffect, useRef, useCallback } from "react";
import {
  createChart,
  AreaSeries,
  LineSeries,
  createSeriesMarkers,
  type IChartApi,
  type DeepPartial,
  type ChartOptions,
  ColorType,
} from "lightweight-charts";

/* ─── Types ───────────────────────────────────────────────── */

export interface TVDataPoint {
  time: string; // YYYY-MM-DD
  value: number;
}

export interface TVMarker {
  time: string;
  position: "aboveBar" | "belowBar" | "inBar";
  color: string;
  shape: "circle" | "square" | "arrowUp" | "arrowDown";
  text: string;
}

interface LightweightChartProps {
  /** Primary series data. */
  data: TVDataPoint[];
  /** Optional benchmark/overlay line. */
  benchmarkData?: TVDataPoint[];
  /** Optional regime-change markers plotted on the chart. */
  markers?: TVMarker[];
  /** Chart height in px. */
  height?: number;
  /** Series type: area (filled) or line. */
  seriesType?: "area" | "line";
  /** Primary series colour. */
  color?: string;
  /** Benchmark line colour. */
  benchmarkColor?: string;
  /** Optional container class. */
  className?: string;
  /** Price format precision. */
  precision?: number;
  /** Show crosshair tooltip values. */
  crosshair?: boolean;
}

/* ─── Component ───────────────────────────────────────────── */

/**
 * Wrapper around TradingView Lightweight Charts v5.
 *
 * Renders an area or line chart optimised for financial time-series data.
 * Replaces Plotly for equity curves and price charts where canvas
 * performance and lightweight bundle size matter.
 */
export default function LightweightChart({
  data,
  benchmarkData,
  markers,
  height = 300,
  seriesType = "area",
  color = "#3b82f6",
  benchmarkColor = "#555568",
  className,
  precision = 2,
  crosshair = true,
}: LightweightChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // Chart options that match the dark AMRCAIS theme
  const getChartOptions = useCallback(
    (width: number): DeepPartial<ChartOptions> => ({
      width,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8888a0",
        fontFamily: "var(--font-geist-sans), system-ui",
      },
      grid: {
        vertLines: { color: "#1a1a2822" },
        horzLines: { color: "#1a1a2844" },
      },
      crosshair: {
        mode: crosshair ? 0 : 1, // 0 = Normal, 1 = Magnet
      },
      rightPriceScale: {
        borderColor: "#2a2a3a",
      },
      timeScale: {
        borderColor: "#2a2a3a",
        timeVisible: false,
      },
    }),
    [height, crosshair],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Create chart
    const chart = createChart(
      container,
      getChartOptions(container.clientWidth),
    );
    chartRef.current = chart;

    // Primary series
    let series: ReturnType<typeof chart.addSeries>;

    if (seriesType === "area") {
      series = chart.addSeries(AreaSeries, {
        lineColor: color,
        topColor: `${color}44`,
        bottomColor: `${color}08`,
        lineWidth: 2,
        priceFormat: { type: "price", precision, minMove: 1 / 10 ** precision },
      });
    } else {
      series = chart.addSeries(LineSeries, {
        color,
        lineWidth: 2,
        priceFormat: { type: "price", precision, minMove: 1 / 10 ** precision },
      });
    }

    // TradingView expects { time, value } sorted ascending
    const sorted = [...data].sort(
      (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime(),
    );
    series.setData(sorted as never);

    // Markers (regime change indicators, etc.)
    if (markers?.length) {
      const sortedMarkers = [...markers].sort(
        (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime(),
      );
      createSeriesMarkers(series, sortedMarkers as never);
    }

    // Benchmark overlay
    if (benchmarkData?.length) {
      const benchSeries = chart.addSeries(AreaSeries, {
        lineColor: benchmarkColor,
        topColor: "transparent",
        bottomColor: "transparent",
        lineWidth: 1,
        lineStyle: 2, // dashed
        priceFormat: { type: "price", precision, minMove: 1 / 10 ** precision },
      });

      const sortedBench = [...benchmarkData].sort(
        (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime(),
      );
      benchSeries.setData(sortedBench as never);
    }

    // Fit content
    chart.timeScale().fitContent();

    // Resize observer for responsive behaviour
    const ro = new ResizeObserver(() => {
      if (container.clientWidth > 0) {
        chart.applyOptions({ width: container.clientWidth });
      }
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [
    data,
    benchmarkData,
    markers,
    seriesType,
    color,
    benchmarkColor,
    precision,
    getChartOptions,
  ]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ width: "100%", height }}
    />
  );
}
