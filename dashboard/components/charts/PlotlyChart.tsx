"use client";

import dynamic from "next/dynamic";
import type { PlotParams } from "react-plotly.js";
import { PLOTLY_DARK_LAYOUT, PLOTLY_CONFIG } from "@/lib/constants";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface PlotlyChartProps extends Partial<PlotParams> {
  data: PlotParams["data"];
  title?: string;
  height?: number;
  className?: string;
}

export default function PlotlyChart({
  data,
  title,
  height = 300,
  layout: layoutOverride,
  config: configOverride,
  className,
  ...rest
}: PlotlyChartProps) {
  const layout: Partial<Plotly.Layout> = {
    ...PLOTLY_DARK_LAYOUT,
    height,
    title: title
      ? { text: title, font: { size: 14, color: "#e4e4ef" } }
      : undefined,
    ...layoutOverride,
  };

  return (
    <div className={className}>
      <Plot
        data={data}
        layout={layout}
        config={{ ...PLOTLY_CONFIG, ...configOverride }}
        useResizeHandler
        style={{ width: "100%", height }}
        {...rest}
      />
    </div>
  );
}
