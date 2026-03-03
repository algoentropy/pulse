import { useEffect, useRef } from "react";
import { createChart, LineSeries, type IChartApi } from "lightweight-charts";
import type { HistoryPoint } from "../types";

interface SparklineProps {
  data: HistoryPoint[];
}

export function Sparkline({ data }: SparklineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length < 2) return;

    const trendUp = data[data.length - 1].value >= data[0].value;
    const lineColor = trendUp ? "#34d399" : "#f87171";

    const chart = createChart(containerRef.current, {
      width: 120,
      height: 36,
      layout: { background: { color: "transparent" }, textColor: "transparent" },
      grid: { vertLines: { visible: false }, horzLines: { visible: false } },
      crosshair: { mode: 0 },
      rightPriceScale: { visible: false },
      timeScale: { visible: false },
      handleScroll: false,
      handleScale: false,
    });

    const series = chart.addSeries(LineSeries, {
      color: lineColor,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });

    series.setData(data);
    chart.timeScale().fitContent();
    chartRef.current = chart;

    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [data]);

  return <div ref={containerRef} className="shrink-0 [&_a[href*='tradingview']]:!hidden" />;
}
