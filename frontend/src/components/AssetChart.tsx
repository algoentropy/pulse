import { useEffect, useRef } from "react";
import { createChart, ColorType, AreaSeries } from "lightweight-charts";
import type { HistoryPoint } from "../types";

export function AssetChart({ data }: { data: HistoryPoint[] }) {
    const chartContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!chartContainerRef.current || !data || data.length === 0) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: "transparent" },
                textColor: "#a1a1aa", // zinc-400
                attributionLogo: false,
            },
            grid: {
                vertLines: { color: "#27272a" }, // zinc-800
                horzLines: { color: "#27272a" }, // zinc-800
            },
            width: chartContainerRef.current.clientWidth,
            height: 250,
            timeScale: {
                timeVisible: true,
                borderColor: "#3f3f46",
            },
            rightPriceScale: {
                borderColor: "#3f3f46",
                autoScale: true,
            },
            crosshair: {
                horzLine: { color: "#a1a1aa", labelBackgroundColor: "#27272a" },
                vertLine: { color: "#a1a1aa", labelBackgroundColor: "#27272a" },
            },
        });

        const lineSeries = chart.addSeries(AreaSeries, {
            lineColor: "#34d399", // emerald-400
            topColor: "rgba(52, 211, 153, 0.4)",
            bottomColor: "rgba(52, 211, 153, 0.0)",
            lineWidth: 2,
        });
        lineSeries.setData(data.map((d) => ({ time: d.time as any, value: d.value })));

        chart.timeScale().fitContent();

        const handleResize = () => {
            chart.applyOptions({ width: chartContainerRef.current?.clientWidth || 0 });
        };

        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
            chart.remove();
        };
    }, [data]);

    return <div ref={chartContainerRef} className="w-full h-[250px]" />;
}
