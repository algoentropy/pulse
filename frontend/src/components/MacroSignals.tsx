import { useEffect, useRef } from 'react';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';
import type { FeaturesResponse, FeaturePoint } from '../types';

interface MacroSignalsProps {
    data: FeaturesResponse;
}

function LineChart({ data, color, title }: { data: FeaturePoint[]; color: string; title: string }) {
    const chartContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!chartContainerRef.current || !data || data.length === 0) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#a1a1aa', // zinc-400
                attributionLogo: false,
            },
            grid: {
                vertLines: { color: '#27272a' }, // zinc-800
                horzLines: { color: '#27272a' }, // zinc-800
            },
            width: chartContainerRef.current.clientWidth,
            height: 250,
            timeScale: {
                timeVisible: true,
                borderColor: '#3f3f46',
            },
            rightPriceScale: {
                borderColor: '#3f3f46',
                autoScale: true,
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.1,
                },
            },
            localization: {
                priceFormatter: (price: number) => {
                    // If price is very small, show more decimals
                    if (price > 0 && price < 0.1) {
                        return price.toFixed(5);
                    }
                    return price.toFixed(2);
                },
            },
        });

        const series = chart.addSeries(LineSeries, {
            color: color,
            lineWidth: 2,
        });

        series.setData(data.map(d => ({ time: d.time as any, value: d.value })));

        const handleResize = () => {
            chart.applyOptions({ width: chartContainerRef.current?.clientWidth || 0 });
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [data, color]);

    return (
        <div className="flex flex-col space-y-2">
            <h3 className="text-sm font-semibold text-zinc-300">{title}</h3>
            <div ref={chartContainerRef} className="w-full h-[250px]" />
        </div>
    );
}

export function MacroSignals({ data }: MacroSignalsProps) {
    if (!data || (!data.copper_gold.length && !data.vix_tnx.length)) {
        return null;
    }

    return (
        <div className="mt-8">
            <div className="mb-4">
                <h2 className="text-xl font-bold tracking-tight text-zinc-100">Macro Signals (1Y Trend)</h2>
                <p className="text-sm text-zinc-500">Engineered features for quantitative modeling</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 rounded-xl border border-zinc-800 bg-zinc-900/80 p-4">
                {data.copper_gold.length > 0 && (
                    <LineChart
                        data={data.copper_gold}
                        color="#fbbf24" // amber-400 (copper/gold ish)
                        title="Copper / Gold Ratio (Growth vs Safety)"
                    />
                )}
                {data.vix_tnx.length > 0 && (
                    <LineChart
                        data={data.vix_tnx}
                        color="#a78bfa" // violet-400
                        title="VIX / 10Y Yield Ratio (Equity Risk / Risk-Free)"
                    />
                )}
            </div>
        </div>
    );
}
