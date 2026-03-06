import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';
import type { BacktestResponse } from '../types';
import { fetchBacktest } from '../lib/api';

export function BacktestChart({ triggerKey }: { triggerKey: number }) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const [data, setData] = useState<BacktestResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let active = true;
        setLoading(true);
        fetchBacktest()
            .then(res => {
                if (active) {
                    setData(res);
                    setError(null);
                }
            })
            .catch(e => {
                if (active) setError(e.message);
            })
            .finally(() => {
                if (active) setLoading(false);
            });

        return () => { active = false; };
    }, [triggerKey]);

    useEffect(() => {
        if (!chartContainerRef.current || !data || data.strategy.length === 0) return;

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
            height: 300,
            timeScale: {
                timeVisible: true,
                borderColor: '#3f3f46',
            },
            rightPriceScale: {
                borderColor: '#3f3f46',
                autoScale: true,
            },
        });

        // Benchmark line (S&P 500)
        const benchmarkSeries = chart.addSeries(LineSeries, {
            color: '#9ca3af', // zinc-400
            lineWidth: 2,
            lineType: 0,
        });
        benchmarkSeries.setData(data.benchmark.map(d => ({ time: d.time as any, value: d.value })));

        // Strategy line
        const strategySeries = chart.addSeries(LineSeries, {
            color: '#34d399', // emerald-400
            lineWidth: 2,
            lineType: 0,
        });
        strategySeries.setData(data.strategy.map(d => ({ time: d.time as any, value: d.value })));

        chart.timeScale().fitContent();

        const handleResize = () => {
            chart.applyOptions({ width: chartContainerRef.current?.clientWidth || 0 });
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [data]);

    if (loading) {
        return <div className="h-[300px] w-full flex items-center justify-center text-zinc-500 animate-pulse">Loading backtest data...</div>;
    }

    if (error) {
        return <div className="h-[300px] w-full flex items-center justify-center text-red-500 text-sm">Error: {error}</div>;
    }

    return (
        <div className="mt-8 border-t border-zinc-800 pt-8">
            <div className="mb-4">
                <h3 className="text-lg font-bold tracking-tight text-zinc-100">Out-Of-Sample Equity Curve (Last 3 Years)</h3>
                <p className="text-sm text-zinc-500">
                    <span className="text-emerald-400 font-medium">Model Strategy</span> vs <span className="text-zinc-400 font-medium">Buy & Hold Benchmark</span>
                </p>
            </div>
            <div ref={chartContainerRef} className="w-full h-[300px]" />
        </div>
    );
}
