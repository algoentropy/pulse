import { useEffect, useState } from "react";
import { fetchAnalogues } from "../lib/api";
import type { AnaloguesResponse, AnalogueResult } from "../types";
import { Sparkline } from "./Sparkline";
import { formatChange } from "../lib/format";

function AnalogueCard({ analogue, isRank1 }: { analogue: AnalogueResult; isRank1: boolean }) {
    const {
        rank, start_date, end_date, combined_score, macro_score, price_score,
        forward_returns, sp500_prices, top_feature_diffs
    } = analogue;

    const scorePct = (score: number) => (score * 100).toFixed(1) + "%";

    const FwdReturn = ({ label, val }: { label: string; val: number | undefined }) => {
        if (val === undefined) return <div className="text-zinc-500 text-sm">{label}: —</div>;
        const color = val > 0 ? "text-emerald-400" : val < 0 ? "text-red-400" : "text-zinc-400";
        return (
            <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-500">{label}</span>
                <span className={`font-medium ${color}`}>{formatChange(val * 100)}</span>
            </div>
        );
    };

    return (
        <div className={`p-4 rounded-xl border ${isRank1 ? "border-indigo-500/50 bg-indigo-500/5" : "border-zinc-800 bg-zinc-900/50"} flex flex-col gap-4`}>
            <div className="flex justify-between items-start">
                <div>
                    <div className="flex items-center gap-2">
                        <span className={`text-sm font-bold px-2 py-0.5 rounded ${isRank1 ? "bg-indigo-500 text-white" : "bg-zinc-800 text-zinc-300"}`}>
                            #{rank}
                        </span>
                        <span className="text-zinc-200 font-medium">{start_date} <span className="text-zinc-500">→</span> {end_date}</span>
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-xl font-bold text-zinc-100">{scorePct(combined_score)} Match</div>
                    <div className="text-xs text-zinc-500">Macro: {scorePct(macro_score)} | Price: {scorePct(price_score)}</div>
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">S&P 500 Path</h4>
                    <div className="h-12 w-full flex items-center bg-zinc-950/50 rounded p-2">
                        <Sparkline data={sp500_prices} />
                    </div>
                </div>

                <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Forward Returns</h4>
                    <div className="space-y-1 bg-zinc-950/50 rounded p-2">
                        <FwdReturn label="5 Days" val={forward_returns["5d"]} />
                        <FwdReturn label="1 Month" val={forward_returns["21d"]} />
                        <FwdReturn label="3 Months" val={forward_returns["63d"]} />
                    </div>
                </div>

                <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Key Macro Differences</h4>
                    <div className="space-y-1 bg-zinc-950/50 rounded p-2 text-xs">
                        {top_feature_diffs.slice(0, 3).map((diff, i) => (
                            <div key={i} className="flex justify-between items-center text-zinc-400">
                                <span className="truncate max-w-[120px]" title={diff.feature}>{diff.feature.replace(/^macro_|^[\^_]/, "").replace(/_ret_/, " ")}</span>
                                <span className="text-zinc-500 font-medium ml-2">Δ {(diff.abs_diff).toFixed(2)}σ</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

export function Analogues() {
    const [data, setData] = useState<AnaloguesResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [refreshing, setRefreshing] = useState(false);

    const loadData = async (refresh = false) => {
        if (refresh) setRefreshing(true);
        else setLoading(true);
        setError(null);

        try {
            const result = await fetchAnalogues(refresh);
            if (result.status === "error") {
                throw new Error((result as any).message || "Unknown error");
            }
            setData(result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    if (loading) {
        return (
            <div className="mt-8">
                <h2 className="text-xl font-bold text-zinc-100 mb-4">Historical Analogues</h2>
                <div className="animate-pulse rounded-xl bg-zinc-800/60 h-32 w-full" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="mt-8">
                <h2 className="text-xl font-bold text-zinc-100 mb-4">Historical Analogues</h2>
                <div className="p-4 rounded-xl border border-red-900/50 bg-red-900/10 text-red-400">
                    Failed to load analogues: {error}
                </div>
            </div>
        );
    }

    if (!data || !data.analogues || data.analogues.length === 0) {
        return null; // hide if no data
    }

    return (
        <div className="mt-8">
            <div className="flex justify-between items-end mb-4">
                <div>
                    <h2 className="text-xl font-bold text-zinc-100">Historical Analogues</h2>
                    <p className="text-sm text-zinc-500 mt-1">
                        Top past periods with macro environments and price action resembling today.
                        (Window: {data.window_days} days)
                    </p>
                </div>
                <button
                    onClick={() => loadData(true)}
                    disabled={refreshing}
                    className="text-xs px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-300 rounded transition-colors"
                >
                    {refreshing ? "Refreshing..." : "Refresh"}
                </button>
            </div>

            <div className="space-y-4">
                {data.analogues.map((analogue, i) => (
                    <AnalogueCard key={`${analogue.start_date}-${analogue.end_date}`} analogue={analogue} isRank1={i === 0} />
                ))}
            </div>
        </div>
    );
}
