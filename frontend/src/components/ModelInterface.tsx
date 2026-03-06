import { useState, useEffect } from "react";
import type { PredictionResponse, TrainResponse } from "../types";
import { fetchPrediction, triggerTrain } from "../lib/api";
import { BacktestChart } from "./BacktestChart";

const FEATURE_NAMES: Record<string, string> = {
    "macro_copper_gold_ratio": "Copper/Gold Ratio",
    "macro_vix_tnx_ratio": "VIX/10Y Yield Ratio",
    "^GSPC_ret_1d": "S&P 500 Daily Return",
    "^GSPC_ret_5d": "S&P 500 5-Way Momentum",
    "^GSPC_ret_21d": "S&P 500 Monthly Momentum",
    "^GSPC_ret_63d": "S&P 500 Quarterly Momentum",
    "^GSPC_vol_21d": "S&P 500 Monthly Volatility",
};

function formatFeatureName(feat: string) {
    if (FEATURE_NAMES[feat]) return FEATURE_NAMES[feat];
    return feat.replace(/_/g, " ").replace("ret", "Return ").replace("vol", "Volatility ");
}

export function ModelInterface() {
    const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
    const [trainRes, setTrainRes] = useState<TrainResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [training, setTraining] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [trainCount, setTrainCount] = useState(0);

    const loadPrediction = async () => {
        try {
            setLoading(true);
            const res = await fetchPrediction();
            if (res.status === "error") {
                setError(res.message || "Failed to load prediction");
            } else {
                setPrediction(res);
                setError(null);
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadPrediction();
    }, []);

    const handleTrain = async () => {
        try {
            setTraining(true);
            setError(null);
            const res = await triggerTrain();
            if (res.status === "error") {
                setError(res.message || "Failed to train model");
            } else {
                setTrainRes(res);
                setTrainCount(c => c + 1); // Trigger backtest chart reload
                // Reload prediction after training
                await loadPrediction();
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setTraining(false);
        }
    };

    return (
        <div className="mt-8">
            <div className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h2 className="text-xl font-bold tracking-tight text-zinc-100">Quantitative Model</h2>
                    <p className="text-sm text-zinc-500">Random Forest 5-Day Directional Forecast</p>
                </div>
                <button
                    onClick={handleTrain}
                    disabled={training}
                    className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed border border-zinc-700 rounded-md text-sm font-medium transition-colors whitespace-nowrap"
                >
                    {training ? "Rebuilding Pipeline..." : "Retrain Model"}
                </button>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-6">
                {error && (
                    <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded text-sm text-red-400">
                        {error}
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Signal Display */}
                    <div className="flex flex-col items-center justify-center p-6 border border-zinc-800 rounded-lg bg-zinc-950">
                        <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-widest mb-4">Live S&P 500 Signal</h3>

                        {loading ? (
                            <div className="animate-pulse h-12 w-32 bg-zinc-800 rounded" />
                        ) : prediction?.prediction ? (
                            <div className="text-center space-y-2">
                                <div className={`text-4xl font-black tracking-tighter ${prediction.prediction === "up" ? "text-emerald-400" : "text-red-400"}`}>
                                    {prediction.prediction === "up" ? "BULLISH" : "BEARISH"}
                                </div>
                                {prediction.probability && (
                                    <div className="text-sm text-zinc-500 font-medium">
                                        {prediction.prediction === "up"
                                            ? `${(prediction.probability * 100).toFixed(1)}% Confidence`
                                            : `${((1 - prediction.probability) * 100).toFixed(1)}% Confidence`
                                        }
                                    </div>
                                )}
                                <div className="text-xs text-zinc-600 mt-2">
                                    As of closing prices: {prediction.date}
                                </div>

                                {prediction.top_features && (
                                    <details className="mt-6 text-left w-full border border-zinc-800 rounded-lg bg-zinc-900/50 shadow-inner overflow-hidden group cursor-pointer">
                                        <summary className="p-3 text-sm text-zinc-400 font-medium hover:text-zinc-300 select-none flex justify-between items-center transition-colors group-open:bg-zinc-800/80">
                                            How does this work?
                                            <span className="text-xs opacity-50 transition-transform group-open:rotate-180">▼</span>
                                        </summary>
                                        <div className="p-4 text-xs text-zinc-400 border-t border-zinc-800/80 space-y-4">
                                            <p className="leading-relaxed">
                                                This signal is dynamically generated by a <strong>Random Forest Classifier</strong>, an ensemble machine learning model that aggregates the independent "votes" of 100 separate decision trees trained on 15 years of daily market history.
                                            </p>
                                            <div className="bg-black/20 p-3 rounded border border-zinc-800/50">
                                                <div className="font-semibold text-zinc-300 mb-2.5 uppercase tracking-wider text-[10px]">Model Drivers (Top 3)</div>
                                                <ul className="space-y-2">
                                                    {prediction.top_features.map((f) => (
                                                        <li key={f.feature} className="flex justify-between items-center">
                                                            <span className="text-zinc-300">{formatFeatureName(f.feature)}</span>
                                                            <span className="font-mono text-[10px] text-emerald-500 bg-emerald-500/10 px-1.5 py-0.5 rounded ml-2">{(f.importance * 100).toFixed(1)}% weight</span>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        </div>
                                    </details>
                                )}
                            </div>
                        ) : (
                            <div className="text-sm text-zinc-500">No prediction available.</div>
                        )}
                    </div>

                    {/* Training Results Display */}
                    <div className="flex flex-col justify-center space-y-4">
                        <h3 className="text-sm font-semibold text-zinc-300">Model Status</h3>

                        {training ? (
                            <div className="text-sm text-zinc-400 animate-pulse">
                                Fetching 15 years of market data and fitting Random Forest... (takes ~15 seconds)
                            </div>
                        ) : trainRes?.metrics ? (
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between border-b border-zinc-800 pb-2">
                                    <span className="text-zinc-500">OOS Accuracy</span>
                                    <span className="font-medium text-emerald-400">{(trainRes.metrics.accuracy * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex justify-between border-b border-zinc-800 pb-2">
                                    <span className="text-zinc-500">OOS Precision</span>
                                    <span className="font-medium text-emerald-400">{(trainRes.metrics.precision * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex justify-between border-b border-zinc-800 pb-2">
                                    <span className="text-zinc-500">Training Samples</span>
                                    <span className="font-medium text-zinc-300">{trainRes.metrics.train_samples}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-zinc-500">Top Driver</span>
                                    <span className="font-medium text-zinc-300">
                                        {trainRes.metrics.top_features[0]?.feature || "N/A"}
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <p className="text-sm text-zinc-500">
                                Click "Retrain Model" to run the backtest pipeline and generate out-of-sample metrics.
                            </p>
                        )}
                    </div>
                </div>

                <BacktestChart triggerKey={trainCount} />
            </div>
        </div>
    );
}
