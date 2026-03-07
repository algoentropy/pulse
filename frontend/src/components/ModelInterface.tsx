import { useState, useEffect } from "react";
import type { PredictionResponse, TrainResponse } from "../types";
import { fetchPrediction, triggerTrain, triggerSimulation } from "../lib/api";
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

    const [simulationMode, setSimulationMode] = useState(false);
    const [overrides, setOverrides] = useState<Record<string, number>>({});
    const [simPrediction, setSimPrediction] = useState<PredictionResponse | null>(null);
    const [simLoading, setSimLoading] = useState(false);

    // Initialize overrides when prediction loads or simulation mode toggled
    useEffect(() => {
        if (simulationMode && prediction?.top_features) {
            const initial: Record<string, number> = {};
            prediction.top_features.forEach(f => {
                initial[f.feature] = f.current_value;
            });
            setOverrides(initial);
            setSimPrediction(prediction);
        } else {
            setSimPrediction(null);
        }
    }, [simulationMode, prediction]);

    // Debounced simulation
    useEffect(() => {
        if (!simulationMode || Object.keys(overrides).length === 0) return;

        setSimLoading(true);
        const timer = setTimeout(async () => {
            try {
                const res = await triggerSimulation(overrides);
                if (res.status !== "error") {
                    setSimPrediction(res);
                }
            } catch (e) {
                console.error(e);
            } finally {
                setSimLoading(false);
            }
        }, 150);

        return () => clearTimeout(timer);
    }, [overrides, simulationMode]);

    const displayPrediction = simulationMode ? (simPrediction || prediction) : prediction;

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
                        ) : displayPrediction?.prediction ? (
                            <div className="text-center space-y-2 relative w-full">
                                {simulationMode && simLoading && (
                                    <div className="absolute -top-4 right-0 flex items-center gap-1.5 text-indigo-400 text-[10px] uppercase font-bold tracking-wider animate-pulse">
                                        <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"></div> Simulating
                                    </div>
                                )}
                                <div className={`text-4xl font-black tracking-tighter transition-colors duration-300 ${displayPrediction.prediction === "up" ? "text-emerald-400" : "text-red-400"} ${simulationMode && simLoading ? 'opacity-50' : 'opacity-100'}`}>
                                    {displayPrediction.prediction === "up" ? "BULLISH" : "BEARISH"}
                                </div>
                                {displayPrediction.probability && (
                                    <div className={`text-sm font-medium transition-opacity duration-300 ${simulationMode ? 'text-indigo-400' : 'text-zinc-500'} ${simulationMode && simLoading ? 'opacity-50' : 'opacity-100'}`}>
                                        {displayPrediction.prediction === "up"
                                            ? `${(displayPrediction.probability * 100).toFixed(1)}% Confidence`
                                            : `${((1 - displayPrediction.probability) * 100).toFixed(1)}% Confidence`
                                        }
                                        {simulationMode && " (Simulated)"}
                                    </div>
                                )}
                                <div className="text-xs text-zinc-600 mt-2">
                                    {simulationMode ? "HYPOTHETICAL SCENARIO" : `As of closing prices: ${displayPrediction.date}`}
                                </div>

                                {prediction?.top_features && (
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
                                                <div className="flex justify-between items-center mb-2.5">
                                                    <div className="font-semibold text-zinc-300 uppercase tracking-wider text-[10px]">Model Drivers (Top 20)</div>
                                                    <label className="flex items-center gap-1.5 text-[10px] uppercase font-bold tracking-wider cursor-pointer text-zinc-400 hover:text-zinc-300">
                                                        <input
                                                            type="checkbox"
                                                            checked={simulationMode}
                                                            onChange={(e) => setSimulationMode(e.target.checked)}
                                                            className="rounded border-zinc-700 bg-zinc-800 text-indigo-500 focus:ring-indigo-500 focus:ring-offset-zinc-900"
                                                        />
                                                        Simulation Mode
                                                    </label>
                                                </div>
                                                <ul className="space-y-3 max-h-72 overflow-y-auto pr-2 custom-scrollbar">
                                                    {prediction?.top_features.map((f) => (
                                                        <li key={f.feature} className="flex flex-col">
                                                            <div className="flex justify-between items-center">
                                                                <span className="text-zinc-300">{formatFeatureName(f.feature)}</span>
                                                                <span className="font-mono text-[10px] text-emerald-500 bg-emerald-500/10 px-1.5 py-0.5 rounded ml-2">{(f.importance * 100).toFixed(1)}%</span>
                                                            </div>
                                                            {simulationMode && (
                                                                <div className="mt-2 flex items-center gap-3">
                                                                    <span className="text-[10px] text-zinc-500 font-mono w-8 text-right">{f.min.toFixed(2)}</span>
                                                                    <input
                                                                        type="range"
                                                                        min={f.min}
                                                                        max={f.max}
                                                                        step={(f.max - f.min) / 100}
                                                                        value={overrides[f.feature] ?? f.current_value}
                                                                        onChange={(e) => setOverrides(prev => ({ ...prev, [f.feature]: parseFloat(e.target.value) }))}
                                                                        className="flex-1 h-1.5 bg-zinc-700 rounded-lg appearance-none cursor-ew-resize accent-indigo-500"
                                                                    />
                                                                    <span className="text-[10px] text-zinc-500 font-mono w-8">{f.max.toFixed(2)}</span>
                                                                    <span className="text-[10px] font-mono font-bold text-zinc-200 w-12 text-right bg-zinc-800 px-1 py-0.5 rounded">{(overrides[f.feature] ?? f.current_value).toFixed(3)}</span>
                                                                </div>
                                                            )}
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
