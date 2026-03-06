import { useEffect, useState } from "react";
import type { PulseResponse, HistoryResponse, InterpretationResponse, FeaturesResponse } from "./types";
import { fetchPulse, fetchHistory, fetchInterpretation, fetchFeatures } from "./lib/api";
import { Dashboard } from "./components/Dashboard";
import { MacroSignals } from "./components/MacroSignals";
import { ModelInterface } from "./components/ModelInterface";

export default function App() {
  const [data, setData] = useState<PulseResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | undefined>();
  const [interpretation, setInterpretation] = useState<InterpretationResponse | undefined>();
  const [features, setFeatures] = useState<FeaturesResponse | undefined>();
  const [interpretationLoading, setInterpretationLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tutorMode, setTutorMode] = useState<"executive" | "beginner">("executive");

  const loadData = (refresh: boolean = false) => {
    setLoading(true);
    setError(null);

    fetchPulse(refresh)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    fetchHistory(refresh)
      .then(setHistory)
      .catch(() => { });

    setInterpretationLoading(true);
    fetchInterpretation(refresh, tutorMode)
      .then(setInterpretation)
      .catch(() => { })
      .finally(() => setInterpretationLoading(false));
  };

  useEffect(() => {
    setInterpretationLoading(true);
    fetchInterpretation(false, tutorMode)
      .then(setInterpretation)
      .catch(() => { })
      .finally(() => setInterpretationLoading(false));
  }, [tutorMode]);

  useEffect(() => {
    loadData();

    fetchFeatures()
      .then(setFeatures)
      .catch((e) => console.error("Error fetching features:", e));
  }, []);

  const handleRefresh = () => {
    loadData(true);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-3 sm:p-6 md:p-8">
      <header className="max-w-5xl mx-auto mb-6 flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Global Pulse</h1>
          <p className="text-sm text-zinc-500">Real-time macro dashboard</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="text-sm px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-200 rounded transition-colors"
        >
          {loading ? "Refreshing..." : "Refresh Data"}
        </button>
      </header>
      <main className="max-w-5xl mx-auto space-y-8">
        {loading && (
          <p className="text-zinc-500 text-center py-20">Loading market data...</p>
        )}
        {error && (
          <p className="text-red-400 text-center py-20">Error: {error}</p>
        )}

        {data && <Dashboard
          data={data}
          history={history}
          interpretation={interpretation}
          interpretationLoading={interpretationLoading}
          tutorMode={tutorMode}
          setTutorMode={setTutorMode}
        />}

        {features && <MacroSignals data={features} />}

        {!loading && !error && <ModelInterface />}
      </main>
    </div>
  );
}

