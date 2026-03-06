import { useEffect, useState } from "react";
import type { PulseResponse, HistoryResponse, InterpretationResponse } from "./types";
import { fetchPulse, fetchHistory, fetchInterpretation } from "./lib/api";
import { Dashboard } from "./components/Dashboard";

export default function App() {
  const [data, setData] = useState<PulseResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | undefined>();
  const [interpretation, setInterpretation] = useState<InterpretationResponse | undefined>();
  const [interpretationLoading, setInterpretationLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPulse()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    fetchHistory()
      .then(setHistory)
      .catch(() => { });

    fetchInterpretation()
      .then(setInterpretation)
      .catch(() => { })
      .finally(() => setInterpretationLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-3 sm:p-6 md:p-8">
      <header className="max-w-5xl mx-auto mb-6">
        <h1 className="text-2xl font-bold tracking-tight">Global Pulse</h1>
        <p className="text-sm text-zinc-500">Real-time macro dashboard</p>
      </header>
      <main className="max-w-5xl mx-auto">
        {loading && (
          <p className="text-zinc-500 text-center py-20">Loading market data...</p>
        )}
        {error && (
          <p className="text-red-400 text-center py-20">Error: {error}</p>
        )}
        {data && <Dashboard data={data} history={history} interpretation={interpretation} interpretationLoading={interpretationLoading} />}
      </main>
    </div>
  );
}
