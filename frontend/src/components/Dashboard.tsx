import type { PulseResponse, HistoryResponse, InterpretationResponse } from "../types";
import { CategoryCard } from "./CategoryCard";

interface DashboardProps {
  data: PulseResponse;
  history?: HistoryResponse;
  interpretation?: InterpretationResponse;
}

const CATEGORY_KEYS = ["vitals", "muscles", "scoreboard", "geopolitics"] as const;

export function Dashboard({ data, history, interpretation }: DashboardProps) {
  return (
    <div>
      {interpretation?.overall && (
        <div className="mb-4 rounded-xl border border-zinc-800 bg-zinc-900/80 p-4">
          <p className="text-sm text-zinc-300 italic">{interpretation.overall}</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {CATEGORY_KEYS.map((key) => (
          <CategoryCard
            key={key}
            categoryKey={key}
            data={data[key]}
            history={history}
            summary={interpretation?.categories[key]}
          />
        ))}
      </div>
    </div>
  );
}
