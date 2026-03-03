import type { PulseResponse } from "../types";
import { CategoryCard } from "./CategoryCard";

interface DashboardProps {
  data: PulseResponse;
}

const CATEGORY_KEYS = ["vitals", "muscles", "scoreboard", "geopolitics"] as const;

export function Dashboard({ data }: DashboardProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {CATEGORY_KEYS.map((key) => (
        <CategoryCard key={key} categoryKey={key} data={data[key]} />
      ))}
    </div>
  );
}
