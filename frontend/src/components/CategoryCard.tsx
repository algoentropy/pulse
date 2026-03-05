import type { CategoryData, HistoryResponse } from "../types";
import { CATEGORY_TOOLTIPS } from "../lib/tooltips";
import { Tooltip } from "./Tooltip";
import { TickerRow } from "./TickerRow";

interface CategoryCardProps {
  categoryKey: string;
  data: CategoryData;
  history?: HistoryResponse;
  summary?: string;
}

export function CategoryCard({ categoryKey, data, history, summary }: CategoryCardProps) {
  const tooltip = CATEGORY_TOOLTIPS[categoryKey];
  const header = (
    <div>
      <h2 className="text-lg font-semibold text-zinc-100">{data.label}</h2>
      <p className="text-xs text-zinc-500">{data.subtitle}</p>
    </div>
  );

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-4 overflow-hidden">
      <div className="mb-3 border-b border-zinc-800 pb-2">
        {tooltip ? <Tooltip text={tooltip}>{header}</Tooltip> : header}
      </div>
      {summary && (
        <p className="text-xs text-zinc-400 italic pt-1 pb-2">{summary}</p>
      )}
      <div className="space-y-0.5">
        {data.tickers.map((t) => (
          <TickerRow key={t.ticker} entry={t} history={history?.[t.ticker]} />
        ))}
      </div>
    </div>
  );
}
