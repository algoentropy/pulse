import type { CategoryData, HistoryResponse } from "../types";
import { CATEGORY_TOOLTIPS } from "../lib/tooltips";
import { Tooltip } from "./Tooltip";
import { TickerRow } from "./TickerRow";

interface CategoryCardProps {
  categoryKey: string;
  data: CategoryData;
  history?: HistoryResponse;
  summary?: string;
  summaryLoading?: boolean;
}

export function CategoryCard({ categoryKey, data, history, summary, summaryLoading }: CategoryCardProps) {
  const tooltip = CATEGORY_TOOLTIPS[categoryKey];
  const header = (
    <div>
      <h2 className="text-lg font-semibold text-zinc-100">{data.label}</h2>
      <p className="text-xs text-zinc-500">{data.subtitle}</p>
    </div>
  );

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-4">
      <div className="mb-3 border-b border-zinc-800 pb-2">
        {tooltip ? <Tooltip text={tooltip}>{header}</Tooltip> : header}
      </div>
      {summaryLoading && !summary && (
        <div className="pt-1 pb-2 space-y-1.5">
          <div className="animate-pulse rounded bg-zinc-800/60 h-2.5 w-5/6" />
          <div className="animate-pulse rounded bg-zinc-800/60 h-2.5 w-2/3" />
        </div>
      )}
      {summary && (
        <p className="text-xs text-zinc-400 italic pt-1 pb-2">{summary}</p>
      )}
      <div className="space-y-0.5 overflow-hidden">
        {data.tickers.map((t) => (
          <TickerRow key={t.ticker} entry={t} history={history?.[t.ticker]} />
        ))}
      </div>
    </div>
  );
}
