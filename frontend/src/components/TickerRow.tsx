import type { TickerEntry, HistoryPoint } from "../types";
import { formatPrice, formatChange } from "../lib/format";
import { TICKER_TOOLTIPS } from "../lib/tooltips";
import { Tooltip } from "./Tooltip";
import { Sparkline } from "./Sparkline";

interface TickerRowProps {
  entry: TickerEntry;
  history?: HistoryPoint[];
}

export function TickerRow({ entry, history }: TickerRowProps) {
  const { name, ticker, price, change_pct, status } = entry;

  if (status !== "active" || price == null || change_pct == null) {
    return (
      <div className="flex items-center justify-between py-1.5 px-1 text-zinc-500">
        <span className="text-sm">{name}</span>
        <span className="text-xs italic">{status === "closed" ? "Closed" : "N/A"}</span>
      </div>
    );
  }

  const colorClass =
    change_pct > 0
      ? "text-emerald-400"
      : change_pct < 0
        ? "text-red-400"
        : "text-zinc-400";

  const tooltip = TICKER_TOOLTIPS[ticker];
  const nameEl = <span className="text-sm text-zinc-200">{name}</span>;

  return (
    <div className="flex items-center gap-2 py-1.5 px-1 hover:bg-white/5 rounded transition-colors">
      <div className="w-40 shrink-0">
        {tooltip ? <Tooltip text={tooltip}>{nameEl}</Tooltip> : nameEl}
      </div>
      <div className="flex-1 flex justify-center">
        {history && history.length >= 2 && <Sparkline data={history} />}
      </div>
      <div className="flex items-center gap-3 tabular-nums shrink-0">
        <span className="text-sm text-zinc-300">{formatPrice(ticker, price)}</span>
        <span className={`text-sm font-medium w-20 text-right ${colorClass}`}>
          {formatChange(change_pct)}
        </span>
      </div>
    </div>
  );
}
