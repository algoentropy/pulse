import type { TickerEntry } from "../types";
import { formatPrice, formatChange } from "../lib/format";
import { TICKER_TOOLTIPS } from "../lib/tooltips";
import { Tooltip } from "./Tooltip";

interface TickerRowProps {
  entry: TickerEntry;
}

export function TickerRow({ entry }: TickerRowProps) {
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
    <div className="flex items-center justify-between py-1.5 px-1 hover:bg-white/5 rounded transition-colors">
      <div>{tooltip ? <Tooltip text={tooltip}>{nameEl}</Tooltip> : nameEl}</div>
      <div className="flex items-center gap-3 tabular-nums">
        <span className="text-sm text-zinc-300">{formatPrice(ticker, price)}</span>
        <span className={`text-sm font-medium w-20 text-right ${colorClass}`}>
          {formatChange(change_pct)}
        </span>
      </div>
    </div>
  );
}
