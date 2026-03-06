import { Fragment } from "react";
import type { PulseResponse, HistoryResponse, InterpretationResponse, HistoryPoint } from "../types";
import { CATEGORY_TOOLTIPS, TICKER_TOOLTIPS } from "../lib/tooltips";
import { formatPrice, formatChange } from "../lib/format";
import { Tooltip } from "./Tooltip";
import { Sparkline } from "./Sparkline";
import { Info } from "lucide-react";

interface DashboardProps {
  data: PulseResponse;
  history?: HistoryResponse;
  interpretation?: InterpretationResponse;
  interpretationLoading?: boolean;
  tutorMode?: "executive" | "beginner";
  setTutorMode?: (mode: "executive" | "beginner") => void;
}

const CATEGORY_KEYS = ["vitals", "muscles", "scoreboard", "geopolitics"] as const;
const COL_SPAN = 6;

function Shimmer({ className = "" }: { className?: string }) {
  return (
    <div className={`animate-pulse rounded bg-zinc-800/60 ${className}`} />
  );
}

function changeFromHistory(points: HistoryPoint[], tradingDays: number): number | null {
  if (!points || points.length < 2) return null;
  const current = points[points.length - 1].value;
  const idx = Math.max(0, points.length - 1 - tradingDays);
  const past = points[idx].value;
  if (past === 0) return null;
  return ((current - past) / past) * 100;
}

function changeColor(val: number | null): string {
  if (val == null) return "text-zinc-600";
  if (val > 0) return "text-emerald-400";
  if (val < 0) return "text-red-400";
  return "text-zinc-400";
}

function ChangeCell({ value, className = "" }: { value: number | null; className?: string }) {
  return (
    <td className={`py-1.5 px-3 text-sm font-medium text-right tabular-nums ${changeColor(value)} ${className}`}>
      {value != null ? formatChange(value) : "—"}
    </td>
  );
}

export function Dashboard({ data, history, interpretation, interpretationLoading, tutorMode, setTutorMode }: DashboardProps) {
  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-zinc-100">Market Pulse</h2>
        {setTutorMode && (
          <div className="flex bg-zinc-900 border border-zinc-800 p-1 rounded-lg">
            <button
              onClick={() => setTutorMode("executive")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${tutorMode === "executive" ? "bg-zinc-800 text-zinc-100 shadow-sm" : "text-zinc-500 hover:text-zinc-300"}`}
            >
              Executive Summary
            </button>
            <button
              onClick={() => setTutorMode("beginner")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${tutorMode === "beginner" ? "bg-indigo-500/10 text-indigo-400 shadow-sm" : "text-zinc-500 hover:text-zinc-300"}`}
            >
              Beginner Breakdown
            </button>
          </div>
        )}
      </div>

      {interpretationLoading && !interpretation?.overall && (
        <div className="mb-4 rounded-xl border border-zinc-800 bg-zinc-900/80 p-4 space-y-2">
          <Shimmer className="h-3 w-3/4" />
          <Shimmer className="h-3 w-1/2" />
        </div>
      )}
      {interpretation?.overall && (
        <div className="mb-4 rounded-xl border border-zinc-800 bg-zinc-900/80 p-4">
          <p className="text-sm text-zinc-300 italic">{interpretation.overall}</p>
        </div>
      )}

      <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-zinc-800 text-xs text-zinc-500 uppercase tracking-wider">
              <th className="text-left py-2 px-2 sm:px-4 font-medium">Ticker</th>
              <th className="text-right py-2 px-2 sm:px-4 font-medium">Last</th>
              <th className="text-right py-2 px-2 sm:px-3 font-medium">1D</th>
              <th className="text-right py-2 px-2 sm:px-3 font-medium">1W</th>
              <th className="text-right py-2 px-2 sm:px-3 font-medium">1M</th>
              <th className="text-center py-2 px-2 sm:px-4 font-medium hidden sm:table-cell">Chart</th>
            </tr>
          </thead>
          <tbody>
            {CATEGORY_KEYS.map((key) => {
              const cat = data[key];
              const catTooltip = CATEGORY_TOOLTIPS[key];
              const summary = interpretation?.categories[key];
              const summaryLoading = interpretationLoading && !interpretation;

              return (
                <Fragment key={key}>
                  <tr className="border-t border-zinc-700/60">
                    <td colSpan={COL_SPAN} className="px-4 pt-4 pb-1">
                      {catTooltip ? (
                        <Tooltip text={catTooltip}>
                          <div>
                            <span className="text-lg font-semibold text-zinc-100">{cat.label}</span>
                            <span className="ml-2 text-xs text-zinc-500">{cat.subtitle}</span>
                          </div>
                        </Tooltip>
                      ) : (
                        <div>
                          <span className="text-lg font-semibold text-zinc-100">{cat.label}</span>
                          <span className="ml-2 text-xs text-zinc-500">{cat.subtitle}</span>
                        </div>
                      )}
                    </td>
                  </tr>

                  {summaryLoading && !summary && (
                    <tr>
                      <td colSpan={COL_SPAN} className="px-4 pb-2">
                        <div className="space-y-1.5">
                          <div className="animate-pulse rounded bg-zinc-800/60 h-2.5 w-5/6" />
                          <div className="animate-pulse rounded bg-zinc-800/60 h-2.5 w-2/3" />
                        </div>
                      </td>
                    </tr>
                  )}
                  {summary && (
                    <tr>
                      <td colSpan={COL_SPAN} className="px-4 pb-2">
                        <p className="text-xs text-zinc-400 italic">{summary}</p>
                      </td>
                    </tr>
                  )}

                  {cat.tickers.map((t) => {
                    const closed = t.status !== "active" || t.price == null || t.change_pct == null;

                    if (closed) {
                      return (
                        <tr key={t.ticker} className="text-zinc-600">
                          <td className="py-1.5 px-2 sm:px-4 text-sm max-w-[80px] sm:max-w-none truncate">
                            <span className="hidden sm:inline">{t.name}</span>
                            <span className="sm:hidden">{t.ticker}</span>
                          </td>
                          <td colSpan={COL_SPAN - 1} className="py-1.5 px-2 sm:px-4 text-xs italic text-right">
                            Market closed
                          </td>
                        </tr>
                      );
                    }

                    const historyData = history?.[t.ticker];
                    const chg1D = t.change_pct;
                    const chg1W = historyData ? changeFromHistory(historyData, 5) : null;
                    const chg1M = historyData ? changeFromHistory(historyData, 21) : null;

                    const tickerTooltip = t.description || TICKER_TOOLTIPS[t.ticker];
                    const tooltipText = tickerTooltip ? `${t.name}: ${tickerTooltip}` : t.name;
                    const nameEl = (
                      <div className="flex items-center gap-1.5">
                        <span className="hidden sm:inline text-sm text-zinc-200">{t.name}</span>
                        <span className="sm:hidden text-sm text-zinc-200 font-medium">{t.ticker}</span>
                        {tickerTooltip && (
                          <Tooltip text={tooltipText}>
                            <Info className="w-3.5 h-3.5 text-zinc-500 hover:text-zinc-300 transition-colors cursor-help" />
                          </Tooltip>
                        )}
                      </div>
                    );

                    return (
                      <tr key={t.ticker} className="hover:bg-white/5 transition-colors">
                        <td className="py-1.5 px-2 sm:px-4 truncate max-w-[120px] sm:max-w-none">
                          {nameEl}
                        </td>
                        <td className="py-1.5 px-2 sm:px-4 text-sm text-zinc-300 text-right tabular-nums">
                          {formatPrice(t.ticker, t.price!)}
                        </td>
                        <ChangeCell value={chg1D} />
                        <ChangeCell value={chg1W} />
                        <ChangeCell value={chg1M} />
                        <td className="py-1.5 px-2 sm:px-4 hidden sm:table-cell">
                          <div className="flex justify-center">
                            {historyData && historyData.length >= 2 && <Sparkline data={historyData} />}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
