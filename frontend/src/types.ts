export interface TickerEntry {
  name: string;
  ticker: string;
  price: number | null;
  change_pct: number | null;
  status: string;
}

export interface CategoryData {
  label: string;
  subtitle: string;
  tickers: TickerEntry[];
}

export interface PulseResponse {
  vitals: CategoryData;
  muscles: CategoryData;
  scoreboard: CategoryData;
  geopolitics: CategoryData;
}
