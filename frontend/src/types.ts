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

export interface HistoryPoint {
  time: string;
  value: number;
}

export type HistoryResponse = Record<string, HistoryPoint[]>;

export interface InterpretationResponse {
  categories: Record<string, string>;
  overall: string;
}

export interface FeaturePoint {
  time: string;
  value: number;
}

export interface FeaturesResponse {
  copper_gold: FeaturePoint[];
  vix_tnx: FeaturePoint[];
}
