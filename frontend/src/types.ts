export interface TickerEntry {
  name: string;
  ticker: string;
  description?: string;
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

export interface PredictionResponse {
  status: string;
  message?: string;
  prediction?: "up" | "down";
  probability?: number;
  date?: string;
  top_features?: Array<{ feature: string; importance: number }>;
}

export interface TrainResponse {
  status: string;
  message: string;
  metrics?: {
    accuracy: number;
    precision: number;
    train_samples: number;
    test_samples: number;
    top_features: Array<{ feature: string; importance: number }>;
  };
}

export interface BacktestResponse {
  strategy: FeaturePoint[];
  benchmark: FeaturePoint[];
}
