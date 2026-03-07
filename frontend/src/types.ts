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
  top_features?: Array<{
    feature: string;
    importance: number;
    current_value: number;
    min: number;
    max: number;
  }>;
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

export interface AnalogueFeatureDiff {
  feature: string;
  ref_value: number;
  analogue_value: number;
  abs_diff: number;
}

export interface AnalogueResult {
  rank: number;
  start_date: string;
  end_date: string;
  combined_score: number;
  macro_score: number;
  price_score: number;
  forward_returns: {
    "5d"?: number;
    "21d"?: number;
    "63d"?: number;
  };
  sp500_prices: Array<{ time: string; value: number }>;
  top_feature_diffs: AnalogueFeatureDiff[];
}

export interface AnaloguesResponse {
  status: string;
  reference_date: string;
  window_days: number;
  macro_weight: number;
  price_weight: number;
  n_candidates_scored: number;
  reference_prices: Array<{ time: string; value: number }>;
  analogues: AnalogueResult[];
}
