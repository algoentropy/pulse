import type { PulseResponse, HistoryResponse, InterpretationResponse } from "../types";

export async function fetchPulse(refresh: boolean = false): Promise<PulseResponse> {
  const url = refresh ? "/api/pulse?refresh=true" : "/api/pulse";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchHistory(refresh: boolean = false): Promise<HistoryResponse> {
  const url = refresh ? "/api/history?refresh=true" : "/api/history";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchInterpretation(refresh: boolean = false, mode: "executive" | "beginner" = "executive"): Promise<InterpretationResponse> {
  const params = new URLSearchParams();
  if (refresh) params.append("refresh", "true");
  if (mode !== "executive") params.append("mode", mode);

  const qs = params.toString();
  const res = await fetch(`/api/interpretation${qs ? "?" + qs : ""}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchFeatures(): Promise<import("../types").FeaturesResponse> {
  const res = await fetch("/api/features");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchPrediction(date?: string): Promise<import("../types").PredictionResponse> {
  const url = date ? `/api/predict?date=${date}` : "/api/predict";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function triggerTrain(): Promise<import("../types").TrainResponse> {
  const res = await fetch("/api/train", { method: "POST" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function triggerSimulation(overrides: Record<string, number>, date?: string): Promise<import("../types").PredictionResponse> {
  const body: any = { overrides };
  if (date) body.date = date;

  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchBacktest(): Promise<import("../types").BacktestResponse> {
  const res = await fetch("/api/backtest");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
