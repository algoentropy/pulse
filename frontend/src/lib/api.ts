import type { PulseResponse, HistoryResponse, InterpretationResponse } from "../types";

export async function fetchPulse(): Promise<PulseResponse> {
  const res = await fetch("/api/pulse");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchHistory(): Promise<HistoryResponse> {
  const res = await fetch("/api/history");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchInterpretation(): Promise<InterpretationResponse> {
  const res = await fetch("/api/interpretation");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchFeatures(): Promise<import("../types").FeaturesResponse> {
  const res = await fetch("/api/features");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchPrediction(): Promise<import("../types").PredictionResponse> {
  const res = await fetch("/api/predict");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function triggerTrain(): Promise<import("../types").TrainResponse> {
  const res = await fetch("/api/train", { method: "POST" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchBacktest(): Promise<import("../types").BacktestResponse> {
  const res = await fetch("/api/backtest");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
