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
