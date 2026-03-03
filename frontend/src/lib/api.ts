import type { PulseResponse } from "../types";

export async function fetchPulse(): Promise<PulseResponse> {
  const res = await fetch("/api/pulse");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
