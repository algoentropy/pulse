const DECIMAL_MAP: Record<string, number> = {
  "^TNX": 3,
  "BTC-USD": 0,
  "CHF=X": 4,
};

export function formatPrice(ticker: string, price: number): string {
  const decimals = DECIMAL_MAP[ticker] ?? 2;
  return price.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatChange(pct: number): string {
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}
