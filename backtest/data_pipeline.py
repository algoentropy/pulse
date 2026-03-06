import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from main import CATEGORIES


def build_features():
    print("Fetching historical data for 15 years...")
    all_tickers = []
    for cat in CATEGORIES.values():
        for symbol, _ in cat["tickers"]:
            all_tickers.append(symbol)

    # Download 15 years of daily data
    # group_by='column' is default, which gives a MultiIndex columns
    data = yf.download(all_tickers, period="15y", progress=True)

    # We only care about the Adjusted Close or Close price
    if "Adj Close" in data.columns.levels[0]:
        close_df = data["Adj Close"]
    else:
        close_df = data["Close"]

    # Forward fill missing values
    close_df = close_df.ffill()

    print(f"Downloaded shape: {close_df.shape}")

    # Create a new DataFrame for our features
    features = pd.DataFrame(index=close_df.index)

    print("Computing features...")
    for symbol in all_tickers:
        # Avoid issues where data couldn't be downloaded at all
        if symbol not in close_df.columns or close_df[symbol].isna().all():
            print(f"Warning: No valid data for {symbol}, skipping.")
            continue

        prices = close_df[symbol]

        # Returns
        features[f"{symbol}_ret_1d"] = prices.pct_change(periods=1)
        features[f"{symbol}_ret_5d"] = prices.pct_change(periods=5)
        features[f"{symbol}_ret_21d"] = prices.pct_change(periods=21)
        features[f"{symbol}_ret_63d"] = prices.pct_change(periods=63)

        # Volatility (21-day rolling std dev of daily returns)
        features[f"{symbol}_vol_21d"] = (
            features[f"{symbol}_ret_1d"].rolling(window=21).std()
        )

    # Macro Ratios
    # Copper/Gold Ratio (HG=F / GC=F)
    if "HG=F" in close_df.columns and "GC=F" in close_df.columns:
        features["macro_copper_gold_ratio"] = close_df["HG=F"] / close_df["GC=F"]
        # And its momentum
        features["macro_copper_gold_ratio_ret_21d"] = features[
            "macro_copper_gold_ratio"
        ].pct_change(periods=21)

    # VIX/10Y Ratio (^VIX / ^TNX)
    if "^VIX" in close_df.columns and "^TNX" in close_df.columns:
        # ^TNX is quoted in % directly (e.g., 4.2), so dividing VIX directly works as a crude ratio
        features["macro_vix_tnx_ratio"] = close_df["^VIX"] / close_df["^TNX"]
        features["macro_vix_tnx_ratio_ret_21d"] = features[
            "macro_vix_tnx_ratio"
        ].pct_change(periods=21)

    # Drop the rows at the start that have NaNs due to the 63-day lookback, but keep recent ones
    features = features.dropna(
        subset=[col for col in features.columns if "ret_63d" in col], how="all"
    )

    # Final forward fill for any weird gaps
    features = features.ffill()

    # And fill remaining NaNs with 0 (e.g. if some ticker only listed 5 years ago)
    # Be careful here though, as 0 means no change
    features = features.fillna(0)

    print(f"Final feature matrix shape: {features.shape}")

    output_file = Path(__file__).parent / "macro_features.parquet"
    features.to_parquet(output_file)
    print(f"Successfully saved {output_file}")


if __name__ == "__main__":
    build_features()
