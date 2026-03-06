import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

from main import CATEGORIES


def build_features():
    db_path = Path(__file__).parent / "market_data.db"
    engine = create_engine(f"sqlite:///{db_path}")

    all_tickers = []
    for cat in CATEGORIES.values():
        for symbol, _ in cat["tickers"]:
            all_tickers.append(symbol)

    # 1. Check existing database
    existing_df = pd.DataFrame()
    last_date = None

    try:
        # Load the existing data
        existing_df = pd.read_sql_table("raw_prices", engine, index_col="Date")
        if not existing_df.empty:
            # Get the max date
            last_date = pd.Timestamp(existing_df.index.max())  # type: ignore
            print(
                f"Loaded {existing_df.shape[0]} existing rows from SQLite database. Last date: {last_date.date()}"
            )
    except ValueError:
        # Table doesn't exist yet
        print("No local database found. Initializing...")

    # 2. Determine what to fetch
    today = datetime.datetime.now().date()
    # Adding a day buffer to ensure we don't skip the current incomplete trading day
    tomorrow = today + datetime.timedelta(days=1)

    fetch_new = False

    if last_date is None:
        print("Fetching historical data for 15 years from Yahoo Finance...")
        data = yf.download(all_tickers, period="15y", progress=True, threads=False)
        fetch_new = True
    else:
        last_date_date = pd.Timestamp(last_date).date()  # type: ignore
        if pd.notna(last_date_date) and last_date_date < today:  # type: ignore
            print(f"Fetching missing data from {last_date_date} to {tomorrow}...")
            # We fetch from last_date to tomorrow just to be safe, then we'll deduplicate
            data = yf.download(
                all_tickers,
                start=last_date_date,
                end=tomorrow,
                progress=False,
                threads=False,
            )
            fetch_new = True
        else:
            print("Database is already up to date for today.")

    if fetch_new:
        df_data: pd.DataFrame = data  # type: ignore
        if "Adj Close" in df_data.columns.levels[0]:  # type: ignore
            close_df = df_data["Adj Close"]
        else:
            close_df = df_data["Close"]

        # Forward fill the new data chunk
        close_df = close_df.ffill()

        # Combine with existing data
        if not existing_df.empty:
            # We might have overlap due to the buffer, so we update existing rows and append new ones
            combined_df = pd.concat([existing_df, close_df])
            # Drop duplicates by index, keeping the latest fetched data
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]  # type: ignore
        else:
            combined_df = close_df

        # Ensure the index is named 'Date' for SQLite
        combined_df.index.name = "Date"  # type: ignore

        # Sort chronologically
        combined_df = combined_df.sort_index()  # type: ignore

        # Save the full combined dataset back to SQLite (replace allows us to cleanly handle schema changes if we add tickers)
        print(f"Writing {combined_df.shape[0]} rows to SQLite...")
        combined_df.to_sql("raw_prices", engine, if_exists="replace", index=True)

        close_df = combined_df
    else:
        close_df = existing_df

    # Create a new DataFrame for our features
    features = pd.DataFrame(index=close_df.index)  # type: ignore

    print("Computing features...")
    for symbol in all_tickers:
        # Avoid issues where data couldn't be downloaded at all
        if symbol not in close_df.columns or close_df[symbol].isna().all():  # type: ignore
            print(f"Warning: No valid data for {symbol}, skipping.")
            continue

        prices = close_df[symbol]

        # Returns
        features[f"{symbol}_ret_1d"] = prices.pct_change(periods=1)  # type: ignore
        features[f"{symbol}_ret_5d"] = prices.pct_change(periods=5)  # type: ignore
        features[f"{symbol}_ret_21d"] = prices.pct_change(periods=21)  # type: ignore
        features[f"{symbol}_ret_63d"] = prices.pct_change(periods=63)  # type: ignore

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
