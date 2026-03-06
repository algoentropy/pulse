import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

TARGET_TICKER = "^GSPC"
FORWARD_DAYS = 5


def visualize_model():
    base_dir = Path(__file__).parent
    features_path = base_dir / "macro_features.parquet"
    model_path = base_dir / "rf_model.pkl"

    if not features_path.exists() or not model_path.exists():
        print("Missing required files. Please run data_pipeline.py and model.py first.")
        return

    print("Loading data and model...")
    df = pd.read_parquet(features_path)
    clf = joblib.load(model_path)

    # 1. Recreate target and features appropriately (same as model.py)
    future_return_col = f"{TARGET_TICKER}_ret_{FORWARD_DAYS}d_future"
    df[future_return_col] = df[f"{TARGET_TICKER}_ret_{FORWARD_DAYS}d"].shift(
        -FORWARD_DAYS
    )
    df = df.dropna(subset=[future_return_col])

    df["target"] = (df[future_return_col] > 0.0).astype(int)

    # Exclude non-features
    exclude_cols = [future_return_col, "target"]
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and "future" not in col.lower()
    ]

    X = df[feature_cols]

    # We must split identically to model.py
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]

    # The actual forward 5-day return mapped back to the daily row
    # To simulate holding for the *next* 5 days, we use the future return we aligned in `df`
    # However, since this is an overlapping 5-day return prediction,
    # to perfectly simulate daily returns, we need the actual 1-day future return.

    # Create the 1-day future return for accurate daily compounding simulation
    daily_future_return_col = f"{TARGET_TICKER}_ret_1d_future"
    df[daily_future_return_col] = df[f"{TARGET_TICKER}_ret_1d"].shift(-1)

    # Re-slice the test set based on the same index so we have the daily future returns
    test_df = df.iloc[split_idx:].copy()

    # Predict the target (1 = S&P going up over next 5 days, 0 = going down)
    print("Generating predictions...")
    test_df["predicted_signal"] = clf.predict(X_test)

    # --- Simulate Strategy ---
    # Rule: If prediction is 1, we hold the S&P 500 for tomorrow.
    #       If prediction is 0, we hold Cash (0% return) for tomorrow.
    # Note: We are using a 5-day prediction to hold for 1-day.
    # In practice, if we predicted 5 days up, we might hold. This is a simplified proxy.

    test_df["strategy_daily_return"] = test_df.apply(
        lambda row: row[daily_future_return_col]
        if row["predicted_signal"] == 1
        else 0.0,
        axis=1,
    )

    # Benchmark: Buy and Hold the S&P 500
    test_df["benchmark_daily_return"] = test_df[daily_future_return_col]

    # Calculate cumulative returns
    # +1 converts e.g. 0.01 to 1.01. cumprod multiplies them all together over time.
    test_df["strategy_cumulative"] = (1 + test_df["strategy_daily_return"]).cumprod()
    test_df["benchmark_cumulative"] = (1 + test_df["benchmark_daily_return"]).cumprod()

    # Drop NAs at the very end
    test_df = test_df.dropna(subset=["strategy_cumulative", "benchmark_cumulative"])

    # --- Plotting 1: Equity Curve ---
    print("Plotting Equity Curve...")
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 6))
    plt.plot(
        test_df.index,
        test_df["strategy_cumulative"],
        label="Macro Model Strategy",
        color="#34d399",
        linewidth=2,
    )
    plt.plot(
        test_df.index,
        test_df["benchmark_cumulative"],
        label="Buy & Hold S&P 500",
        color="#9ca3af",
        alpha=0.7,
        linewidth=1.5,
    )

    plt.title("Out-of-Sample Performance (3 Years)", fontsize=16, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Growth (1.0 = Base)", fontsize=12)
    plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)
    plt.legend(fontsize=12, loc="upper left")

    # Fill the area where strategy is in Cash (0 return, flat lines)
    cash_days = test_df[test_df["predicted_signal"] == 0]
    plt.scatter(
        cash_days.index,
        cash_days["strategy_cumulative"],
        color="red",
        s=5,
        alpha=0.3,
        label="In Cash",
    )

    plt.tight_layout()
    plt.savefig(base_dir / "equity_curve.png", dpi=150)
    plt.close()

    # --- Plotting 2: Feature Importance ---
    print("Plotting Feature Importances...")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15]  # Top 15

    top_features = [feature_cols[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title("Top 15 Macro Feature Importances", fontsize=16, pad=15)
    plt.xlabel("Gini Importance", fontsize=12)
    plt.ylabel("Macro Feature", fontsize=12)
    plt.tight_layout()

    plt.savefig(base_dir / "feature_importance.png", dpi=150)
    plt.close()

    print(f"Visualizations saved to {base_dir}")


if __name__ == "__main__":
    visualize_model()
