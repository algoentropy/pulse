import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

TARGET_TICKER = "^GSPC"  # S&P 500
FORWARD_DAYS = 5  # Predict 1 week into the future


def train_model():
    print("Loading features from macro_features.parquet...")
    features_path = Path(__file__).parent / "macro_features.parquet"
    if not features_path.exists():
        print("Error: macro_features.parquet not found. Run data_pipeline.py first.")
        return

    df = pd.read_parquet(features_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Defining target: {FORWARD_DAYS}-day return of {TARGET_TICKER}")

    # Target: To predict 5 days ahead, we shift the 5-day return backwards by 5 days
    future_return_col = f"{TARGET_TICKER}_ret_{FORWARD_DAYS}d_future"

    # If today is Monday, we want to predict the return from Monday to next Monday.
    # The return_5d column currently at "next Monday" is (Price_NextMon - Price_Mon) / Price_Mon.
    # By shifting it backwards, today's row will contain that value.
    df[future_return_col] = df[f"{TARGET_TICKER}_ret_{FORWARD_DAYS}d"].shift(
        -FORWARD_DAYS
    )

    # We must cleanly drop rows where we don't have future data (the very last 5 days of the dataset)
    df = df.dropna(subset=[future_return_col])

    # Binary Classification Target: 1 if return > 0, else 0
    df["target"] = (df[future_return_col] > 0.0).astype(int)

    # Define our feature set
    # We explicitly EXCLUDE any forward-looking data or the raw closing prices themselves
    exclude_cols = [future_return_col, "target"]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Additional verification: Ensure no "future" strings slipped into feature names
    feature_cols = [col for col in feature_cols if "future" not in col.lower()]

    X = df[feature_cols]
    y = df["target"]

    print(f"Using {len(feature_cols)} features.")

    # Time-Series Split: 80% Train, 20% Test
    # STRICT RULE: NEVER randomly shuffle time-series data for quantitative modeling
    split_idx = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples (Out-of-Sample)")

    # Train Random Forest
    print("\nTraining Random Forest Classifier (this may take a moment)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Keep depth shallow to prevent extreme overfitting to noise
        min_samples_leaf=20,  # Require at least 20 days in a leaf
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0.0)  # type: ignore

    print("\n--- Out-of-Sample Evaluation ---")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%} (When we predict UP, how often are we right?)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n--- Top 10 Macro Features ---")
    top_features = []
    for i in range(10):
        idx = sorted_idx[i]
        top_features.append(
            {"feature": feature_cols[idx], "importance": float(importances[idx])}
        )
        print(f"{i + 1:2d}. {importances[idx]:.4f} - {feature_cols[idx]}")

    model_path = Path(__file__).parent / "rf_model.pkl"
    joblib.dump(clf, model_path)
    print(f"\nModel saved successfully to {model_path}")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "top_features": top_features,
    }


if __name__ == "__main__":
    train_model()
