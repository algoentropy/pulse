from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path

import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

# --- Models ---


class TickerEntry(BaseModel):
    name: str
    ticker: str
    price: float | None = None
    change_pct: float | None = None
    status: str = "active"


class CategoryData(BaseModel):
    label: str
    subtitle: str
    tickers: list[TickerEntry]


class PulseResponse(BaseModel):
    vitals: CategoryData
    muscles: CategoryData
    scoreboard: CategoryData
    geopolitics: CategoryData


# --- Ticker config ---

CATEGORIES: dict[str, dict] = {
    "vitals": {
        "label": "The Vitals",
        "subtitle": "Fear & Cost of Money",
        "tickers": [
            ("^VIX", "VIX"),
            ("^TNX", "US 10-Year Yield"),
            ("DX-Y.NYB", "US Dollar Index"),
        ],
    },
    "muscles": {
        "label": "The Muscles",
        "subtitle": "Growth & Industry",
        "tickers": [
            ("HG=F", "Copper Futures"),
            ("CL=F", "WTI Crude Oil"),
            ("^GDAXI", "DAX"),
            ("INDA", "MSCI India ETF"),
        ],
    },
    "scoreboard": {
        "label": "The Scoreboard",
        "subtitle": "Wealth & Sentiment",
        "tickers": [
            ("^GSPC", "S&P 500"),
            ("^NDX", "Nasdaq 100"),
            ("^HSI", "Hang Seng"),
            ("GC=F", "Gold Futures"),
            ("BTC-USD", "Bitcoin"),
        ],
    },
    "geopolitics": {
        "label": "Geopolitics",
        "subtitle": "Risk & Chokepoints",
        "tickers": [
            ("CHF=X", "USD/CHF"),
            ("ITA", "Aerospace & Defense ETF"),
            ("BDRY", "Dry Bulk Shipping ETF"),
            ("CIBR", "Cybersecurity ETF"),
        ],
    },
}


_yf_lock = threading.Lock()


def _all_tickers() -> list[str]:
    tickers = []
    for cat in CATEGORIES.values():
        for symbol, _ in cat["tickers"]:
            tickers.append(symbol)
    return tickers


def fetch_history_data() -> dict[str, list[dict]]:
    all_tickers = _all_tickers()
    with _yf_lock:
        data = yf.download(all_tickers, period="3mo", progress=False)

    result: dict[str, list[dict]] = {}
    for symbol, _ in (t for cat in CATEGORIES.values() for t in cat["tickers"]):
        try:
            close = data[("Close", symbol)] if len(all_tickers) > 1 else data["Close"]
            close = close.dropna()
            points = []
            for ts, val in close.items():
                v = float(val)
                if not math.isnan(v):
                    points.append(
                        {"time": ts.strftime("%Y-%m-%d"), "value": round(v, 6)}
                    )
            result[symbol] = points
        except Exception:
            result[symbol] = []
    return result


def fetch_pulse_data() -> PulseResponse:
    all_tickers = []
    for cat in CATEGORIES.values():
        for symbol, _ in cat["tickers"]:
            all_tickers.append(symbol)

    with _yf_lock:
        data = yf.download(all_tickers, period="1mo", progress=False)

    result = {}
    for key, cat in CATEGORIES.items():
        entries = []
        for symbol, name in cat["tickers"]:
            try:
                close = (
                    data[("Close", symbol)] if len(all_tickers) > 1 else data["Close"]
                )
                close = close.dropna()
                if len(close) == 0:
                    entries.append(
                        TickerEntry(name=name, ticker=symbol, status="closed")
                    )
                    continue
                current = float(close.iloc[-1])
                previous = float(close.iloc[-2]) if len(close) >= 2 else current
                if previous == 0 or math.isnan(current) or math.isnan(previous):
                    entries.append(
                        TickerEntry(name=name, ticker=symbol, status="error")
                    )
                    continue
                change = ((current - previous) / previous) * 100
                entries.append(
                    TickerEntry(
                        name=name,
                        ticker=symbol,
                        price=round(current, 6),
                        change_pct=round(change, 4),
                    )
                )
            except Exception:
                entries.append(TickerEntry(name=name, ticker=symbol, status="error"))
        result[key] = CategoryData(
            label=cat["label"],
            subtitle=cat["subtitle"],
            tickers=entries,
        )

    return PulseResponse(**result)


def fetch_interpretation() -> dict:
    empty = {
        "categories": {
            "vitals": "",
            "muscles": "",
            "scoreboard": "",
            "geopolitics": "",
        },
        "overall": "",
    }
    try:
        pulse = fetch_pulse_data()
        lines: list[str] = []
        for key in ("vitals", "muscles", "scoreboard", "geopolitics"):
            cat = getattr(pulse, key)
            lines.append(f"\n## {cat.label} — {cat.subtitle}")
            for t in cat.tickers:
                if t.price is not None and t.change_pct is not None:
                    lines.append(
                        f"  {t.name} ({t.ticker}): ${t.price:,.2f}  {t.change_pct:+.2f}%"
                    )
                else:
                    lines.append(f"  {t.name} ({t.ticker}): {t.status}")

        snapshot = "\n".join(lines)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise macro-market analyst. Given a market snapshot, "
                        "return JSON with two keys:\n"
                        '  "categories": an object with keys "vitals", "muscles", "scoreboard", "geopolitics" '
                        "each containing a 1-2 sentence interpretation of that category's data.\n"
                        '  "overall": a 2-3 sentence overall market narrative.\n'
                        "Return ONLY valid JSON, no markdown fences."
                    ),
                },
                {"role": "user", "content": f"Current market snapshot:\n{snapshot}"},
            ],
            temperature=0.4,
        )
        text = resp.choices[0].message.content or ""
        return json.loads(text)
    except Exception:
        return empty


# --- App ---

app = FastAPI(title="Global Pulse")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/pulse")
def get_pulse():
    return fetch_pulse_data()


@app.get("/api/history")
def get_history():
    return fetch_history_data()


@app.get("/api/interpretation")
def get_interpretation():
    return fetch_interpretation()


@app.get("/api/features")
def get_features():
    try:
        import pandas as pd

        features_path = Path(__file__).parent / "backtest" / "macro_features.parquet"
        if not features_path.exists():
            return {"copper_gold": [], "vix_tnx": []}

        df = pd.read_parquet(features_path)
        # Take the last 252 trading days (~1 year)
        df = df.tail(252)

        # Prepare the lists according to lightweight-charts expected format { time: "YYYY-MM-DD", value: float }
        copper_gold = []
        vix_tnx = []

        for idx, row in df.iterrows():
            ts_str = idx.strftime("%Y-%m-%d")

            if pd.notna(row.get("macro_copper_gold_ratio")):
                copper_gold.append(
                    {
                        "time": ts_str,
                        "value": round(float(row["macro_copper_gold_ratio"]), 6),
                    }
                )

            if pd.notna(row.get("macro_vix_tnx_ratio")):
                vix_tnx.append(
                    {
                        "time": ts_str,
                        "value": round(float(row["macro_vix_tnx_ratio"]), 4),
                    }
                )

        return {"copper_gold": copper_gold, "vix_tnx": vix_tnx}
    except Exception as e:
        print(f"Error loading features: {e}")
        return {"copper_gold": [], "vix_tnx": []}


@app.get("/api/predict")
def get_prediction():
    try:
        import pandas as pd
        import joblib

        base_path = Path(__file__).parent / "backtest"
        features_path = base_path / "macro_features.parquet"
        model_path = base_path / "rf_model.pkl"

        if not features_path.exists() or not model_path.exists():
            return {"status": "error", "message": "Model or features not found."}

        df = pd.read_parquet(features_path)
        clf = joblib.load(model_path)

        # The last row represents the most recent close
        latest_row = df.iloc[[-1]]

        # Reconstruct the feature columns the model expects
        feature_cols = [
            col
            for col in df.columns
            if col not in ["target", "^GSPC_ret_5d_future", "^GSPC_ret_1d_future"]
            and "future" not in col.lower()
        ]

        X_latest = latest_row[feature_cols]

        pred = clf.predict(X_latest)[0]
        # pred_proba returns probability of classes [0, 1]
        probs = clf.predict_proba(X_latest)[0]
        prob_up = float(probs[1])

        return {
            "status": "success",
            "prediction": "up" if pred == 1 else "down",
            "probability": prob_up,
            "date": latest_row.index[-1].strftime("%Y-%m-%d"),
        }

    except Exception as e:
        print(f"Predict error: {e}")
        return {"status": "error", "message": str(e)}


class TrainResponse(BaseModel):
    status: str
    message: str
    metrics: dict | None = None


@app.post("/api/train", response_model=TrainResponse)
def train():
    try:
        from backtest.data_pipeline import build_features
        from backtest.model import train_model

        # Warning: This is a synchronous call that takes ~15-30s
        # In a real app we'd use celery/background tasks.
        # For this dashboard it's acceptable.
        print("Starting data pipeline...")
        build_features()
        print("Data pipeline finished. Starting model training...")
        metrics = train_model()

        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": metrics,
        }
    except Exception as e:
        print(f"Train error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/backtest")
def get_backtest():
    try:
        import pandas as pd
        import joblib
        import numpy as np

        base_path = Path(__file__).parent / "backtest"
        features_path = base_path / "macro_features.parquet"
        model_path = base_path / "rf_model.pkl"

        if not features_path.exists() or not model_path.exists():
            return {"strategy": [], "benchmark": []}

        df = pd.read_parquet(features_path)
        clf = joblib.load(model_path)

        # 1. Recreate target and features as in visualize_model.py
        TARGET_TICKER = "^GSPC"
        FORWARD_DAYS = 5

        future_return_col = f"{TARGET_TICKER}_ret_{FORWARD_DAYS}d_future"
        df[future_return_col] = df[f"{TARGET_TICKER}_ret_{FORWARD_DAYS}d"].shift(
            -FORWARD_DAYS
        )
        df = df.dropna(subset=[future_return_col])

        # Exclude non-features
        exclude_cols = [future_return_col, "target"]
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and "future" not in col.lower()
        ]

        X = df[feature_cols]

        # 2. Split exactly precisely at 80% to match training
        split_idx = int(len(df) * 0.8)
        X_test = X.iloc[split_idx:]

        # Create 1-day future return for accurate compounding
        daily_future_return_col = f"{TARGET_TICKER}_ret_1d_future"
        df[daily_future_return_col] = df[f"{TARGET_TICKER}_ret_1d"].shift(-1)

        test_df = df.iloc[split_idx:].copy()

        # 3. Generate Predictions on the Test Set
        test_df["predicted_signal"] = clf.predict(X_test)

        # Simulate Strategy vs Benchmark
        test_df["strategy_daily_return"] = test_df.apply(
            lambda row: row[daily_future_return_col]
            if row["predicted_signal"] == 1
            else 0.0,
            axis=1,
        )
        test_df["benchmark_daily_return"] = test_df[daily_future_return_col]

        test_df["strategy_cumulative"] = (
            1 + test_df["strategy_daily_return"]
        ).cumprod()
        test_df["benchmark_cumulative"] = (
            1 + test_df["benchmark_daily_return"]
        ).cumprod()

        test_df = test_df.dropna(subset=["strategy_cumulative", "benchmark_cumulative"])

        # 4. Format for lightweight charts
        strategy_points = []
        benchmark_points = []

        for idx, row in test_df.iterrows():
            ts_str = idx.strftime("%Y-%m-%d")
            strategy_points.append(
                {"time": ts_str, "value": round(float(row["strategy_cumulative"]), 4)}
            )
            benchmark_points.append(
                {"time": ts_str, "value": round(float(row["benchmark_cumulative"]), 4)}
            )

        return {"strategy": strategy_points, "benchmark": benchmark_points}
    except Exception as e:
        print(f"Backtest API error: {e}")
        return {"strategy": [], "benchmark": []}


# Serve frontend static files in production
DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="spa")
