from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

from config import CATEGORIES

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from backtest.data_pipeline import build_features
from backtest.model import train_model

# --- Models ---


class TickerEntry(BaseModel):
    name: str
    ticker: str
    description: str | None = None
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


_yf_lock = threading.Lock()


def _all_tickers() -> list[str]:
    tickers = []
    for cat in CATEGORIES.values():
        for symbol, _, _ in cat["tickers"]:
            tickers.append(symbol)
    return tickers


def fetch_history_data(force_refresh: bool = False) -> dict[str, list[dict]]:
    cache_file = Path(__file__).parent / "pulse_cache" / "history.json"
    cache_ttl_seconds = 30 * 60  # 30 mins

    if not force_refresh and cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < cache_ttl_seconds:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

    all_tickers = _all_tickers()
    with _yf_lock:
        data = yf.download(all_tickers, period="3mo", progress=False, threads=False)
        if data is None:
            return {}
        data = pd.DataFrame(data)  # type: ignore

    result: dict[str, list[dict]] = {}
    for symbol, _, _ in (
        t for cat in CATEGORIES.values() if cat is not None for t in cat["tickers"]
    ):
        try:
            close = data[("Close", symbol)] if len(all_tickers) > 1 else data["Close"]
            close = close.dropna()
            points = []
            for ts, val in close.items():
                v = float(val)  # type: ignore
                if not math.isnan(v):
                    points.append(
                        {
                            "time": pd.Timestamp(str(ts)).strftime("%Y-%m-%d"),
                            "value": round(v, 6),
                        }  # type: ignore
                    )
            result[symbol] = points
        except Exception:
            result[symbol] = []

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def fetch_pulse_data(force_refresh: bool = False) -> PulseResponse:
    cache_file = Path(__file__).parent / "pulse_cache" / "pulse.json"
    cache_ttl_seconds = 30 * 60  # 30 mins

    if not force_refresh and cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < cache_ttl_seconds:
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    return PulseResponse(**cached_data)
            except Exception:
                pass

    all_tickers = []
    for cat in CATEGORIES.values():
        for symbol, _, _ in cat["tickers"]:
            all_tickers.append(symbol)

    with _yf_lock:
        data = yf.download(all_tickers, period="1mo", progress=False, threads=False)
        if data is None:
            raise ValueError("Failed to download data")
        data = pd.DataFrame(data)  # type: ignore

    result = {}
    for key, cat in CATEGORIES.items():
        entries = []
        if cat is None:
            continue
        for symbol, name, desc in cat["tickers"]:
            try:
                close = (
                    data[("Close", symbol)] if len(all_tickers) > 1 else data["Close"]
                )
                close = close.dropna()
                if len(close) == 0:
                    entries.append(
                        TickerEntry(
                            name=name, ticker=symbol, description=desc, status="closed"
                        )
                    )
                    continue
                current = float(close.iloc[-1])
                previous = float(close.iloc[-2]) if len(close) >= 2 else current
                if previous == 0 or math.isnan(current) or math.isnan(previous):
                    entries.append(
                        TickerEntry(
                            name=name, ticker=symbol, description=desc, status="error"
                        )
                    )
                    continue
                change = ((current - previous) / previous) * 100
                entries.append(
                    TickerEntry(
                        name=name,
                        ticker=symbol,
                        description=desc,
                        price=round(current, 6),
                        change_pct=round(change, 4),
                    )
                )
            except Exception:
                entries.append(
                    TickerEntry(
                        name=name, ticker=symbol, description=desc, status="error"
                    )
                )
        result[key] = CategoryData(
            label=cat["label"],
            subtitle=cat["subtitle"],
            tickers=entries,
        )

    response = PulseResponse(**result)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(response.model_dump(), f)

    return response


def fetch_interpretation(force_refresh: bool = False, mode: str = "executive") -> dict:
    empty = {
        "categories": {
            "vitals": "",
            "muscles": "",
            "scoreboard": "",
            "geopolitics": "",
        },
        "overall": "",
    }

    cache_file = Path(__file__).parent / "pulse_cache" / f"interpretation_{mode}.json"
    cache_ttl_seconds = 4 * 3600  # 4 hours

    try:
        if not force_refresh and cache_file.exists():
            mtime = cache_file.stat().st_mtime
            if time.time() - mtime < cache_ttl_seconds:
                with open(cache_file, "r") as f:
                    return json.load(f)

        pulse = fetch_pulse_data(force_refresh=force_refresh)
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

        executive_prompt = (
            "You are a concise macro-market analyst. Given a market snapshot, "
            "return JSON with two keys:\n"
            '  "categories": an object with keys "vitals", "muscles", "scoreboard", "geopolitics" '
            "each containing a 1-2 sentence interpretation of that category's data.\n"
            '  "overall": a 2-3 sentence overall market narrative.\n'
            "Return ONLY valid JSON, no markdown fences."
        )

        beginner_prompt = (
            "You are an expert macro-economics tutor explaining the markets to a layman beginner. "
            "Given a market snapshot, return JSON with two keys:\n"
            '  "categories": an object with keys "vitals", "muscles", "scoreboard", "geopolitics" '
            "each containing a 2-3 sentence highly educational explanation of *why* the assets in that category are moving and *how* they influence the broader economy. Avoid purely stating prices, explain the mechanics.\n"
            '  "overall": a 3-sentence plain-English summary of what the economy is currently doing, acting almost like a tutorial.\n'
            "Return ONLY valid JSON, no markdown fences."
        )

        sys_prompt = beginner_prompt if mode == "beginner" else executive_prompt

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {"role": "user", "content": f"Current market snapshot:\n{snapshot}"},
            ],
            temperature=0.4,
        )
        text = resp.choices[0].message.content or ""
        result = json.loads(text)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result
    except Exception as e:
        print(f"Interpretation API error: {e}")
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
def get_pulse(refresh: bool = False):
    return fetch_pulse_data(force_refresh=refresh)


@app.get("/api/history")
def get_history(refresh: bool = False):
    return fetch_history_data(force_refresh=refresh)


@app.get("/api/interpretation")
def get_interpretation(refresh: bool = False, mode: str = "executive"):
    return fetch_interpretation(force_refresh=refresh, mode=mode)


@app.get("/api/features")
def get_features():
    try:
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
            ts_str = pd.Timestamp(idx).strftime("%Y-%m-%d")  # type: ignore

            if pd.notna(row.get("macro_copper_gold_ratio")) is True:
                copper_gold.append(
                    {
                        "time": ts_str,
                        "value": round(float(row["macro_copper_gold_ratio"]), 6),  # type: ignore
                    }
                )

            if pd.notna(row.get("macro_vix_tnx_ratio")) is True:  # type: ignore
                vix_tnx.append(
                    {
                        "time": ts_str,
                        "value": round(float(row["macro_vix_tnx_ratio"]), 4),  # type: ignore
                    }
                )

        return {"copper_gold": copper_gold, "vix_tnx": vix_tnx}
    except Exception as e:
        print(f"Error loading features: {e}")
        return {"copper_gold": [], "vix_tnx": []}


@app.get("/api/predict")
def get_prediction(date: str | None = None):
    try:
        base_path = Path(__file__).parent / "backtest"
        features_path = base_path / "macro_features.parquet"
        model_path = base_path / "rf_model.pkl"

        if not features_path.exists() or not model_path.exists():
            return {"status": "error", "message": "Model or features not found."}

        df = pd.read_parquet(features_path)
        clf = joblib.load(model_path)

        if date:
            target_date = pd.to_datetime(date)
            # Find the closest matching date that is <= target_date
            # Assuming df index is already sorted datetime
            df = df.loc[:target_date]  # type: ignore
            if df.empty:
                return {
                    "status": "error",
                    "message": "No data available before this date.",
                }

        # The last row represents the most recent close (or the requested date)
        latest_row = df.iloc[[-1]]

        # Reconstruct the feature columns the model expects
        feature_cols = [
            col
            for col in df.columns
            if col not in ["target", "^GSPC_ret_5d_future", "^GSPC_ret_1d_future"]
            and "future" not in col.lower()
        ]

        X_latest = latest_row[feature_cols]

        pred = clf.predict(X_latest)[0]  # type: ignore
        # pred_proba returns probability of classes [0, 1]
        probs = clf.predict_proba(X_latest)[0]  # type: ignore
        prob_up = float(probs[1])  # type: ignore

        importances = clf.feature_importances_
        feature_names = X_latest.columns
        feat_imp = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )[:20]

        top_features = []
        for f, i in feat_imp:
            # Calculate 1-year historical min/max
            recent_history = df[str(f)].tail(252).dropna()
            if recent_history.empty:
                f_min, f_max = -1.0, 1.0
            else:
                f_min = float(recent_history.min())  # type: ignore
                f_max = float(recent_history.max())  # type: ignore

            # Expand bounds substantially (1000%) to allow simulating absolute Black Swan events
            f_range = f_max - f_min if f_max != f_min else 0.1
            f_min -= f_range * 10.0
            f_max += f_range * 10.0

            current_val = float(X_latest[str(f)].iloc[0])  # type: ignore
            top_features.append(
                {
                    "feature": str(f),
                    "importance": float(i),
                    "current_value": current_val,
                    "min": f_min,
                    "max": f_max,
                }
            )

        return {
            "status": "success",
            "prediction": "up" if int(pred) == 1 else "down",  # type: ignore
            "probability": prob_up,
            "date": pd.Timestamp(latest_row.index[-1]).strftime("%Y-%m-%d"),  # type: ignore
            "top_features": top_features,
        }

    except Exception as e:
        print(f"Predict error: {e}")
        return {"status": "error", "message": str(e)}


class SimulateRequest(BaseModel):
    overrides: dict[str, float]
    date: str | None = None


@app.post("/api/simulate")
def check_simulation(req: SimulateRequest):
    try:
        base_path = Path(__file__).parent / "backtest"
        features_path = base_path / "macro_features.parquet"
        model_path = base_path / "rf_model.pkl"

        if not features_path.exists() or not model_path.exists():
            return {"status": "error", "message": "Model or features not found."}

        df = pd.read_parquet(features_path)
        clf = joblib.load(model_path)

        if req.date:
            target_date = pd.to_datetime(req.date)
            df = df.loc[:target_date]  # type: ignore
            if df.empty:
                return {
                    "status": "error",
                    "message": "No data available before this date.",
                }

        latest_row = df.iloc[[-1]].copy()

        feature_cols = [
            col
            for col in df.columns
            if col not in ["target", "^GSPC_ret_5d_future", "^GSPC_ret_1d_future"]
            and "future" not in col.lower()
        ]

        X_latest = latest_row[feature_cols]

        # Apply overrides
        for feature, value in req.overrides.items():
            if feature in X_latest.columns:
                X_latest.loc[:, feature] = float(value)  # type: ignore

        pred = clf.predict(X_latest)[0]  # type: ignore
        probs = clf.predict_proba(X_latest)[0]  # type: ignore
        prob_up = float(probs[1])  # type: ignore

        return {
            "status": "success",
            "prediction": "up" if int(pred) == 1 else "down",
            "probability": prob_up,
        }

    except Exception as e:
        print(f"Simulate error: {e}")
        return {"status": "error", "message": str(e)}


class TrainResponse(BaseModel):
    status: str
    message: str
    metrics: dict | None = None


@app.post("/api/train", response_model=TrainResponse)
def train():
    try:
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
            ts_str = pd.Timestamp(idx).strftime("%Y-%m-%d")  # type: ignore
            strategy_points.append(
                {"time": ts_str, "value": round(float(row["strategy_cumulative"]), 4)}  # type: ignore
            )
            benchmark_points.append(
                {"time": ts_str, "value": round(float(row["benchmark_cumulative"]), 4)}  # type: ignore
            )

        return {"strategy": strategy_points, "benchmark": benchmark_points}
    except Exception as e:
        print(f"Backtest API error: {e}")
        return {"strategy": [], "benchmark": []}


# Serve frontend static files in production
DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="spa")
