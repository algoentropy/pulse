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
            close = (
                data[("Close", symbol)] if len(all_tickers) > 1 else data["Close"]
            )
            close = close.dropna()
            points = []
            for ts, val in close.items():
                v = float(val)
                if not math.isnan(v):
                    points.append({"time": ts.strftime("%Y-%m-%d"), "value": round(v, 6)})
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
        data = yf.download(all_tickers, period="5d", progress=False)

    result = {}
    for key, cat in CATEGORIES.items():
        entries = []
        for symbol, name in cat["tickers"]:
            try:
                close = (
                    data[("Close", symbol)] if len(all_tickers) > 1 else data["Close"]
                )
                close = close.dropna()
                if len(close) < 2:
                    entries.append(
                        TickerEntry(name=name, ticker=symbol, status="closed")
                    )
                    continue
                current = float(close.iloc[-1])
                previous = float(close.iloc[-2])
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
    empty = {"categories": {"vitals": "", "muscles": "", "scoreboard": "", "geopolitics": ""}, "overall": ""}
    try:
        pulse = fetch_pulse_data()
        lines: list[str] = []
        for key in ("vitals", "muscles", "scoreboard", "geopolitics"):
            cat = getattr(pulse, key)
            lines.append(f"\n## {cat.label} — {cat.subtitle}")
            for t in cat.tickers:
                if t.price is not None and t.change_pct is not None:
                    lines.append(f"  {t.name} ({t.ticker}): ${t.price:,.2f}  {t.change_pct:+.2f}%")
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


# Serve frontend static files in production
DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="spa")
