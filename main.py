from __future__ import annotations

import math
from pathlib import Path

import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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


def fetch_pulse_data() -> PulseResponse:
    all_tickers = []
    for cat in CATEGORIES.values():
        for symbol, _ in cat["tickers"]:
            all_tickers.append(symbol)

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


# Serve frontend static files in production
DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="spa")
