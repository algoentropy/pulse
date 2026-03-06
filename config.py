from typing import TypedDict


class CategoryDict(TypedDict):
    label: str
    subtitle: str
    tickers: list[tuple[str, str]]


# --- Ticker config ---

CATEGORIES: dict[str, CategoryDict] = {
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
