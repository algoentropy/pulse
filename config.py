from typing import TypedDict


class CategoryDict(TypedDict):
    label: str
    subtitle: str
    tickers: list[tuple[str, str, str]]


# --- Ticker config ---

CATEGORIES: dict[str, CategoryDict] = {
    "vitals": {
        "label": "The Vitals",
        "subtitle": "Fear & Cost of Money",
        "tickers": [
            (
                "^VIX",
                "VIX",
                "The 'fear index'. Measures expected stock market volatility. High values indicate panic or uncertainty.",
            ),
            (
                "^TNX",
                "US 10-Year Yield",
                "The baseline 'cost of money'. When this rises, borrowing gets more expensive, often hurting growth and tech stocks.",
            ),
            (
                "DX-Y.NYB",
                "US Dollar Index",
                "Measures the US Dollar against a basket of foreign currencies. A strong dollar can pressure multinational earnings and commodity prices.",
            ),
        ],
    },
    "muscles": {
        "label": "The Muscles",
        "subtitle": "Growth & Industry",
        "tickers": [
            (
                "HG=F",
                "Copper Futures",
                "Known as 'Dr. Copper' due to its widespread use in construction and manufacturing. Rising copper suggests a healthy, expanding global economy.",
            ),
            (
                "CL=F",
                "WTI Crude Oil",
                "The lifeblood of the industrial economy. Spikes can cause inflation, while crashes indicate a slowdown in demand.",
            ),
            (
                "^GDAXI",
                "DAX",
                "The primary stock index for Germany, serving as a bellwether for the industrial health of the entire European Union.",
            ),
            (
                "INDA",
                "MSCI India ETF",
                "Tracks large and mid-cap Indian equities. Used as a proxy for emerging market growth and rising middle-class consumption.",
            ),
        ],
    },
    "scoreboard": {
        "label": "The Scoreboard",
        "subtitle": "Wealth & Sentiment",
        "tickers": [
            (
                "^GSPC",
                "S&P 500",
                "The primary benchmark for US stock market performance and overall investor wealth.",
            ),
            (
                "^NDX",
                "Nasdaq 100",
                "A tech-heavy stock index. Highly sensitive to interest rates and a proxy for innovation and risk appetite.",
            ),
            (
                "^HSI",
                "Hang Seng",
                "The main stock market index in Hong Kong. Serves as a primary indicator of China's economic health and Asian market sentiment.",
            ),
            (
                "GC=F",
                "Gold Futures",
                "The classic 'safe haven' asset. Investors flock to gold during inflation fears, geopolitical crises, or when they lose faith in fiat currency.",
            ),
            (
                "BTC-USD",
                "Bitcoin",
                "A highly volatile digital asset. Often acts as a gauge for extreme risk appetite and global liquidity.",
            ),
        ],
    },
    "geopolitics": {
        "label": "Geopolitics",
        "subtitle": "Risk & Chokepoints",
        "tickers": [
            (
                "CHF=X",
                "USD/CHF",
                "The Swiss Franc is a traditional safe-haven currency. When global geopolitical risks rise, investors often buy Francs to protect capital.",
            ),
            (
                "ITA",
                "Aerospace & Defense ETF",
                "Tracks companies that manufacture military and defense equipment. Usually rallies during global conflicts or rising defense budgets.",
            ),
            (
                "BDRY",
                "Dry Bulk Shipping ETF",
                "Tracks the cost of moving raw materials like coal and iron ore across the ocean. Sensitive to supply chain bottlenecks and global trade volume.",
            ),
            (
                "CIBR",
                "Cybersecurity ETF",
                "Tracks companies providing cybersecurity hardware and software. Highly correlated with the frequency of digital warfare and corporate hacks.",
            ),
        ],
    },
}
