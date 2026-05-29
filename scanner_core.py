from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from datetime import datetime, timezone
from datetime import datetime, timezone

DEFAULT_TICKERS = [
    "MSFT", "AAPL", "GOOG", "AMZN", "META", "NVDA", "AVGO", "AMD", "INTC",
    "TSM", "MU", "SMCI", "ARM", "QCOM", "SNPS", "ANET", "DELL",
    "CRWD", "PANW", "FTNT", "DDOG", "ZS", "NET", "MDB", "CRM", "NOW", "ORCL",
    "PLTR", "SNOW", "EPAM",
    "VRT", "ETN", "VST", "CEG", "GEV", "NRG", "IREN", "BE", "CWEN", "NNE", "OKLO",
    "KTOS", "LMT", "RTX", "LHX", "NOC", "AVAV", "BA", "HON", "RKLB", "ACHR",
    "RDW", "XAR", "ESLT",
    "MSTR", "IBIT", "CLSK", "MARA", "HUT", "HIVE", "CAN", "BMNR", "BTCS",
    "JPM", "BAC", "GS", "MA", "V", "SOFI", "IBKR", "KRE",
    "OXY", "CVX", "XOM", "SLB", "BKR", "SU", "BNO", "MP", "GLD", "SLV",
    "LLY", "UNH", "MRNA", "EXEL", "TMDX", "TEM", "ISRG", "SYK", "GRAL", "NTSK",
    "QUBT", "RGTI", "QBTS", "IONQ", "NBIS", "CRWV", "ODD", "FIG", "SEDG",
    "BABA", "ZIM",
    "SPY", "QQQ", "QQQM", "RSP", "SOXX", "SMH", "XLK", "IGV", "CIBR", "HACK",
    "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLB", "XLU", "XLP", "XLRE",
    "ITB", "REMX", "NLR", "QTUM", "TAN", "MAGS",
]

CORE_MARKET = [
    "^VIX", "BTC-USD", "ETH-USD", "GC=F", "SI=F",
    "RSP", "SPY", "^GSPC", "QQQM", "^NDX", "^DJI", "IWM"
]

MACRO = [
    "TIP",
    "DX-Y.NYB",
    "LQD",
    "HYG",

    # US Treasury yields
    "^IRX",   # 13-week / short-term proxy
    "^FVX",   # 5Y yield
    "^TNX",   # 10Y yield
    "^TYX",   # 30Y yield
]

SECTORS = [
    "XAR", "MAGS", "SOXX", "SMH", "CIBR", "HACK", "TAN",
    "XLF", "XLE", "IGV", "XLK", "XLV", "QTUM", "XLU",
    "XLP", "XLRE", "XLI", "XLY", "XLC", "XLB", "ITB",
    "REMX", "NLR"
]


@dataclass
class Setup:
    ticker: str
    setup: str
    score: int
    price: float
    ma50: Optional[float] = None
    ma150: Optional[float] = None
    rsi: Optional[float] = None
    breakout_level: Optional[float] = None
    distance_to_breakout_pct: Optional[float] = None
    rs_20d_vs_qqq: Optional[float] = None
    notes: str = ""


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_daily(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=12,
        )
        if df is None or df.empty or len(df) < 170:
            return None
        return _flatten(df).dropna()
    except Exception:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["MA50"] = close.rolling(50).mean()
    out["MA150"] = close.rolling(150).mean()
    out["RSI"] = RSIIndicator(close=close, window=14).rsi()
    out["HIGH_20_PREV"] = high.shift(1).rolling(20).max()

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["ATR14"] = tr.rolling(14).mean()
    return out


def qqq_return_20d() -> Optional[float]:
    qqq = download_daily("QQQ", period="3mo")
    if qqq is None or len(qqq) < 22:
        return None
    close = qqq["Close"]
    return float((close.iloc[-1] / close.iloc[-21] - 1) * 100)


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info or {}

        revenue_growth = info.get("revenueGrowth")
        forward_pe = info.get("forwardPE")
        peg_like = None

        if revenue_growth and revenue_growth > 0 and forward_pe:
            peg_like = forward_pe / (revenue_growth * 100)

        flags = []

        if revenue_growth is not None and revenue_growth >= 0.20:
            flags.append("High revenue growth")

        if info.get("grossMargins") is not None and info.get("grossMargins") >= 0.50:
            flags.append("High gross margin")

        if info.get("operatingMargins") is not None and info.get("operatingMargins") >= 0.20:
            flags.append("Strong operating margin")

        if info.get("freeCashflow") is not None and info.get("freeCashflow") > 0:
            flags.append("Positive free cash flow")

        if peg_like is not None and peg_like <= 1.5:
            flags.append("Reasonable growth valuation")

        if info.get("shortPercentOfFloat") is not None and info.get("shortPercentOfFloat") >= 0.10:
            flags.append("High short interest")

        return {
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": forward_pe,
            "peg_like": round(peg_like, 2) if peg_like is not None else None,
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "revenue_growth_pct": round(revenue_growth * 100, 2) if revenue_growth is not None else None,
            "gross_margins_pct": round(info.get("grossMargins") * 100, 2) if info.get("grossMargins") is not None else None,
            "operating_margins_pct": round(info.get("operatingMargins") * 100, 2) if info.get("operatingMargins") is not None else None,
            "profit_margins_pct": round(info.get("profitMargins") * 100, 2) if info.get("profitMargins") is not None else None,
            "free_cashflow": info.get("freeCashflow"),
            "analyst_target": info.get("targetMeanPrice"),
            "short_float_pct": round(info.get("shortPercentOfFloat") * 100, 2) if info.get("shortPercentOfFloat") is not None else None,
            "fundamental_flags": flags,
        }
    except Exception as e:
        return {"fundamental_error": str(e)}
        
def get_latest_news(ticker: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch latest ticker-related news from yfinance.
    """
    ticker = ticker.upper().strip()

    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []

        results: List[Dict[str, Any]] = []

        for item in raw_news[:limit]:
            published_raw = item.get("providerPublishTime")
            published_utc = None

            if published_raw:
                try:
                    published_utc = datetime.fromtimestamp(
                        published_raw,
                        tz=timezone.utc
                    ).isoformat()
                except Exception:
                    published_utc = str(published_raw)

            title = item.get("title")
            publisher = item.get("publisher")
            link = item.get("link")
            item_type = item.get("type")

            title_l = (title or "").lower()

            importance = "LOW"
            impact = "UNKNOWN"
            reason = []

            high_keywords = [
                "earnings", "guidance", "forecast", "outlook",
                "downgrade", "upgrade", "price target",
                "sec", "lawsuit", "probe", "investigation",
                "antitrust", "merger", "acquisition",
                "contract", "deal", "partnership",
                "beats", "misses"
            ]

            medium_keywords = [
                "analyst", "shares", "stock", "ai",
                "data center", "chip", "cloud",
                "bitcoin", "crypto", "oil", "fed",
                "rates", "tariff", "iran"
            ]

            negative_keywords = [
                "downgrade", "misses", "lawsuit", "probe",
                "investigation", "antitrust", "cuts",
                "falls", "drops", "warning"
            ]

            positive_keywords = [
                "upgrade", "beats", "raises", "wins",
                "contract", "partnership", "surges",
                "rises", "record"
            ]

            if any(k in title_l for k in high_keywords):
                importance = "HIGH"
                reason.append("high-impact keyword")
            elif any(k in title_l for k in medium_keywords):
                importance = "MEDIUM"
                reason.append("market-relevant keyword")

            if any(k in title_l for k in negative_keywords):
                impact = "NEGATIVE"
            elif any(k in title_l for k in positive_keywords):
                impact = "POSITIVE"

            results.append({
                "title": title,
                "publisher": publisher,
                "published_utc": published_utc,
                "type": item_type,
                "link": link,
                "importance": importance,
                "estimated_impact": impact,
                "reason": reason,
            })

        return results

    except Exception as e:
        return [{
            "news_error": str(e)
        }]

def _technical_analysis(ticker: str, qqq_20d: Optional[float] = None) -> Optional[Dict[str, Any]]:
    ticker = ticker.upper().strip()

    df = download_daily(ticker)
    if df is None:
        return None

    df = add_indicators(df)
    latest = df.iloc[-1]

    price = float(latest["Close"])
    ma50 = float(latest["MA50"])
    ma150 = float(latest["MA150"])
    rsi = float(latest["RSI"])
    breakout = float(latest["HIGH_20_PREV"])

    if pd.isna(ma50) or pd.isna(ma150) or pd.isna(rsi) or pd.isna(breakout):
        return None

    close = df["Close"]
    stock_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 22 else None
    rs_diff = round(stock_20d - qqq_20d, 2) if stock_20d is not None and qqq_20d is not None else None

    distance_to_breakout = ((breakout / price) - 1) * 100
    distance_from_ma50 = ((price / ma50) - 1) * 100
    distance_from_ma150 = ((price / ma150) - 1) * 100

    setups: List[Setup] = []

    if price > ma150 and 45 <= rsi <= 68 and 0 < distance_to_breakout <= 5 and distance_from_ma50 <= 8:
        score = 5
        if rs_diff is not None and rs_diff > 0:
            score += 2
        if price > ma50:
            score += 1

        setups.append(
            Setup(
                ticker=ticker,
                setup="NEAR_BREAKOUT",
                score=score,
                price=round(price, 2),
                ma50=round(ma50, 2),
                ma150=round(ma150, 2),
                rsi=round(rsi, 1),
                breakout_level=round(breakout, 2),
                distance_to_breakout_pct=round(distance_to_breakout, 2),
                rs_20d_vs_qqq=rs_diff,
                notes="Near 20-day breakout; verify volume and structure on TradingView.",
            )
        )

    recent_high = float(df["High"].tail(20).max())
    drawdown = ((price / recent_high) - 1) * 100

    if price > ma150 and -8 <= drawdown <= -2 and -2 <= distance_from_ma50 <= 3 and 40 <= rsi <= 60:
        score = 5
        if rs_diff is not None and rs_diff > 0:
            score += 2

        setups.append(
            Setup(
                ticker=ticker,
                setup="PULLBACK_TO_MA50",
                score=score,
                price=round(price, 2),
                ma50=round(ma50, 2),
                ma150=round(ma150, 2),
                rsi=round(rsi, 1),
                rs_20d_vs_qqq=rs_diff,
                notes=f"Pullback {drawdown:.1f}% from 20-day high; check support reaction.",
            )
        )

    atr_now = df["ATR14"].iloc[-1]
    atr_20ago = df["ATR14"].iloc[-20]

    lows = df["Low"].tail(20)
    higher_lows = lows.iloc[10:].min() > lows.iloc[:10].min()

    atr_compression = (
        not pd.isna(atr_now)
        and not pd.isna(atr_20ago)
        and atr_now < atr_20ago * 0.85
    )

    near_breakout = 0 < distance_to_breakout <= 5
    volume_accel = (
        df["Volume"].rolling(20).mean().iloc[-1]
        > df["Volume"].rolling(50).mean().iloc[-1] * 1.05
    )

    coiled_score = 0
    if atr_compression:
        coiled_score += 2
    if higher_lows:
        coiled_score += 2
    if near_breakout:
        coiled_score += 2
    if volume_accel:
        coiled_score += 1
    if price > ma150:
        coiled_score += 1

    if coiled_score >= 5 and distance_from_ma50 < 10:
        setups.append(
            Setup(
                ticker=ticker,
                setup="COILED",
                score=coiled_score,
                price=round(price, 2),
                ma50=round(ma50, 2),
                ma150=round(ma150, 2),
                rsi=round(rsi, 1),
                breakout_level=round(breakout, 2),
                distance_to_breakout_pct=round(distance_to_breakout, 2),
                rs_20d_vs_qqq=rs_diff,
                notes="Compression plus higher lows; watch for clean breakout candle.",
            )
        )

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "ma50": round(ma50, 2),
        "ma150": round(ma150, 2),
        "rsi": round(rsi, 1),
        "distance_from_ma50_pct": round(distance_from_ma50, 2),
        "distance_from_ma150_pct": round(distance_from_ma150, 2),
        "rs_20d_vs_qqq": rs_diff,
        "setups": [asdict(s) for s in sorted(setups, key=lambda s: s.score, reverse=True)],
    }


def analyze_ticker(ticker: str, qqq_20d: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Full single-ticker analysis:
    technicals + fundamentals + latest news.
    Use this only after scan_watchlist finds candidates, or when the user asks for a specific ticker.
    """
    ticker = ticker.upper().strip()

    technical = _technical_analysis(ticker, qqq_20d=qqq_20d)
    if technical is None:
        return None

    technical["fundamentals"] = get_fundamentals(ticker)
    technical["latest_news"] = get_latest_news(ticker)

    return technical


def scan_watchlist(tickers: Optional[List[str]] = None, max_results: int = 25) -> Dict[str, Any]:
    """
    Fast technical scan only.
    Does NOT call fundamentals, to avoid timeouts.
    """
    tickers = tickers or DEFAULT_TICKERS
    qqq_20d = qqq_return_20d()

    found: List[Dict[str, Any]] = []
    errors = 0

    for ticker in tickers:
        result = _technical_analysis(ticker.strip().upper(), qqq_20d=qqq_20d)

        if result is None:
            errors += 1
            continue

        for setup in result.get("setups", []):
            found.append(setup)

    found.sort(
        key=lambda x: (x.get("score", 0), x.get("rs_20d_vs_qqq") or -999),
        reverse=True,
    )

    return {
        "count": len(found[:max_results]),
        "qqq_20d_return": round(qqq_20d, 2) if qqq_20d is not None else None,
        "errors": errors,
        "results": found[:max_results],
        "note": "Technical scan only. Run analyze_ticker on selected candidates for fundamentals.",
    }


def market_map() -> Dict[str, Any]:
    groups = {
        "core_market": CORE_MARKET,
        "macro": MACRO,
        "sectors": SECTORS,
    }

    results: List[Dict[str, Any]] = []

    for group_name, tickers in groups.items():
        for ticker in tickers:
            data = download_daily(ticker)

            if data is None:
                results.append({"group": group_name, "ticker": ticker, "error": "No data"})
                continue

            data = add_indicators(data)
            latest = data.iloc[-1]

            price = float(latest["Close"])
            ma50 = float(latest["MA50"])
            ma150 = float(latest["MA150"])
            rsi = float(latest["RSI"])

            close = data["Close"]
            change_1d_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
            momentum_20d_pct = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)

            trend = "Bullish"
            if price < ma50:
                trend = "Weak"
            if price < ma150:
                trend = "Bearish"

            results.append(
                {
                    "group": group_name,
                    "ticker": ticker,
                    "price": round(price, 2),
                    "change_1d_pct": round(change_1d_pct, 2),
                    "momentum_20d_pct": round(momentum_20d_pct, 2),
                    "ma50": round(ma50, 2),
                    "ma150": round(ma150, 2),
                    "rsi": round(rsi, 1),
                    "trend": trend,
                    "above_ma50": price > ma50,
                    "above_ma150": price > ma150,
                }
            )

    bullish_count = sum(1 for r in results if r.get("trend") == "Bullish")
    weak_count = sum(1 for r in results if r.get("trend") == "Weak")
    bearish_count = sum(1 for r in results if r.get("trend") == "Bearish")

    return {
        "summary": {
            "total": len(results),
            "bullish": bullish_count,
            "weak": weak_count,
            "bearish": bearish_count,
        },
        "results": results,
    }
