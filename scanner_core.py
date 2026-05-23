from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

DEFAULT_TICKERS = [
    "INTC", "MSFT", "AAPL", "AMD", "TSLA", "GOOG", "NVDA", "AVGO", "META",
    "PLTR", "SNPS", "AMZN", "TSM", "MU", "SMCI", "ARM", "DDOG", "CRWD", "ZS",
    "PANW", "CRM", "NOW", "SNOW", "ORCL", "FTNT", "NET", "MDB", "QQQM", "SOXX",
    "SMH", "CIBR", "XLK", "XLF", "IGV", "JPM", "IBKR", "GS", "MA", "V", "SOFI",
    "TMDX", "TEM", "UNH", "LLY", "ISRG", "SYK", "QUBT", "RGTI", "QBTS", "IONQ",
    "ACHR", "RKLB", "RTX", "LHX", "NOC", "KTOS", "BA", "LMT", "AVAV", "HON",
    "QCOM", "NNE", "OKLO", "BKR", "SLB", "CAT", "OXY", "CVX", "XOM", "ANET",
    "IREN", "ETN", "NRG", "VST", "CEG", "GEV", "VRT", "MP", "MSTR", "CLSK",
    "IBIT", "GLD", "SLV", "BABA", "ZIM", "ODD", "CRWV"
]
MARKET_INDEXES = [
    "SPY",
    "QQQ",
    "IWM",
    "SOXX",
    "SMH",
    "IGV",
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLI",
    "XLY",
    "XLC",
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
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
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
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    return out


def qqq_return_20d() -> Optional[float]:
    qqq = download_daily("QQQ", period="3mo")
    if qqq is None or len(qqq) < 22:
        return None
    close = qqq["Close"]
    return float((close.iloc[-1] / close.iloc[-21] - 1) * 100)


def analyze_ticker(ticker: str, qqq_20d: Optional[float] = None) -> Optional[Dict[str, Any]]:
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

    # Near-breakout setup
    if price > ma150 and 45 <= rsi <= 68 and 0 < distance_to_breakout <= 5 and distance_from_ma50 <= 8:
        score = 5
        if rs_diff is not None and rs_diff > 0: score += 2
        if price > ma50: score += 1
        setups.append(Setup(ticker, "NEAR_BREAKOUT", score, round(price,2), round(ma50,2), round(ma150,2), round(rsi,1), round(breakout,2), round(distance_to_breakout,2), rs_diff, "Near 20-day breakout; verify volume/structure on TradingView."))

    # Pullback setup
    recent_high = float(df["High"].tail(20).max())
    drawdown = ((price / recent_high) - 1) * 100
    if price > ma150 and -8 <= drawdown <= -2 and -2 <= distance_from_ma50 <= 3 and 40 <= rsi <= 60:
        score = 5
        if rs_diff is not None and rs_diff > 0: score += 2
        setups.append(Setup(ticker, "PULLBACK_TO_MA50", score, round(price,2), round(ma50,2), round(ma150,2), round(rsi,1), None, None, rs_diff, f"Pullback {drawdown:.1f}% from 20d high; check support reaction."))

    # Coiled setup
    atr_now = df["ATR14"].iloc[-1]
    atr_20ago = df["ATR14"].iloc[-20]
    lows = df["Low"].tail(20)
    higher_lows = lows.iloc[10:].min() > lows.iloc[:10].min()
    atr_compression = bool(not pd.isna(atr_now) and not pd.isna(atr_20ago) and atr_now < atr_20ago * 0.85)
    near_breakout = 0 < distance_to_breakout <= 5
    volume_accel = df["Volume"].rolling(20).mean().iloc[-1] > df["Volume"].rolling(50).mean().iloc[-1] * 1.05
    coiled_score = sum([2 if atr_compression else 0, 2 if higher_lows else 0, 2 if near_breakout else 0, 1 if volume_accel else 0, 1 if price > ma150 else 0])
    if coiled_score >= 5 and distance_from_ma50 < 10:
        setups.append(Setup(ticker, "COILED", coiled_score, round(price,2), round(ma50,2), round(ma150,2), round(rsi,1), round(breakout,2), round(distance_to_breakout,2), rs_diff, "Compression + higher lows; watch for clean breakout candle."))

    if not setups:
        return {
            "ticker": ticker, "price": round(price,2), "ma50": round(ma50,2), "ma150": round(ma150,2),
            "rsi": round(rsi,1), "rs_20d_vs_qqq": rs_diff, "setups": []
        }
    return {"ticker": ticker, "setups": [asdict(s) for s in sorted(setups, key=lambda s: s.score, reverse=True)]}


def scan_watchlist(tickers: Optional[List[str]] = None, max_results: int = 20) -> Dict[str, Any]:
    tickers = tickers or DEFAULT_TICKERS
    qqq_20d = qqq_return_20d()
    found: List[Dict[str, Any]] = []
    errors = 0
    for t in tickers:
        result = analyze_ticker(t.strip().upper(), qqq_20d=qqq_20d)
        if result is None:
            errors += 1
            continue
        for setup in result.get("setups", []):
            found.append(setup)
    found.sort(key=lambda x: (x.get("score", 0), x.get("rs_20d_vs_qqq") or -999), reverse=True)
    return {"count": len(found[:max_results]), "qqq_20d_return": round(qqq_20d,2) if qqq_20d is not None else None, "errors": errors, "results": found[:max_results]}
