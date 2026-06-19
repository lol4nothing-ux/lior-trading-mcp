from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

LOOKBACK_SESSIONS = 20
PRE_CLOSE_WINDOW_MINUTES = 120
SHORT_WINDOW_MINUTES = 30
EXTENDED_INTRADAY_PCT = 7.0
EXTENDED_VWAP_DISTANCE_PCT = 5.0
MIN_AVG_DOLLAR_VOLUME_20D = 20_000_000
MIN_HISTORICAL_INTRADAY_SESSIONS = 10
US_EASTERN = ZoneInfo("America/New_York")

VOLUME_SIGNAL_NO_UNUSUAL = "NO_UNUSUAL_VOLUME"
VOLUME_SIGNAL_MILD = "MILD_INTEREST"
VOLUME_SIGNAL_STRONG = "STRONG_POSITIVE_INTEREST"
VOLUME_SIGNAL_ACCUMULATION = "POSSIBLE_ACCUMULATION"
VOLUME_SIGNAL_DISTRIBUTION = "DISTRIBUTION_RISK"
VOLUME_SIGNAL_CHASE = "CHASE_RISK"
VOLUME_SIGNAL_INSUFFICIENT = "INSUFFICIENT_DATA"

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


def download_intraday(ticker: str, period: str = "30d", interval: str = "5m") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=12,
        )
        if df is None or df.empty:
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
        
def _parse_news_datetime(value: Any) -> Optional[datetime]:
    """
    Supports:
    - Unix timestamp
    - ISO datetime strings
    - Yahoo/yfinance pubDate formats
    """
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
    except Exception:
        pass

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        try:
            # Handles 2026-06-02T14:30:00Z
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass

        try:
            # Fallback for some Yahoo-style date strings
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    return None


def _extract_news_url(item: Dict[str, Any], content: Dict[str, Any]) -> Optional[str]:
    """
    Handles old and new yfinance/Yahoo schemas.
    """
    direct = item.get("link") or item.get("url")
    if direct:
        return direct

    canonical = content.get("canonicalUrl")
    if isinstance(canonical, dict):
        if canonical.get("url"):
            return canonical.get("url")

    clickthrough = content.get("clickThroughUrl")
    if isinstance(clickthrough, dict):
        if clickthrough.get("url"):
            return clickthrough.get("url")

    return None


def _extract_news_provider(item: Dict[str, Any], content: Dict[str, Any]) -> Optional[str]:
    provider = item.get("publisher")

    if provider:
        return provider

    provider_obj = content.get("provider")
    if isinstance(provider_obj, dict):
        return (
            provider_obj.get("displayName")
            or provider_obj.get("name")
            or provider_obj.get("source")
        )

    return None


def get_latest_news(
    ticker: str,
    limit: int = 8,
    lookback_hours: int = 72
) -> List[Dict[str, Any]]:
    """
    Fetch recent ticker-related news from yfinance/Yahoo Finance.

    Strict rules:
    - Keeps only news with title + source + published time + url.
    - Keeps only news from the last lookback_hours.
    - Distinguishes provider failure from no fresh news.
    """
    ticker = ticker.upper().strip()
    now = datetime.now(timezone.utc)
    cutoff = now - pd.Timedelta(hours=lookback_hours)

    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []

        if not raw_news:
            return [{
                "news_status": "NO_NEWS_RETURNED_BY_PROVIDER",
                "ticker": ticker,
                "message": "Yahoo/yfinance returned no news items."
            }]

        results: List[Dict[str, Any]] = []

        for item in raw_news:
            if not isinstance(item, dict):
                continue

            content = item.get("content") or {}
            if not isinstance(content, dict):
                content = {}

            title = (
                item.get("title")
                or content.get("title")
                or content.get("headline")
            )

            publisher = _extract_news_provider(item, content)
            link = _extract_news_url(item, content)

            published_dt = (
                _parse_news_datetime(item.get("providerPublishTime"))
                or _parse_news_datetime(item.get("published_at"))
                or _parse_news_datetime(content.get("pubDate"))
                or _parse_news_datetime(content.get("displayTime"))
            )

            if not title or not publisher or not link or not published_dt:
                continue

            if published_dt < cutoff:
                continue

            title_l = title.lower()

            importance = "LOW"
            impact = "UNKNOWN"
            reason = []

            high_keywords = [
                "earnings", "guidance", "forecast", "outlook",
                "downgrade", "upgrade", "price target",
                "sec", "lawsuit", "probe", "investigation",
                "antitrust", "merger", "acquisition",
                "contract", "deal", "partnership",
                "beats", "misses", "revenue", "eps"
            ]

            medium_keywords = [
                "analyst", "shares", "stock", "ai",
                "data center", "chip", "cloud",
                "bitcoin", "crypto", "oil", "fed",
                "rates", "tariff", "iran", "export controls"
            ]

            negative_keywords = [
                "downgrade", "misses", "lawsuit", "probe",
                "investigation", "antitrust", "cuts",
                "falls", "drops", "warning", "slumps"
            ]

            positive_keywords = [
                "upgrade", "beats", "raises", "wins",
                "contract", "partnership", "surges",
                "rises", "record", "tops"
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
                "ticker": ticker,
                "title": title.strip(),
                "publisher": publisher,
                "published_utc": published_dt.isoformat(),
                "age_hours": round((now - published_dt).total_seconds() / 3600, 2),
                "link": link,
                "importance": importance,
                "estimated_impact": impact,
                "reason": reason,
            })

        results.sort(key=lambda x: x.get("published_utc", ""), reverse=True)

        if not results:
            return [{
                "news_status": "NO_FRESH_NEWS",
                "ticker": ticker,
                "lookback_hours": lookback_hours,
                "message": f"No valid fresh news found in the last {lookback_hours} hours."
            }]

        return results[:limit]

    except Exception as e:
        return [{
            "news_status": "NEWS_PROVIDER_FAILED",
            "ticker": ticker,
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


def analyze_volume(ticker: str) -> Dict[str, Any]:
    """
    Analyze unusual intraday volume and pre-close interest.

    If intraday OHLCV is unavailable or the provider fails, return a normal
    INSUFFICIENT_DATA object instead of breaking the MCP tool.
    """
    ticker = ticker.upper().strip()
    try:
        return _analyze_volume_interest(ticker)
    except Exception as exc:
        result = _empty_volume_interest()
        result["volume_explanation_he"] = "אין מספיק נתוני ווליום תוך-יומיים לחישוב אמין."
        result["volume_error"] = str(exc)
        return result


def _analyze_volume_interest(ticker: str) -> Dict[str, Any]:
    result = _empty_volume_interest()

    daily = download_daily(ticker, period="1y")
    avg_dollar_volume = _avg_daily_dollar_volume(daily)
    liquidity_ok = (
        avg_dollar_volume >= MIN_AVG_DOLLAR_VOLUME_20D
        if avg_dollar_volume is not None
        else None
    )
    result["avg_daily_dollar_volume_20d"] = _round(avg_dollar_volume)
    result["liquidity_ok"] = liquidity_ok

    intraday_raw = download_intraday(ticker, period="30d", interval="5m")
    intraday = _prepare_intraday(intraday_raw)
    if intraday.empty:
        result["volume_explanation_he"] = "אין נתוני ווליום תוך-יומיים זמינים."
        return result

    session_dates = sorted(intraday["session_date"].dropna().unique())
    if len(session_dates) < MIN_HISTORICAL_INTRADAY_SESSIONS + 1:
        result["volume_explanation_he"] = "אין מספיק היסטוריה תוך-יומית אמינה לחישוב RVOL לפי שעה."
        return result

    latest_date = session_dates[-1]
    historical_dates = session_dates[-(LOOKBACK_SESSIONS + 1):-1]
    today = intraday[intraday["session_date"] == latest_date].copy()
    historical = intraday[intraday["session_date"].isin(historical_dates)].copy()

    if today.empty or historical.empty:
        result["volume_explanation_he"] = "נתוני הווליום התוך-יומיים חסרים או קצרים מדי."
        return result

    latest_ts = today["timestamp_eastern"].max()
    latest_tod = latest_ts.time()
    short_start = (latest_ts - pd.Timedelta(minutes=SHORT_WINDOW_MINUTES)).time()
    long_start = (latest_ts - pd.Timedelta(minutes=PRE_CLOSE_WINDOW_MINUTES)).time()

    cumulative_today = _sum_until(today, latest_tod)
    cumulative_avg = _average_session_sum(historical, None, latest_tod)
    last_2h_today = _sum_window(today, long_start, latest_tod)
    last_2h_avg = _average_session_sum(historical, long_start, latest_tod)
    last_30m_today = _sum_window(today, short_start, latest_tod)
    last_30m_avg = _average_session_sum(historical, short_start, latest_tod)
    previous_90m_today = _sum_window(today, long_start, short_start)
    previous_90m_avg = _average_session_sum(historical, long_start, short_start)

    cumulative_rvol = _safe_ratio(cumulative_today, cumulative_avg)
    last_2h_ratio = _safe_ratio(last_2h_today, last_2h_avg)
    last_30m_ratio = _safe_ratio(last_30m_today, last_30m_avg)
    previous_90m_ratio = _safe_ratio(previous_90m_today, previous_90m_avg)
    acceleration = _safe_ratio(last_30m_ratio, previous_90m_ratio)

    price = _safe_float(today.iloc[-1].get("Close"))
    first_open = _safe_float(today.iloc[0].get("Open"))
    day_high = _safe_float(today["High"].max())
    day_low = _safe_float(today["Low"].min())
    vwap = _vwap(today)

    price_vs_vwap_pct = _pct_change(price, vwap)
    above_vwap = price > vwap if price is not None and vwap not in (None, 0) else None
    day_range_position = _day_range_position(price, day_low, day_high)
    intraday_change = _pct_change(price, first_open)
    extended = _extended_intraday(intraday_change, price_vs_vwap_pct)

    score = _score_volume_interest(
        cumulative_rvol=cumulative_rvol,
        last_2h_ratio=last_2h_ratio,
        last_30m_ratio=last_30m_ratio,
        acceleration=acceleration,
        above_vwap=above_vwap,
        day_range_position=day_range_position,
        extended=extended,
        liquidity_ok=liquidity_ok,
    )
    if liquidity_ok is False:
        score = min(score, 6.0)

    signal = _volume_signal(
        cumulative_rvol=cumulative_rvol,
        last_2h_ratio=last_2h_ratio,
        above_vwap=above_vwap,
        day_range_position=day_range_position,
        intraday_change=intraday_change,
        extended=extended,
    )

    result.update(
        {
            "volume_interest_score": round(score, 2),
            "volume_signal": signal,
            "volume_explanation_he": _volume_explanation_he(signal, liquidity_ok),
            "cumulative_rvol_by_time": _round(cumulative_rvol),
            "last_2h_volume_ratio": _round(last_2h_ratio),
            "last_30m_volume_ratio": _round(last_30m_ratio),
            "volume_acceleration": _round(acceleration),
            "above_vwap": above_vwap,
            "price_vs_vwap_pct": _round(price_vs_vwap_pct),
            "day_range_position_pct": _round(day_range_position),
            "intraday_change_pct": _round(intraday_change),
            "extended_intraday": extended,
            "avg_daily_dollar_volume_20d": _round(avg_dollar_volume),
            "liquidity_ok": liquidity_ok,
        }
    )
    return result


def _empty_volume_interest() -> Dict[str, Any]:
    return {
        "volume_interest_score": 0,
        "volume_signal": VOLUME_SIGNAL_INSUFFICIENT,
        "volume_explanation_he": "אין מספיק נתונים לחישוב ווליום חריג.",
        "cumulative_rvol_by_time": None,
        "last_2h_volume_ratio": None,
        "last_30m_volume_ratio": None,
        "volume_acceleration": None,
        "above_vwap": None,
        "price_vs_vwap_pct": None,
        "day_range_position_pct": None,
        "intraday_change_pct": None,
        "extended_intraday": None,
        "avg_daily_dollar_volume_20d": None,
        "liquidity_ok": None,
    }


def _prepare_intraday(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    out = df.copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    out = out[out["Volume"] > 0]
    if out.empty:
        return pd.DataFrame()

    timestamps = pd.to_datetime(out.index)
    if getattr(timestamps, "tz", None) is None:
        timestamps = timestamps.tz_localize(US_EASTERN)
    else:
        timestamps = timestamps.tz_convert(US_EASTERN)

    out["timestamp_eastern"] = timestamps
    out = out[_regular_session_mask(out["timestamp_eastern"])]
    if out.empty:
        return pd.DataFrame()

    out["session_date"] = out["timestamp_eastern"].dt.date
    out["session_time"] = out["timestamp_eastern"].dt.time
    return out.sort_values("timestamp_eastern").reset_index(drop=True)


def _regular_session_mask(series: pd.Series) -> pd.Series:
    session_start = time(9, 30)
    session_end = time(16, 0)
    session_times = series.dt.time
    return (session_times >= session_start) & (session_times <= session_end)


def _sum_until(df: pd.DataFrame, end: time) -> float:
    return float(df[df["session_time"] <= end]["Volume"].sum())


def _sum_window(df: pd.DataFrame, start: time, end: time) -> float:
    if start <= end:
        mask = (df["session_time"] > start) & (df["session_time"] <= end)
    else:
        mask = df["session_time"] <= end
    return float(df[mask]["Volume"].sum())


def _average_session_sum(df: pd.DataFrame, start: Optional[time], end: time) -> Optional[float]:
    values: List[float] = []
    for _, session in df.groupby("session_date"):
        value = _sum_until(session, end) if start is None else _sum_window(session, start, end)
        if value > 0:
            values.append(value)

    if len(values) < MIN_HISTORICAL_INTRADAY_SESSIONS:
        return None

    return float(sum(values) / len(values))


def _vwap(df: pd.DataFrame) -> Optional[float]:
    volume = df["Volume"].sum()
    if volume in (None, 0):
        return None
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    return float((typical_price * df["Volume"]).sum() / volume)


def _avg_daily_dollar_volume(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        return None
    recent = df.dropna(subset=["Close", "Volume"]).tail(LOOKBACK_SESSIONS)
    if len(recent) < MIN_HISTORICAL_INTRADAY_SESSIONS:
        return None
    return float((recent["Close"] * recent["Volume"]).mean())


def _score_volume_interest(
    cumulative_rvol: Optional[float],
    last_2h_ratio: Optional[float],
    last_30m_ratio: Optional[float],
    acceleration: Optional[float],
    above_vwap: Optional[bool],
    day_range_position: Optional[float],
    extended: Optional[bool],
    liquidity_ok: Optional[bool],
) -> float:
    score = 0.0
    score += _tier(cumulative_rvol, [(3.0, 3.0), (2.0, 2.5), (1.5, 2.0), (1.2, 1.0)])
    score += _tier(last_2h_ratio, [(3.0, 2.0), (2.0, 1.6), (1.5, 1.2), (1.2, 0.6)])
    score += min(
        _tier(last_30m_ratio, [(2.0, 0.9), (1.5, 0.65), (1.2, 0.35)])
        + _tier(acceleration, [(1.5, 0.6), (1.2, 0.35)]),
        1.5,
    )

    if above_vwap is True:
        score += 1.0

    if day_range_position is not None:
        if day_range_position >= 80:
            score += 1.0
        elif day_range_position >= 65:
            score += 0.7
        elif day_range_position >= 50:
            score += 0.35

    if extended is False:
        score += 1.0
    elif extended is None:
        score += 0.25

    if liquidity_ok is True:
        score += 0.5

    return min(score, 10.0)


def _tier(value: Optional[float], levels: List[tuple]) -> float:
    if value is None:
        return 0.0
    for threshold, points in levels:
        if value >= threshold:
            return points
    return 0.0


def _volume_signal(
    cumulative_rvol: Optional[float],
    last_2h_ratio: Optional[float],
    above_vwap: Optional[bool],
    day_range_position: Optional[float],
    intraday_change: Optional[float],
    extended: Optional[bool],
) -> str:
    high_rvol = max(cumulative_rvol or 0, last_2h_ratio or 0)

    if high_rvol >= 3.0 and extended is True:
        return VOLUME_SIGNAL_CHASE

    if high_rvol >= 1.5 and above_vwap is False and (day_range_position or 100) <= 35:
        return VOLUME_SIGNAL_DISTRIBUTION

    if (
        (cumulative_rvol or 0) >= 1.5
        and (last_2h_ratio or 0) >= 1.5
        and above_vwap is True
        and (day_range_position or 0) >= 65
        and extended is False
    ):
        return VOLUME_SIGNAL_STRONG

    if (
        high_rvol >= 1.5
        and above_vwap is True
        and intraday_change is not None
        and -1.0 <= intraday_change <= 4.0
        and extended is False
    ):
        return VOLUME_SIGNAL_ACCUMULATION

    if high_rvol >= 1.2:
        return VOLUME_SIGNAL_MILD

    return VOLUME_SIGNAL_NO_UNUSUAL


def _volume_explanation_he(signal: str, liquidity_ok: Optional[bool]) -> str:
    explanations = {
        VOLUME_SIGNAL_STRONG: "ווליום חריג חיובי: המחיר מעל VWAP והווליום בשעתיים האחרונות גבוה מהממוצע.",
        VOLUME_SIGNAL_ACCUMULATION: "ייתכן איסוף: ווליום חריג בלי תזוזת מחיר גדולה, והמחיר מחזיק מעל VWAP.",
        VOLUME_SIGNAL_CHASE: "סיכון רדיפה: הווליום חריג אך המחיר כבר עלה חזק מדי.",
        VOLUME_SIGNAL_DISTRIBUTION: "סיכון הפצה: ווליום חריג אך המחיר מתחת ל-VWAP ובחלק התחתון של הטווח היומי.",
        VOLUME_SIGNAL_MILD: "יש עניין ווליום מתון, אבל אין עדיין אישור מספיק חזק.",
        VOLUME_SIGNAL_NO_UNUSUAL: "אין ווליום חריג ביחס לשעה הנוכחית.",
        VOLUME_SIGNAL_INSUFFICIENT: "אין מספיק נתונים לחישוב ווליום חריג.",
    }
    text = explanations.get(signal, explanations[VOLUME_SIGNAL_INSUFFICIENT])
    if liquidity_ok is False:
        text += " נזילות נמוכה יחסית, לכן הציון מוגבל ויש סיכון החלקה."
    return text


def _extended_intraday(
    intraday_change: Optional[float],
    price_vs_vwap_pct: Optional[float],
) -> Optional[bool]:
    if intraday_change is None and price_vs_vwap_pct is None:
        return None
    return (
        (intraday_change or 0) > EXTENDED_INTRADAY_PCT
        or (price_vs_vwap_pct or 0) > EXTENDED_VWAP_DISTANCE_PCT
    )


def _day_range_position(
    price: Optional[float],
    low: Optional[float],
    high: Optional[float],
) -> Optional[float]:
    if price is None or low is None or high is None or high <= low:
        return None
    return max(0.0, min(100.0, ((price - low) / (high - low)) * 100))


def _pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous in (None, 0):
        return None
    return ((current - previous) / previous) * 100


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 2)


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

    technical["volume_interest"] = analyze_volume(ticker)
    technical["fundamentals"] = get_fundamentals(ticker)
    technical["latest_news"] = get_latest_news(ticker, limit=8, lookback_hours=72)

    return technical


def scan_watchlist(tickers: Optional[List[str]] = None, max_results: int = 25) -> Dict[str, Any]:
    """
    Fast technical scan only.
    Does NOT call fundamentals, to avoid timeouts.
    """
    tickers = tickers or DEFAULT_TICKERS
    qqq_20d = qqq_return_20d()

    found: List[Dict[str, Any]] = []
    volume_cache: Dict[str, Dict[str, Any]] = {}
    errors = 0

    for ticker in tickers:
        normalized_ticker = ticker.strip().upper()
        result = _technical_analysis(normalized_ticker, qqq_20d=qqq_20d)

        if result is None:
            errors += 1
            continue

        for setup in result.get("setups", []):
            if normalized_ticker not in volume_cache:
                volume_cache[normalized_ticker] = analyze_volume(normalized_ticker)

            volume_interest = volume_cache[normalized_ticker]
            volume_score = volume_interest.get("volume_interest_score") or 0
            setup["volume_interest"] = volume_interest
            setup["volume_interest_score"] = volume_interest.get("volume_interest_score")
            setup["combined_score"] = round(
                (setup.get("score") or 0) + min(float(volume_score), 3.0),
                2,
            )
            found.append(setup)

    found.sort(
        key=lambda x: (
            x.get("combined_score") or x.get("score", 0),
            x.get("score", 0),
            x.get("volume_interest_score") or 0,
            x.get("rs_20d_vs_qqq") or -999,
        ),
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
