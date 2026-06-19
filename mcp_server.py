from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
import scanner_core

mcp = FastMCP("Lior Trading Scanner")


VOLUME_NOTE = (
    "Volume analysis is applied selectively to BREAKOUT_NOW, "
    "NEAR_BREAKOUT and COILED candidates to avoid timeout."
)


def _volume_enrich_total_limit() -> int:
    return getattr(scanner_core, "VOLUME_ENRICH_TOTAL_LIMIT", 10)


def _volume_enrich_breakout_now_limit() -> int:
    return getattr(scanner_core, "VOLUME_ENRICH_BREAKOUT_NOW_LIMIT", 5)


def _volume_enrich_near_breakout_limit() -> int:
    return getattr(scanner_core, "VOLUME_ENRICH_NEAR_BREAKOUT_LIMIT", 5)


def _volume_enrich_coiled_limit() -> int:
    return getattr(scanner_core, "VOLUME_ENRICH_COILED_LIMIT", 3)


def _setup_label(setup: Dict[str, Any]) -> str:
    value = (
        setup.get("setup")
        or setup.get("type")
        or setup.get("setup_type")
        or setup.get("name")
        or ""
    )
    return str(value).strip().upper()


def _select_volume_enrichment_candidates(setups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    total_limit = _volume_enrich_total_limit()

    def add_matching(label: str, limit: int) -> None:
        added_for_label = 0
        for setup in setups:
            if len(selected) >= total_limit or added_for_label >= limit:
                return
            if id(setup) in selected_ids:
                continue
            if _setup_label(setup) != label:
                continue
            selected.append(setup)
            selected_ids.add(id(setup))
            added_for_label += 1

    add_matching("BREAKOUT_NOW", _volume_enrich_breakout_now_limit())
    add_matching("NEAR_BREAKOUT", _volume_enrich_near_breakout_limit())
    add_matching("COILED", _volume_enrich_coiled_limit())

    for setup in setups:
        if len(selected) >= total_limit:
            break
        if id(setup) in selected_ids:
            continue
        selected.append(setup)
        selected_ids.add(id(setup))

    return selected


def _fast_scan_watchlist(tickers: Optional[List[str]] = None, max_results: int = 25) -> Dict[str, Any]:
    """
    Two-stage scanner for the MCP tool.

    Stage A scans all tickers using technical logic only.
    Stage B enriches only selected high-priority candidates with intraday volume.
    """
    tickers = tickers or scanner_core.DEFAULT_TICKERS
    qqq_20d = scanner_core.qqq_return_20d()

    found: List[Dict[str, Any]] = []
    errors = 0

    for ticker in tickers:
        normalized_ticker = str(ticker).strip().upper()
        result = scanner_core._technical_analysis(normalized_ticker, qqq_20d=qqq_20d)

        if result is None:
            errors += 1
            continue

        for setup in result.get("setups", []):
            found.append(setup)

    found.sort(
        key=lambda x: (x.get("score", 0), x.get("rs_20d_vs_qqq") or -999),
        reverse=True,
    )

    returned = found[:max_results]
    volume_candidates = _select_volume_enrichment_candidates(returned)
    volume_candidate_ids = {id(setup) for setup in volume_candidates}
    volume_cache: Dict[str, Dict[str, Any]] = {}
    volume_enriched_count = 0

    for setup in returned:
        technical_score = setup.get("score") or 0
        setup["combined_score"] = technical_score
        setup["volume_interest"] = None
        setup["volume_interest_score"] = None
        setup["volume_enriched"] = False

        if id(setup) not in volume_candidate_ids:
            continue

        setup_ticker = str(setup.get("ticker", "")).strip().upper()
        if setup_ticker and setup_ticker not in volume_cache:
            try:
                volume_cache[setup_ticker] = scanner_core.analyze_volume(setup_ticker)
            except Exception:
                volume_cache[setup_ticker] = scanner_core._empty_volume_interest()

        volume_interest = volume_cache.get(setup_ticker) or scanner_core._empty_volume_interest()
        volume_score = volume_interest.get("volume_interest_score") or 0

        setup["volume_interest"] = volume_interest
        setup["volume_interest_score"] = volume_interest.get("volume_interest_score")
        setup["volume_enriched"] = True
        setup["combined_score"] = round(
            technical_score + min(float(volume_score), 3.0),
            2,
        )
        volume_enriched_count += 1

    returned.sort(
        key=lambda x: (
            x.get("combined_score") or x.get("score", 0),
            x.get("score", 0),
            x.get("volume_interest_score") or 0,
            x.get("rs_20d_vs_qqq") or -999,
        ),
        reverse=True,
    )

    return {
        "count": len(returned),
        "qqq_20d_return": round(qqq_20d, 2) if qqq_20d is not None else None,
        "errors": errors,
        "results": returned,
        "note": "Technical scan only. Run analyze_ticker on selected candidates for fundamentals.",
        "volume_note": VOLUME_NOTE,
        "volume_enriched_count": volume_enriched_count,
        "volume_enrich_limit": _volume_enrich_total_limit(),
    }


@mcp.tool()
def analyze_ticker(ticker: str):
    """
    Analyze a stock ticker using technical indicators.
    """
    return scanner_core.analyze_ticker(ticker.upper())


@mcp.tool()
def scan_watchlist():
    """
    Scan the default watchlist for setups.
    """
    return _fast_scan_watchlist()


@mcp.tool()
def market_map():
    """
    Return market and sector map.
    """
    return scanner_core.market_map()


@mcp.tool()
def analyze_volume(ticker: str):
    """
    Analyze unusual intraday volume and pre-close interest for a ticker.
    """
    return scanner_core.analyze_volume(ticker.upper())


app = mcp.http_app(path="/mcp")
