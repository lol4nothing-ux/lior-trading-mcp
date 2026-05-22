from __future__ import annotations

from typing import List, Optional
from fastmcp import FastMCP
from scanner_core import analyze_ticker, scan_watchlist

mcp = FastMCP("Lior Trading Scanner")

@mcp.tool()
def scan_market(tickers: Optional[List[str]] = None, max_results: int = 15) -> dict:
    """Scan a watchlist for near-breakout, pullback, and coiled technical setups."""
    return scan_watchlist(tickers=tickers, max_results=max_results)

@mcp.tool()
def analyze_stock(ticker: str) -> dict:
    """Analyze one stock ticker and return technical setup data."""
    result = analyze_ticker(ticker.upper())
    return result or {"ticker": ticker.upper(), "error": "No usable data returned from Yahoo Finance."}

if __name__ == "__main__":
    # For local dev: fastmcp run mcp_server.py --transport http --port 8000
    mcp.run()
