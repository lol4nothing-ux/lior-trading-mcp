from fastmcp import FastMCP
import scanner_core

mcp = FastMCP("Lior Trading Scanner")


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
    return scanner_core.scan_watchlist()


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
