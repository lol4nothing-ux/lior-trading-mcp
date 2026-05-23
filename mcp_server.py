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

app = mcp.http_app(path="/mcp")
