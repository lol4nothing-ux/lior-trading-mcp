from fastapi import FastAPI
from scanner_core import scan_market

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Lior Trading MCP"
    }

@app.get("/scan")
def scan():
    results = scan_market()
    return {
        "results": results
    }
