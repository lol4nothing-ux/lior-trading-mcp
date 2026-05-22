from fastapi import FastAPI
import scanner_core

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Lior Trading MCP"
    }

@app.get("/analyze")
def analyze(ticker: str):
    result = scanner_core.analyze_ticker(ticker.upper())
    return result
