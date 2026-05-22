from fastapi import FastAPI
import scanner_core

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Lior Trading MCP",
        "available": dir(scanner_core)
    }

@app.get("/scan")
def scan():
    return {
        "message": "server online, scanner_core loaded",
        "available_functions": [
            name for name in dir(scanner_core)
            if not name.startswith("_")
        ]
    }
