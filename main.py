from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import your agent — completely unchanged. The web layer just wraps it.

from agent import (run_analysis, list_current_matches,
                   get_scorecard_only, predict_match)


app = FastAPI(title="Cricket Analyst Agent")


# --- Request body shape for the analyze endpoint ---
class AnalyzeRequest(BaseModel):
    match_id: str = ""


# --- Endpoint 1: list current matches (for the dropdown) ---
@app.get("/matches")
def get_matches(kind: str = "recent"):
    return {"matches": list_current_matches(kind)}


@app.get("/scorecard")
def get_scorecard(match_id: str):
    return {"scorecard": get_scorecard_only(match_id)}


# --- Endpoint 2: run the agent on a chosen match ---
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = run_analysis(req.match_id)
    return {
        "stats": result["stats"],
        "turning_points": result["turning_points"],
        "analysis": result["analysis"],
        "revisions": result["revision_count"],
        "approved": result["approved"],
        "scorecard": result.get("scorecard", {}),   # <-- ADD
    }

# --- Endpoint 3: serve the UI ---
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("/predict")
def predict(match_id: str, match_type: str = "", status: str = ""):
    return {"prediction": predict_match(match_id, match_type, status)}
    
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)