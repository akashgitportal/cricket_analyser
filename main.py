from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import your agent — completely unchanged. The web layer just wraps it.
from agent import run_analysis, list_current_matches

app = FastAPI(title="Cricket Analyst Agent")


# --- Request body shape for the analyze endpoint ---
class AnalyzeRequest(BaseModel):
    match_id: str = ""


# --- Endpoint 1: list current matches (for the dropdown) ---
@app.get("/matches")
def get_matches():
    return {"matches": list_current_matches()}


# --- Endpoint 2: run the agent on a chosen match ---
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = run_analysis(req.match_id)
    # Return just what the UI needs (the graph state has more than we display).
    return {
        "stats": result["stats"],
        "turning_points": result["turning_points"],
        "analysis": result["analysis"],
        "revisions": result["revision_count"],
        "approved": result["approved"],
    }


# --- Endpoint 3: serve the UI ---
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()