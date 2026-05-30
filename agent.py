import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
CRICKET_API_KEY = os.getenv("CRICKET_API_KEY")

# ---------------------------------------------------------------
# THE STATE — our shared "clipboard"
# ---------------------------------------------------------------
class MatchState(TypedDict):
    match_id: str
    raw_data: dict
    stats: dict
    turning_points: list
    analysis: str
    critique: str
    revision_count: int
    approved: bool


# ---------------------------------------------------------------
# NODE: fetch_match — for now, hands over the mock innings.
# (Later this is the only node that changes to call a real API.)
# ---------------------------------------------------------------
def fetch_match(state: MatchState) -> dict:
    print("→ [fetch_match] calling live cricket API")

    # Call the live API for current matches.
    url = "https://api.cricapi.com/v1/matches?apikey=26cd9747-30bb-4b7e-870e-8ee5926ecf3f&offset=0"
    resp = requests.get(url, params={"apikey": CRICKET_API_KEY, "offset": 0})
    payload = resp.json()

    matches = payload.get("data", [])
    if not matches:
        raise ValueError("No matches returned by the API right now.")

    # Pick the match: if the user gave a real id, find it; else take the first
    # one that actually has a score recorded.
    chosen = None
    wanted_id = state.get("match_id", "")
    for m in matches:
        if m.get("id") == wanted_id:
            chosen = m
            break
    if chosen is None:
        # fall back to the first match that has score data
        for m in matches:
            if m.get("score"):
                chosen = m
                break
    if chosen is None:
        chosen = matches[0]

    # --- Reshape API response into our clean internal structure ---
    raw = {
        "name": chosen.get("name", "Unknown match"),
        "status": chosen.get("status", ""),
        "venue": chosen.get("venue", ""),
        "match_type": chosen.get("matchType", ""),
        "innings": chosen.get("score", []),   # list of {r, w, o, inning}
    }

    print(f"   matched: {raw['name']}")
    return {"raw_data": raw}


# ---------------------------------------------------------------
# NODE: compute_stats — REAL cricket math now (pure Python).
# ---------------------------------------------------------------
def compute_stats(state: MatchState) -> dict:
    print("→ [compute_stats] crunching numbers")
    raw = state["raw_data"]
    innings_list = raw["innings"]

    # Build a clean per-innings stats list.
    innings_stats = []
    for inn in innings_list:
        runs = inn.get("r", 0)
        wickets = inn.get("w", 0)
        overs = inn.get("o", 0)
        # run rate = runs / overs, guard against divide-by-zero
        run_rate = round(runs / overs, 2) if overs else 0.0
        innings_stats.append({
            "inning": inn.get("inning", "Innings"),
            "runs": runs,
            "wickets": wickets,
            "overs": overs,
            "run_rate": run_rate,
        })

    stats = {
        "name": raw["name"],
        "status": raw["status"],
        "venue": raw["venue"],
        "match_type": raw["match_type"],
        "innings": innings_stats,
    }
    return {"stats": stats}


# ---------------------------------------------------------------
# NODES: still placeholders for now — we'll make these real next.
# ---------------------------------------------------------------
def find_turning_points(state: MatchState) -> dict:
    print("→ [find_turning_points] spotting key moments")
    s = state["stats"]

    # Format the innings into readable lines for the prompt.
    innings_text = "\n".join(
        f"- {i['inning']}: {i['runs']}/{i['wickets']} in {i['overs']} overs "
        f"(run rate {i['run_rate']})"
        for i in s["innings"]
    )

    prompt = f"""You are a sharp cricket analyst. Based ONLY on this match data, \
identify 2-3 factors that shaped the match so far.

CRITICAL RULES:
- You ONLY have team totals. You do NOT have player names or partnership data.
- NEVER invent players, partnerships, or specific events not present in the data.
- Base every point only on: run rate, wickets, overs, match status, and the toss decision.

Match: {s['name']}
Format: {s['match_type']}
Venue: {s['venue']}
Status: {s['status']}
Innings (all the data you have):
{innings_text}

List the key factors (grounded in the numbers only):"""

    response = llm.invoke(prompt)
    # Keep only lines that look like real list items (start with a number or bullet),
    # dropping any intro/preamble sentences.
    points = []
    for line in response.content.split("\n"):
        line = line.strip()
        if not line:
            continue
        # a real factor line usually starts with "1.", "2.", "-", "•" etc.
        if line[0].isdigit() or line[0] in "-•*":
            points.append(line)
    # fallback: if filtering removed everything, keep all non-empty lines
    if not points:
        points = [l.strip() for l in response.content.split("\n") if l.strip()]
    return {"turning_points": points}

def list_current_matches() -> list:
    """Return a lightweight list of current matches for the UI dropdown."""
    url = "https://api.cricapi.com/v1/matches?apikey=26cd9747-30bb-4b7e-870e-8ee5926ecf3f&offset=0"
    resp = requests.get(url, params={"apikey": CRICKET_API_KEY, "offset": 0})
    payload = resp.json()
    matches = payload.get("data", [])

    # Return only what the dropdown needs: id, name, and whether it has a score yet.
    result = []
    for m in matches:
        result.append({
            "id": m.get("id", ""),
            "name": m.get("name", "Unknown match"),
            "status": m.get("status", ""),
            "has_score": bool(m.get("score")),
        })
    return result

def write_analysis(state: MatchState) -> dict:
    count = state.get("revision_count", 0) + 1
    print(f"→ [write_analysis] writing draft (attempt {count})")
    s = state["stats"]
    points = "\n".join(state["turning_points"])

    feedback = state.get("critique", "")
    feedback_block = ""
    if feedback and count > 1:
        feedback_block = f"\n\nA reviewer gave this feedback — address it:\n{feedback}"

    innings_text = "\n".join(
        f"- {i['inning']}: {i['runs']}/{i['wickets']} in {i['overs']} overs "
        f"(run rate {i['run_rate']})"
        for i in s["innings"]
    )

    prompt = f"""You are a cricket analyst writing a concise match analysis. \
Write ONE tight paragraph (3-4 sentences).

CRITICAL RULES:
- You ONLY have the team totals below. You do NOT have player names, \
individual scores, or partnership details.
- NEVER invent player names, partnerships, or specific events. \
If you mention a player or partnership you were not given, that is a serious error.
- Analyse ONLY what the numbers support: run rate, wickets lost, match situation, \
the toss/bowling decision, and what the totals imply.

Match: {s['name']}
Format: {s['match_type']}
Status: {s['status']}
Innings (these totals are ALL the data you have):
{innings_text}

Key factors:
{points}{feedback_block}

Write a grounded analysis using only the totals above:"""

    response = llm.invoke(prompt)
    return {"analysis": response.content, "revision_count": count}


def critique(state: MatchState) -> dict:
    print("→ [critique] reviewing the draft")
    draft = state["analysis"]

    prompt = f"""You are a demanding editor reviewing a cricket analysis paragraph. \
Judge it on: specificity (uses real numbers), insight (explains WHY, not just WHAT), \
and clarity. 

Respond in EXACTLY this format:
VERDICT: APPROVE or REVISE
FEEDBACK: <one sentence — if APPROVE, say what works; if REVISE, say precisely what to fix>

Here is the analysis to review:
{draft}"""

    response = llm.invoke(prompt)
    text = response.content

    # Parse the model's verdict out of its response.
    approved = "APPROVE" in text.upper().split("FEEDBACK")[0]

    # Pull out the feedback line to pass back to the writer if needed.
    feedback = text
    if "FEEDBACK:" in text:
        feedback = text.split("FEEDBACK:", 1)[1].strip()

    print(f"   verdict: {'APPROVED' if approved else 'NEEDS REVISION'}")
    return {"critique": feedback, "approved": approved}


# ---------------------------------------------------------------
# CONDITIONAL EDGE
# ---------------------------------------------------------------
def should_continue(state: MatchState) -> str:
    if state["approved"] or state["revision_count"] >= 3:
        return "finalize"
    return "revise"

def run_analysis(match_id: str = "") -> dict:
    """Run the full agent graph and return the result dict.
    Called by both the CLI below and the FastAPI server."""
    initial_state = {"match_id": match_id, "revision_count": 0}
    return graph.invoke(initial_state)


def list_current_matches() -> list:
    """Return a lightweight list of current matches for the UI dropdown."""
    url = "https://api.cricapi.com/v1/matches?apikey=26cd9747-30bb-4b7e-870e-8ee5926ecf3f&offset=0"
    resp = requests.get(url, params={"apikey": CRICKET_API_KEY, "offset": 0})
    payload = resp.json()

    # --- DEBUG: show what the API actually said ---
    print("=" * 50)
    print("[list_current_matches] status:", payload.get("status"))
    print("[list_current_matches] info:", payload.get("info"))
    print("[list_current_matches] data count:", len(payload.get("data", [])))
    print("=" * 50)

    matches = payload.get("data", [])
    result = []
    for m in matches:
        result.append({
            "id": m.get("id", ""),
            "name": m.get("name", "Unknown match"),
            "status": m.get("status", ""),
            "has_score": bool(m.get("score")),
        })
    return result


# ---------------------------------------------------------------
# BUILD THE GRAPH (unchanged from Step 1)
# ---------------------------------------------------------------
builder = StateGraph(MatchState)
builder.add_node("fetch_match", fetch_match)
builder.add_node("compute_stats", compute_stats)
builder.add_node("find_turning_points", find_turning_points)
builder.add_node("write_analysis", write_analysis)
builder.add_node("critique", critique)

builder.add_edge(START, "fetch_match")
builder.add_edge("fetch_match", "compute_stats")
builder.add_edge("compute_stats", "find_turning_points")
builder.add_edge("find_turning_points", "write_analysis")
builder.add_edge("write_analysis", "critique")
builder.add_conditional_edges(
    "critique",
    should_continue,
    {"revise": "write_analysis", "finalize": END},
)

graph = builder.compile()

# Print an ASCII diagram of your graph structure
print(graph.get_graph().draw_ascii())

# ---------------------------------------------------------------
# RUN IT — now we print the real stats so you can verify the math
# ---------------------------------------------------------------
if __name__ == "__main__":
    initial_state = {"match_id": "", "revision_count": 0}   # empty = auto-pick a match
    result = graph.invoke(initial_state)

    s = result["stats"]
    print("\n" + "=" * 55)
    print(f"MATCH:  {s['name']}")
    print(f"VENUE:  {s['venue']}")
    print(f"STATUS: {s['status']}")
    print("-" * 55)
    for i in s["innings"]:
        print(f"  {i['inning']}: {i['runs']}/{i['wickets']} "
              f"in {i['overs']} ov (RR {i['run_rate']})")
    print("-" * 55)
    print("KEY FACTORS:")
    for p in result["turning_points"]:
        print("  •", p)
    print("-" * 55)
    print("ANALYSIS:")
    print(result["analysis"])
    print("-" * 55)
    print(f"REVISIONS: {result['revision_count']} | "
          f"VERDICT: {'approved' if result['approved'] else 'gave up'}")
    print("=" * 55)