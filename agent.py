import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "cricbuzz-cricket2.p.rapidapi.com"   # <-- new provider
RAPIDAPI_HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
CRICKET_API_KEY = os.getenv("CRICKET_API_KEY")

# ---------------------------------------------------------------
# THE STATE — our shared "clipboard"
# ---------------------------------------------------------------
class MatchState(TypedDict):
    match_id: str
    raw_data: dict
    scorecard: dict
    stats: dict
    turning_points: list
    analysis: str
    critique: str
    revision_count: int
    approved: bool


# ---------------------------------------------------------------
# Overs helper (defined early so any function can use it)
# 6 balls = a full over, so "19.6" -> "20"; "12.3" stays "12.3".
# ---------------------------------------------------------------
def _fmt_overs(overs_raw) -> str:
    try:
        overs_val = float(overs_raw or 0)
    except (ValueError, TypeError):
        return str(overs_raw)
    whole = int(overs_val)
    balls = round((overs_val - whole) * 10)
    whole += balls // 6      # roll completed overs over
    balls = balls % 6
    return f"{whole}" if balls == 0 else f"{whole}.{balls}"


# ---------------------------------------------------------------
# Helper: dig through Cricbuzz's nested match-list structure
# ---------------------------------------------------------------
def _extract_matches(payload: dict) -> list:
    """Dig through Cricbuzz's nested structure and return a flat list of matches.
    Each item is the raw match dict with 'matchInfo' and (maybe) 'matchScore'."""
    flat = []
    for type_block in payload.get("typeMatches", []):
        for series in type_block.get("seriesMatches", []):
            wrapper = series.get("seriesAdWrapper", {})
            for match in wrapper.get("matches", []):
                flat.append(match)
    return flat


# ---------------------------------------------------------------
# NODE: fetch_match — find the match (live or recent) and flatten totals
# ---------------------------------------------------------------
def fetch_match(state: MatchState) -> dict:
    print("→ [fetch_match] calling Cricbuzz API")

    wanted_id = state.get("match_id", "")

    # Search BOTH live and recent so we find the match regardless of tab.
    all_matches = []
    for endpoint in ["live", "recent"]:
        url = f"https://{RAPIDAPI_HOST}/matches/v1/{endpoint}"
        resp = requests.get(url, headers=RAPIDAPI_HEADERS)
        if resp.status_code == 200:
            all_matches.extend(_extract_matches(resp.json()))

    if not all_matches:
        raise ValueError("No matches returned by the API right now.")

    # Find the requested match by id.
    chosen = None
    for m in all_matches:
        if str(m.get("matchInfo", {}).get("matchId", "")) == wanted_id:
            chosen = m
            break
    # Only fall back if NO id was requested at all.
    if chosen is None and not wanted_id:
        for m in all_matches:
            if m.get("matchScore"):
                chosen = m
                break
    if chosen is None:
        raise ValueError(f"Match id {wanted_id} not found in live or recent.")

    info = chosen.get("matchInfo", {})
    score = chosen.get("matchScore", {})

    innings = []
    for team_key, team_obj in [("team1", "team1Score"), ("team2", "team2Score")]:
        team_name = info.get(team_key, {}).get("teamName", "Team")
        team_score = score.get(team_obj, {})
        for inn_key in sorted(team_score.keys()):
            inn = team_score[inn_key]
            innings.append({
                "team": team_name,
                "runs": inn.get("runs", 0),
                "wickets": inn.get("wickets", 0),
                "overs": inn.get("overs", 0),
            })

    raw = {
        "match_id": str(info.get("matchId", "")),
        "name": f"{info.get('team1', {}).get('teamName', '?')} vs "
                f"{info.get('team2', {}).get('teamName', '?')}",
        "status": info.get("status", ""),
        "venue": f"{info.get('venueInfo', {}).get('ground', '')}, "
                 f"{info.get('venueInfo', {}).get('city', '')}",
        "match_type": info.get("matchFormat", ""),
        "innings": innings,
    }

    print(f"   matched: {raw['name']}")
    return {"raw_data": raw}


# ---------------------------------------------------------------
# NODE: fetch_scorecard — pull detailed player data for a match
# ---------------------------------------------------------------
def fetch_scorecard(state: MatchState) -> dict:
    print("→ [fetch_scorecard] pulling player detail")
    match_id = state["raw_data"].get("match_id", "")

    if not match_id:
        print("   no match_id; skipping scorecard")
        return {"scorecard": {"available": False, "innings": []}}

    url = f"https://{RAPIDAPI_HOST}/mcenter/v1/{match_id}/scard"
    resp = requests.get(url, headers=RAPIDAPI_HEADERS)

    if resp.status_code != 200:
        print(f"   scorecard unavailable ({resp.status_code}); continuing with totals only")
        return {"scorecard": {"available": False, "innings": []}}

    payload = resp.json()
    innings_detail = []

    for inn in payload.get("scorecard", []):
        # --- FIX: only include batsmen who ACTUALLY batted ---
        # In a live match the scorecard lists players yet to bat (0 balls, no
        # dismissal). Those look identical to a duck and cause hallucinated
        # dismissals, so we exclude them entirely.
        batted = [
            b for b in inn.get("batsman", [])
            if b.get("balls", 0) > 0 or b.get("outdec", "").strip()
        ]

        # Top 3 batsmen by runs (curated, for the LLM) — from those who batted
        batsmen = sorted(batted, key=lambda b: b.get("runs", 0), reverse=True)[:3]
        top_bat = [
            {"name": b.get("name", ""), "runs": b.get("runs", 0),
             "balls": b.get("balls", 0), "fours": b.get("fours", 0),
             "sixes": b.get("sixes", 0)}
            for b in batsmen
        ]

        # Top 3 bowlers by wickets (curated, for the LLM)
        bowlers = sorted(
            inn.get("bowler", []),
            key=lambda b: b.get("wickets", 0),
            reverse=True,
        )[:3]
        top_bowl = [
            {"name": b.get("name", ""), "wickets": b.get("wickets", 0),
             "runs": b.get("runs", 0), "overs": b.get("overs", "0")}
            for b in bowlers
        ]

        # Best partnership by total runs (curated, for the LLM)
        plist = inn.get("partnership", {}).get("partnership", [])
        best_p = max(plist, key=lambda p: p.get("totalruns", 0), default=None)
        best_partnership = None
        if best_p:
            best_partnership = {
                "bat1": best_p.get("bat1name", ""),
                "bat2": best_p.get("bat2name", ""),
                "runs": best_p.get("totalruns", 0),
                "balls": best_p.get("totalballs", 0),
            }

        # FULL batting list for UI display — also only those who batted
        full_batting = [
            {"name": b.get("name", ""), "runs": b.get("runs", 0),
             "balls": b.get("balls", 0), "fours": b.get("fours", 0),
             "sixes": b.get("sixes", 0), "sr": b.get("strkrate", ""),
             "out": b.get("outdec", "")}
            for b in batted
        ]
        full_bowling = [
            {"name": b.get("name", ""), "overs": b.get("overs", "0"),
             "maidens": b.get("maidens", 0), "runs": b.get("runs", 0),
             "wickets": b.get("wickets", 0), "econ": b.get("economy", "")}
            for b in inn.get("bowler", [])
        ]
        extras = inn.get("extras", {})

        # Fall of wickets (for the worm / run-progression chart)
        fow_list = inn.get("fow", {}).get("fow", [])
        fall_of_wickets = [
            {"name": f.get("batsmanname", ""), "runs": f.get("runs", 0),
             "over": f.get("overnbr", 0)}
            for f in fow_list
        ]

        # All partnerships (for the partnership bar chart)
        all_partnerships = [
            {"pair": f"{p.get('bat1name','')} & {p.get('bat2name','')}",
             "runs": p.get("totalruns", 0)}
            for p in plist
        ]

        innings_detail.append({
            "team": inn.get("batteamname", "Team"),
            "score": inn.get("score", 0),
            "wickets": inn.get("wickets", 0),
            "overs": inn.get("overs", 0),
            "top_batsmen": top_bat,
            "top_bowlers": top_bowl,
            "best_partnership": best_partnership,
            "full_batting": full_batting,
            "full_bowling": full_bowling,
            "extras_total": extras.get("total", 0),
            "fall_of_wickets": fall_of_wickets,
            "all_partnerships": all_partnerships,
        })

    return {"scorecard": {"available": True, "innings": innings_detail}}


def get_scorecard_only(match_id: str) -> dict:
    """Fetch just the scorecard for a match id — no agent run."""
    fake_state = {"raw_data": {"match_id": match_id}}
    return fetch_scorecard(fake_state)["scorecard"]


def _format_scorecard(scorecard: dict) -> str:
    """Turn curated scorecard data into readable lines for the LLM prompt."""
    if not scorecard.get("available"):
        return "No detailed scorecard available — analyse from totals only."

    lines = []
    for inn in scorecard.get("innings", []):
        lines.append(f"\n{inn['team']} — {inn['score']}/{inn['wickets']} ({inn['overs']} ov)")

        bat = inn.get("top_batsmen", [])
        if bat:
            bat_str = ", ".join(
                f"{b['name']} {b['runs']}({b['balls']})" for b in bat
            )
            lines.append(f"  Top batting: {bat_str}")

        bowl = inn.get("top_bowlers", [])
        if bowl:
            bowl_str = ", ".join(
                f"{b['name']} {b['wickets']}/{b['runs']}" for b in bowl
            )
            lines.append(f"  Top bowling: {bowl_str}")

        p = inn.get("best_partnership")
        if p:
            lines.append(f"  Best stand: {p['bat1']} & {p['bat2']} ({p['runs']} runs)")

    return "\n".join(lines)


def _valid_player_names(scorecard: dict) -> set:
    """Collect every real player name in the scorecard, for fact-checking."""
    names = set()
    for inn in scorecard.get("innings", []):
        for b in inn.get("full_batting", []):
            if b.get("name"):
                names.add(b["name"])
        for b in inn.get("full_bowling", []):
            if b.get("name"):
                names.add(b["name"])
    return names


# Shared accuracy rules injected into the writing prompts.
ACCURACY_RULES = """CRITICAL ACCURACY RULES:
- The wickets number in the totals is AUTHORITATIVE. If an innings shows "/0", NO player is out — never describe any dismissal or collapse for that innings.
- Only discuss players who appear in the player detail below. NEVER name a player who is not listed.
- If the match status shows it is still in progress, describe it as ongoing — do not invent a final result or a collapse that has not happened.
- Never contradict the numbers. If unsure, describe less rather than inventing."""


# ---------------------------------------------------------------
# NODE: compute_stats — pure Python cricket math
# ---------------------------------------------------------------
def compute_stats(state: MatchState) -> dict:
    print("→ [compute_stats] crunching numbers")
    raw = state["raw_data"]

    innings_stats = []
    for inn in raw["innings"]:
        runs = inn.get("runs", 0) or 0
        wickets = inn.get("wickets", 0) or 0
        overs_raw = inn.get("overs", 0) or 0      # guard against None

        # Convert "19.6" (19 overs, 6 balls) to true overs safely.
        try:
            whole = int(overs_raw)
            balls = round((float(overs_raw) - whole) * 10)
            true_overs = whole + balls / 6
            run_rate = round(runs / true_overs, 2) if true_overs > 0 else 0.0
        except (ValueError, TypeError, ZeroDivisionError):
            run_rate = 0.0

        innings_stats.append({
            "inning": inn.get("team", "Innings"),
            "runs": runs,
            "wickets": wickets,
            "overs": _fmt_overs(overs_raw),
            "run_rate": run_rate,
        })

    stats = {
        "name": raw["name"],
        "status": raw["status"],
        "venue": raw["venue"],
        "match_type": raw["match_type"],
        "innings": innings_stats,
        "scorecard": state.get("scorecard", {}),
    }
    return {"stats": stats}


# ---------------------------------------------------------------
# NODE: find_turning_points — LLM picks the key moments
# ---------------------------------------------------------------
def find_turning_points(state: MatchState) -> dict:
    print("→ [find_turning_points] spotting key moments")
    s = state["stats"]

    innings_text = "\n".join(
        f"- {i['inning']}: {i['runs']}/{i['wickets']} in {i['overs']} overs "
        f"(run rate {i['run_rate']})"
        for i in s["innings"]
    )

    sc_text = _format_scorecard(s.get("scorecard", {}))

    prompt = f"""You are a sharp cricket analyst. Identify the 2-3 factors that most \
shaped this match, citing specific players, spells, or partnerships from the data.

{ACCURACY_RULES}

Match: {s['name']}
Format: {s['match_type']}
Venue: {s['venue']}
Status: {s['status']}

Team totals:
{innings_text}

Player detail (real, accurate names and numbers):
{sc_text}

List the key factors (cite ONLY real players/partnerships from the data above):"""

    response = llm.invoke(prompt)
    points = []
    for line in response.content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() or line[0] in "-•*":
            points.append(line)
    if not points:
        points = [l.strip() for l in response.content.split("\n") if l.strip()]
    return {"turning_points": points}


# ---------------------------------------------------------------
# Helper for the UI dropdown (called by FastAPI, not part of graph)
# ---------------------------------------------------------------
def list_current_matches(kind: str = "recent") -> list:
    """Return matches for the UI dropdown. kind = 'live' or 'recent'."""
    endpoint = "live" if kind == "live" else "recent"
    url = f"https://{RAPIDAPI_HOST}/matches/v1/{endpoint}"
    resp = requests.get(url, headers=RAPIDAPI_HEADERS)

    if resp.status_code != 200:
        raise ValueError(f"API returned {resp.status_code}: {resp.text[:120]}")
    payload = resp.json()

    result = []
    for match in _extract_matches(payload):
        info = match.get("matchInfo", {})
        score = match.get("matchScore", {})

        # Build a short score line per team that has batted, e.g. "AUS 280/6 (50)"
        score_parts = []
        for team_key, team_obj in [("team1", "team1Score"), ("team2", "team2Score")]:
            tname = info.get(team_key, {}).get("teamSName", "") or info.get(team_key, {}).get("teamName", "")
            tscore = score.get(team_obj, {})
            for inn_key in sorted(tscore.keys()):
                inn = tscore[inn_key]
                r = inn.get("runs", 0)
                w = inn.get("wickets", 0)
                o = _fmt_overs(inn.get("overs", 0))
                score_parts.append(f"{tname} {r}/{w} ({o})")

        result.append({
            "id": str(info.get("matchId", "")),
            "name": f"{info.get('team1', {}).get('teamName', '?')} vs "
                    f"{info.get('team2', {}).get('teamName', '?')} — "
                    f"{info.get('matchDesc', '')}",
            "status": info.get("status", ""),
            "scores": score_parts,          # list of per-innings score lines
            "has_score": bool(match.get("matchScore")),
        })
    return result


# ---------------------------------------------------------------
# NODE: write_analysis — LLM writes the narrative (with critic feedback loop)
# ---------------------------------------------------------------
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

    sc_text = _format_scorecard(s.get("scorecard", {}))

    prompt = f"""You are a cricket analyst writing a concise match analysis. \
Write ONE tight paragraph (4-5 sentences). Be insightful — explain WHY the match \
unfolded as it did, citing specific players and partnerships from the data below.

{ACCURACY_RULES}

Match: {s['name']}
Format: {s['match_type']}
Status: {s['status']}

Team totals:
{innings_text}

Player detail (use these REAL names and numbers — they are accurate):
{sc_text}

Key factors:
{points}{feedback_block}

Write a sharp, specific analysis grounded ONLY in the real player data above:"""

    response = llm.invoke(prompt)
    return {"analysis": response.content, "revision_count": count}


# ---------------------------------------------------------------
# NODE: critique — fact-checking editor; approves or sends back
# ---------------------------------------------------------------
def critique(state: MatchState) -> dict:
    print("→ [critique] fact-checking the draft")
    draft = state["analysis"]

    # Build the list of real player names + the authoritative wicket situation
    sc = state["stats"].get("scorecard", {})
    valid_names = _valid_player_names(sc)
    names_str = ", ".join(sorted(valid_names)) if valid_names else "(none available)"

    totals_str = "; ".join(
        f"{i['inning']} {i['runs']}/{i['wickets']}"
        for i in state["stats"].get("innings", [])
    )

    prompt = f"""You are a demanding fact-checking editor reviewing a cricket analysis.

Check the analysis against these rules and REVISE if ANY fail:
1. ACCURACY vs NUMBERS: The authoritative totals are: {totals_str}. \
A "/0" means NOBODY is out — reject any claim of a dismissal or collapse for such an innings.
2. REAL PLAYERS ONLY: Every player named in the analysis MUST be in this list: {names_str}. \
If it names ANYONE not in this list, that is a fabrication — REVISE.
3. NO INVENTED EVENTS: Reject any specific event (dismissal, partnership, milestone) not supported by the data.
4. INSIGHT & CLARITY: It should explain WHY using real numbers, and read clearly.

Respond in EXACTLY this format:
VERDICT: APPROVE or REVISE
FEEDBACK: <one sentence — if REVISE, name the exact problem, e.g. "names Babar Azam who is not in the scorecard" or "describes a collapse but the score is 19/0">

Analysis to review:
{draft}"""

    response = llm.invoke(prompt)
    text = response.content

    approved = "APPROVE" in text.upper().split("FEEDBACK")[0]

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


# ---------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------
builder = StateGraph(MatchState)
builder.add_node("fetch_match", fetch_match)
builder.add_node("fetch_scorecard", fetch_scorecard)
builder.add_node("compute_stats", compute_stats)
builder.add_node("find_turning_points", find_turning_points)
builder.add_node("write_analysis", write_analysis)
builder.add_node("critique", critique)

builder.add_edge(START, "fetch_match")
builder.add_edge("fetch_match", "fetch_scorecard")
builder.add_edge("fetch_scorecard", "compute_stats")
builder.add_edge("compute_stats", "find_turning_points")
builder.add_edge("find_turning_points", "write_analysis")
builder.add_edge("write_analysis", "critique")
builder.add_conditional_edges(
    "critique",
    should_continue,
    {"revise": "write_analysis", "finalize": END},
)

graph = builder.compile()


# ---------------------------------------------------------------
# CLI run (for terminal testing)
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