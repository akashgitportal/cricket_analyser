import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("CRICKET_API_KEY")

# Step 1: get the list of current matches to find a match id
url = "https://api.cricapi.com/v1/currentMatches?apikey=26cd9747-30bb-4b7e-870e-8ee5926ecf3f&offset=0"
resp = requests.get(url, params={"apikey": KEY, "offset": 0})
data = resp.json()

# Pretty-print just the top-level structure so we can see the shape
print("STATUS:", data.get("status"))
print("INFO:", data.get("info"))   # tells you hits used / left

matches = data.get("data", [])
print(f"\nFound {len(matches)} matches. First one looks like:\n")
print(json.dumps(matches[0] if matches else {}, indent=2))