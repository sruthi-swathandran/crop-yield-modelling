"""
Probe the NASS Quick Stats API with a single request and report what
comes back.

    python src/00_probe_nass.py

This is not part of the pipeline. It exists to answer questions that
should be answered before any parsing code is written:

  - what fields does a record actually have
  - how is the yield value typed, number or string
  - are there multiple short_desc variants for "corn yield"
  - how are counties identified, FIPS or name or both
  - what does a suppressed or missing value look like

Writing a parser against documentation and discovering the answers
later is how the Airbnb load in the SQL portfolio went wrong four
separate ways. One request now is cheaper.
"""

import json
import os
from collections import Counter

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("NASS_API_KEY")

if not KEY:
    raise SystemExit("NASS_API_KEY not found. Put it in .env at the repo root.")

URL = "https://quickstats.nass.usda.gov/api/api_GET/"

# Deliberately broad. Narrowing the query before seeing what exists
# would hide the variants we need to know about.
params = {
    "key": KEY,
    "source_desc": "SURVEY",
    "sector_desc": "CROPS",
    "commodity_desc": "CORN",
    "statisticcat_desc": "YIELD",
    "agg_level_desc": "COUNTY",
    "state_alpha": "IA",
    "year": "2020",
    "format": "JSON",
}

print("requesting ...")
r = requests.get(URL, params=params, timeout=60)
print(f"status {r.status_code}")

if r.status_code != 200:
    print(r.text[:2000])
    raise SystemExit("request failed")

payload = r.json()
records = payload.get("data", [])
print(f"records: {len(records)}")

if not records:
    print(json.dumps(payload)[:2000])
    raise SystemExit("no records returned")

first = records[0]

print("\n--- fields on a record ---")
for k in sorted(first):
    print(f"  {k}")

print("\n--- first record ---")
print(json.dumps(first, indent=2))

print("\n--- distinct short_desc (what counts as 'corn yield') ---")
for desc, n in Counter(r_["short_desc"] for r_ in records).most_common():
    print(f"  {n:>4}  {desc}")

print("\n--- distinct unit_desc ---")
for u, n in Counter(r_.get("unit_desc") for r_ in records).most_common():
    print(f"  {n:>4}  {u}")

print("\n--- how Value is typed, and what non-numeric values appear ---")
vals = [r_.get("Value") for r_ in records]
print(f"  python type of first Value: {type(vals[0]).__name__}")
odd = sorted({v for v in vals if isinstance(v, str) and not v.replace(",", "").replace(".", "").isdigit()})
print(f"  non-numeric Values: {odd if odd else 'none'}")

print("\n--- county identification ---")
sample = records[:3]
for r_ in sample:
    print(f"  {r_.get('state_fips_code')}-{r_.get('county_code')}  "
          f"{r_.get('county_name')}  ({r_.get('county_ansi')})")

print(f"\n  distinct counties in response: {len({r_.get('county_code') for r_ in records})}")
print("  Iowa has 99 counties.")
