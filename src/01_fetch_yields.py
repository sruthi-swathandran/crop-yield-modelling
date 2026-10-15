"""
Fetch county corn yields from the USDA NASS Quick Stats API.

    python src/01_fetch_yields.py

Writes data/nass_corn_yields.csv, one row per county-year.

WHAT 00_probe_nass.py ESTABLISHED, AND WHY EACH THING MATTERS HERE
------------------------------------------------------------------
Value arrives as a string. "183.9", not 183.9. Cast explicitly and
report anything that fails to cast rather than coercing it to NaN
and moving on.

"Corn yield" is two commodities. A single Iowa year returned 96
records of grain in bushels per acre and 62 of silage in tons per
acre. They are filtered apart here rather than at the API, so the
count of what was dropped is visible.

County FIPS is state_fips_code concatenated with county_code, giving
five digits. Names are not used for joining: the Olympic project lost
five zones to inconsistent naming and that was with 265 of them.

asd_code identifies the Agricultural Statistics District, a sub-state
grouping of counties. It is kept because it is the spatial unit the
cross-validation will block on, and it arrives free.

CV (%) is the coefficient of variation NASS attaches to each estimate.
Kept so that model error can later be checked against the survey's own
uncertainty.

SUPPRESSION
-----------
NASS withholds county figures when too few farms report. The probe
found no suppressed values in one state-year, which is not evidence
that none exist. Anything that will not parse as a number is counted
and reported by state and year rather than silently dropped.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("NASS_API_KEY")
if not KEY:
    raise SystemExit("NASS_API_KEY not found. Put it in .env at the repo root.")

URL = "https://quickstats.nass.usda.gov/api/api_GET/"
OUT = Path(__file__).resolve().parent.parent / "data" / "nass_corn_yields.csv"

# The Corn Belt. Chosen for density of corn production rather than for
# convenience: these states contain most of the counties where a yield
# model has enough signal to be worth building.
STATES = ["IA", "IL", "IN", "NE", "MN", "OH", "MO", "SD"]

# 2008 is not arbitrary. The Cropland Data Layer, which supplies the
# per-county corn mask in 03, has patchy coverage before then.
YEAR_FROM, YEAR_TO = 2008, 2023

# Columns worth keeping out of the 39 the API returns.
KEEP = [
    "state_alpha", "state_fips_code", "county_code", "county_name",
    "asd_code", "asd_desc", "year", "Value", "CV (%)",
    "util_practice_desc", "prodn_practice_desc", "unit_desc", "short_desc",
]


def fetch_state(state: str) -> pd.DataFrame:
    """One request per state, all years. NASS caps a response at 50,000
    records; a state across 16 years returns a few thousand."""
    params = {
        "key": KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "agg_level_desc": "COUNTY",
        "state_alpha": state,
        "year__GE": YEAR_FROM,
        "year__LE": YEAR_TO,
        "format": "JSON",
    }
    r = requests.get(URL, params=params, timeout=120)
    if r.status_code != 200:
        print(f"  {state}: HTTP {r.status_code}  {r.text[:200]}")
        return pd.DataFrame()
    data = r.json().get("data", [])
    return pd.DataFrame(data)


def main() -> None:
    # Defaults reproduce the original fetch. For the out-of-time test:
    #   python src/01_fetch_yields.py --years 2022-2024 --out data/nass_corn_yields_recent.csv
    global YEAR_FROM, YEAR_TO, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--years")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.years:
        YEAR_FROM, YEAR_TO = (int(x) for x in args.years.split("-"))
    if args.out:
        OUT = args.out if args.out.is_absolute() else Path.cwd() / args.out
    print(f"years {YEAR_FROM}-{YEAR_TO}  ->  {OUT}")

    frames = []
    for state in STATES:
        df = fetch_state(state)
        print(f"  {state}: {len(df):>6} records")
        if len(df):
            frames.append(df[KEEP])
        time.sleep(1)  # courtesy, not a documented requirement

    raw = pd.concat(frames, ignore_index=True)
    print(f"\ntotal records returned: {len(raw):,}")

    # --- grain against silage -------------------------------------
    counts = raw.util_practice_desc.value_counts()
    print("\nutil_practice_desc:")
    for k, v in counts.items():
        print(f"  {v:>7,}  {k}")

    grain = raw[raw.util_practice_desc == "GRAIN"].copy()
    print(f"\nkeeping GRAIN: {len(grain):,} records, "
          f"dropping {len(raw) - len(grain):,}")

    # --- production practice --------------------------------------
    # The first run of this script produced 1,875 duplicate
    # county-years. The cause is here: NASS reports irrigated and
    # non-irrigated corn separately AND as a combined total, so a
    # Nebraska county-year can appear three times. Taking all three
    # would train the model on the same county three times with
    # different yields.
    #
    # ALL PRODUCTION PRACTICES is the combined figure and the one
    # that matches a whole-county satellite average.
    print("\nprodn_practice_desc:")
    for k, v in grain.prodn_practice_desc.value_counts().items():
        print(f"  {v:>7,}  {k}")

    grain = grain[grain.prodn_practice_desc == "ALL PRODUCTION PRACTICES"].copy()
    print(f"  keeping ALL PRODUCTION PRACTICES: {len(grain):,}")

    # --- combined-county records ----------------------------------
    # county_code 998 is NASS's "OTHER (COMBINED) COUNTIES", an
    # aggregate of counties too small to report individually. It has
    # no boundary and no crop mask, so it cannot be joined to
    # anything. Four states returned one county more than they have.
    combined = grain[grain.county_code == "998"]
    print(f"\ncombined-county records (county_code 998): {len(combined):,}")
    grain = grain[grain.county_code != "998"].copy()

    # --- period ---------------------------------------------------
    # year__LE was ignored by the API, which returned through 2025.
    # Filtered here instead. The ceiling is set by the Cropland Data
    # Layer used in 03 rather than by NASS.
    before = len(grain)
    grain = grain[grain.year.between(YEAR_FROM, YEAR_TO)].copy()
    print(f"restricting to {YEAR_FROM}-{YEAR_TO}: dropped {before - len(grain):,}")

    # One unit, or the numbers are not comparable.
    units = grain.unit_desc.unique()
    assert list(units) == ["BU / ACRE"], f"unexpected units: {units}"

    # --- the string-to-number cast --------------------------------
    grain["yield_bu_acre"] = pd.to_numeric(
        grain.Value.str.replace(",", "", regex=False), errors="coerce"
    )
    bad = grain[grain.yield_bu_acre.isna()]
    print(f"\nvalues that would not parse as a number: {len(bad):,}")
    if len(bad):
        print("  distinct raw values:", sorted(bad.Value.unique())[:10])
        print(bad.groupby(["state_alpha", "year"]).size().to_string())
    grain = grain[grain.yield_bu_acre.notna()]

    grain["cv_pct"] = pd.to_numeric(
        grain["CV (%)"].replace("", pd.NA), errors="coerce"
    )

    # --- keys -----------------------------------------------------
    grain["fips"] = grain.state_fips_code.str.zfill(2) + grain.county_code.str.zfill(3)
    grain["asd"] = grain.state_fips_code.str.zfill(2) + grain.asd_code.str.zfill(2)

    out = grain[[
        "fips", "state_alpha", "county_name", "asd", "asd_desc",
        "year", "yield_bu_acre", "cv_pct",
    ]].sort_values(["fips", "year"]).reset_index(drop=True)

    # --- checks, stated before they run ---------------------------
    print("\n--- checks ---")
    print(f"  rows                : {len(out):,}")
    print(f"  distinct counties   : {out.fips.nunique():,}")
    print(f"  distinct ASDs       : {out.asd.nunique():,}")
    print(f"  years               : {out.year.min()} to {out.year.max()}")
    print(f"  duplicate county-yrs: {out.duplicated(['fips','year']).sum()}")
    print(f"  yield range         : {out.yield_bu_acre.min():.1f} "
          f"to {out.yield_bu_acre.max():.1f} bu/acre")
    print(f"  CV missing          : {out.cv_pct.isna().sum():,} "
          f"({100*out.cv_pct.isna().mean():.1f}%)")

    # Corn yields outside roughly 40 to 260 bu/acre are not credible for
    # these states. Reported, not removed: a real outlier is a drought
    # year and belongs in the training data.
    odd = out[(out.yield_bu_acre < 40) | (out.yield_bu_acre > 260)]
    print(f"  outside 40-260      : {len(odd)}")
    if len(odd):
        print(odd.head(10).to_string(index=False))

    print("\n  counties per state:")
    print(out.groupby("state_alpha").fips.nunique().to_string())

    print("\n  county-years per year:")
    print(out.groupby("year").size().to_string())

    # The panel is unbalanced and shrinking, because NASS has been
    # reducing county-level estimates. That is a property of the
    # survey rather than of the counties, and it matters for the
    # evaluation: a held-out recent year is a smaller and
    # differently-composed sample than a held-out early one.
    print("\n  mean yield by year (2012 should be the trough):")
    print(out.groupby("year").yield_bu_acre.mean().round(1).to_string())

    # NASS revises published estimates. Where this fetch overlaps the
    # original one, say whether any county-year changed, because the
    # training data came from the original.
    original = Path(__file__).resolve().parent.parent / "data" / "nass_corn_yields.csv"
    if OUT != original and original.exists():
        old = pd.read_csv(original, dtype={"fips": str})
        both = out.merge(old[["fips", "year", "yield_bu_acre"]],
                         on=["fips", "year"], suffixes=("", "_original"))
        if len(both):
            changed = both[(both.yield_bu_acre - both.yield_bu_acre_original).abs() > 1e-9]
            print(f"\n  overlap with the original fetch: {len(both):,} county-years, "
                  f"{len(changed)} revised since")
            if len(changed):
                print(changed.head(10).to_string(index=False))

    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
