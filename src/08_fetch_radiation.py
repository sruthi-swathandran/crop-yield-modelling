"""
Daily solar radiation and weather from NASA POWER for every county,
2008 to 2021, aggregated onto the MODIS composite windows.

    python src/08_fetch_radiation.py

Run 08a_probe_power.py first and check the response shape.

Reads   data/counties.gpkg
Writes  data/power/raw/FIPS.parquet       one per county, daily
        data/weather_composites.parquet   one row per county-year-composite

WHY THIS EXISTS
---------------
The light use efficiency model needs the light. NDVI says how much of
the incoming radiation the canopy could intercept; it says nothing
about how much radiation arrived. A cloudy July and a clear one can
produce the same NDVI and very different crops.

WHAT IS FETCHED
---------------
ALLSKY_SFC_SW_DWN   shortwave reaching the ground, MJ/m^2/day
T2M_MAX, T2M_MIN    daily air temperature at 2 m, degrees C
PRECTOTCORR         precipitation, mm/day

Only shortwave feeds the light use efficiency model. The other three
cost nothing extra in the same request and are kept for diagnosing
2012 afterwards. They are not features of the satellite-only gradient
boosting model, which stays satellite-only so the comparison means
what it says.

Shortwave is stored as shortwave. Converting it to photosynthetically
active radiation needs a published fraction, and that conversion is a
modelling assumption, so it lives in the model script where it can be
cited, not here.

ONE POINT PER COUNTY
--------------------
POWER is a point API on a coarse grid. Each county is queried at its
centroid, computed in an equal-area projection so it is a true
centroid rather than one distorted by latitude. Neighbouring counties
often fall in the same grid cell and get identical series; the script
counts how many distinct series came back, so that is measured rather
than assumed.

RESUMING
--------
Each county is saved as soon as it arrives. If the run stops, run it
again and it carries on from where it was.
"""

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
YIELDS = ROOT / "data" / "nass_corn_yields.csv"
RAW = ROOT / "data" / "power" / "raw"
OUT = ROOT / "data" / "weather_composites.parquet"

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"]
START, END = "20080101", "20211231"

FIRST_DOY = 81          # same composite grid as 07
N_COMPOSITES = 14
PAUSE = 0.5             # seconds between requests, to be a polite client


def centroids() -> pd.DataFrame:
    """Centroid of every county that has a yield, in lon/lat."""
    c = gpd.read_file(COUNTIES, layer="counties")
    have = set(pd.read_csv(YIELDS, dtype={"fips": str}).fips)
    c = c[c.fips.isin(have)].copy()
    pts = c.to_crs("EPSG:5070").geometry.centroid.to_crs("EPSG:4326")
    return pd.DataFrame({"fips": c.fips.values,
                         "lon": pts.x.round(4).values,
                         "lat": pts.y.round(4).values})


def fetch(lon: float, lat: float, retries: int = 4) -> dict:
    q = {"parameters": ",".join(PARAMS), "community": "AG",
         "longitude": lon, "latitude": lat,
         "start": START, "end": END, "format": "JSON"}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=q, timeout=180)
            if r.status_code == 429:                 # rate limited
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} tries: {last}")


def to_frame(d: dict, fips: str) -> pd.DataFrame:
    fill = d.get("header", {}).get("fill_value", -999)
    p = d["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(p[k]) for k in PARAMS})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(fill, np.nan).astype("float64")
    df = df.rename(columns={"ALLSKY_SFC_SW_DWN": "sw", "T2M_MAX": "tmax",
                            "T2M_MIN": "tmin", "PRECTOTCORR": "precip"})
    df.index.name = "date"
    df = df.reset_index()
    df.insert(0, "fips", fips)
    return df


def download(pts: pd.DataFrame) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    todo = pts[~pts.fips.map(lambda f: (RAW / f"{f}.parquet").exists())]
    print(f"counties: {len(pts)}   already saved: {len(pts) - len(todo)}   "
          f"to fetch: {len(todo)}", flush=True)

    t0, failed = time.time(), []
    for n, row in enumerate(todo.itertuples(), 1):
        try:
            df = to_frame(fetch(row.lon, row.lat), row.fips)
            df.to_parquet(RAW / f"{row.fips}.parquet", index=False)
        except Exception as e:                       # noqa: BLE001
            failed.append(row.fips)
            print(f"  {row.fips} FAILED: {e}", flush=True)
            continue
        el = time.time() - t0
        left = (len(todo) - n) * el / n / 60
        print(f"  {n:>3}/{len(todo)}  {row.fips}  {len(df):,} days  "
              f"{el/60:5.1f} min elapsed, ~{left:5.1f} min left", flush=True)
        time.sleep(PAUSE)

    if failed:
        print(f"\n{len(failed)} counties failed. Run the script again to "
              f"retry only those: {failed[:10]}")


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(RAW.glob("*.parquet"))
    daily = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    daily["year"] = daily.date.dt.year
    doy = daily.date.dt.dayofyear
    daily["composite"] = (doy - FIRST_DOY) // 16 + 1
    daily = daily[daily.composite.between(1, N_COMPOSITES)]

    g = daily.groupby(["fips", "year", "composite"])
    out = pd.DataFrame({
        "sw_mj_m2": g.sw.sum(min_count=1),       # total over the 16 days
        "sw_days": g.sw.count(),                 # days with a real value
        "tmax_c": g.tmax.mean(),
        "tmin_c": g.tmin.mean(),
        "precip_mm": g.precip.sum(min_count=1),
    }).reset_index()
    return out, daily


def checks(out: pd.DataFrame, daily: pd.DataFrame) -> None:
    print("\n--- checks ---")
    print(f"  rows {len(out):,}  counties {out.fips.nunique()}  "
          f"years {out.year.min()}-{out.year.max()}  "
          f"composites {out.composite.min()}-{out.composite.max()}")
    short = (out.sw_days < 16).sum()
    print(f"  composites with fewer than 16 days of shortwave: {short}")
    print(f"  missing daily shortwave: {daily.sw.isna().sum():,} "
          f"of {len(daily):,}")

    # How many counties share a grid cell. Identical 14-year series
    # mean identical cells.
    sig = daily.groupby("fips").sw.apply(lambda s: round(float(s.sum()), 1))
    print(f"  distinct shortwave series: {sig.nunique()} across "
          f"{sig.size} counties")

    # Physical sanity. Midsummer clear-sky shortwave in the Corn Belt
    # is around 30 MJ/m^2/day; a 16-day total should sit well under
    # 16 x 30 = 480.
    print(f"  16-day shortwave, range: {out.sw_mj_m2.min():.0f} to "
          f"{out.sw_mj_m2.max():.0f} MJ/m^2")
    by_comp = out.groupby("composite").sw_mj_m2.mean().round(0)
    print("  mean 16-day shortwave by composite (should peak near 5 to 6, "
          "late May to June):")
    print("   " + "  ".join(f"{int(k)}:{int(v)}" for k, v in by_comp.items()))

    # 2012 should look hot and dry around silking. If it doesn't,
    # something is wrong with the join or the dates.
    silk = out[out.composite.isin([8, 9])]      # days 193 to 224
    t = silk.groupby("year")[["tmax_c", "precip_mm"]].mean().round(1)
    print("\n  Mid July to early August (composites 8 and 9), "
          "mean across counties:")
    print(t.to_string())
    print("  2012 should stand out: highest tmax, among the lowest rain.")


def main() -> None:
    # --recent fetches 2022 to 2024 for the out-of-time test into its own
    # folder and file, leaving the 2008 to 2021 data untouched.
    global RAW, OUT, START, END
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", action="store_true")
    if ap.parse_args().recent:
        RAW = ROOT / "data" / "power" / "raw_recent"
        OUT = ROOT / "data" / "weather_composites_recent.parquet"
        START, END = "20220101", "20241231"
    tee_to("power")
    print(f"{START} to {END}  ->  {OUT.name}")
    pts = centroids()
    download(pts)
    out, daily = aggregate()
    checks(out, daily)
    out.to_parquet(OUT, index=False)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
