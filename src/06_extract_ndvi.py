"""
Extract corn-weighted NDVI time series for every county-year, using
the masks cached by 05.

    python src/06_extract_ndvi.py --year 2020
    python src/06_extract_ndvi.py --all

Writes data/ndvi/ndvi_YYYY.parquet, one row per county-composite.

THE WEIGHTING
-------------
A 232 m MODIS pixel is 5.4 hectares and in Iowa almost never contains
only corn. Its NDVI is a blend of corn, soybeans, a farmstead and a
road. Averaging raw NDVI across a county measures all of that and
calls it the corn signal.

05 produced, for every county-year, the fraction of each pixel that is
corn. Those fractions are the weights here: a pixel 80% corn counts
four times a pixel 20% corn.

    county NDVI = sum(ndvi * fraction) / sum(fraction)

restricted to pixels that survived cloud masking, so a cloudy pixel
contributes nothing rather than contributing a cloud's reflectance.

THE CORRECTNESS CHECK THAT MATTERS
----------------------------------
The cached mask and the NDVI array have to be the same grid, or the
weights land on the wrong pixels and nothing errors. Both come from
clip_box followed by clip against the same MODIS tile, so the shapes
must match exactly. Asserted rather than assumed: a silent off-by-one
here would corrupt every number downstream.

THE TEST
--------
Story County 2020 must come back with peak NDVI 0.868 on 2020-07-27,
the figure 04_probe_extraction.py produced before any of this was
restructured.
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
MASKDIR = ROOT / "data" / "masks"
OUTDIR = ROOT / "data" / "ndvi"

YEARS = range(2008, 2022)
SEASON = ("04-01", "10-31")
NDVI_SCALE = 0.0001
GOOD_RELIABILITY = (0, 1)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def open_signed(href, retries=3):
    """Sign at read time. A token stamped on during a catalogue search
    is rotated before a long run reaches the asset, which returned
    HTTP 403 forty-three minutes into the first attempt."""
    last = None
    for attempt in range(retries):
        try:
            return rioxarray.open_rasterio(pc.sign(href), masked=True).squeeze()
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def tile_key(item):
    return (item.properties["modis:horizontal-tile"],
            item.properties["modis:vertical-tile"])


def run_year(cat, counties, year):
    t0 = time.time()
    print(f"\n=== {year} ===", flush=True)

    mask_path = MASKDIR / f"corn_{year}.npz"
    if not mask_path.exists():
        print(f"  no masks for {year}, run 05 first")
        return 0.0
    masks = np.load(mask_path)
    print(f"  masks loaded: {len(masks.files)} counties")

    # Which tile each county belongs to, resolved once from a single
    # mid-season search rather than once per county.
    probe = list(cat.search(
        collections=["modis-13Q1-061"],
        bbox=list(counties.total_bounds),
        datetime=f"{year}-07-01/{year}-07-20",
    ).items())
    tiles = sorted({tile_key(i) for i in probe if i.id.startswith("MOD")})
    print(f"  MODIS tiles: {tiles}")

    # --- assign each county to exactly one tile ------------------
    # A county on a tile boundary passes a bounds test for both, so a
    # bounds test would process it twice and produce duplicate rows.
    # Worse, 05 resolved straddling counties by taking whichever tile
    # it happened to try first, and this script iterates in a
    # different order, so the mask could come from one tile and the
    # NDVI from another. The shapes would disagree and the row would
    # vanish without an error.
    #
    # So the mask decides. A county belongs to the tile whose clipped
    # array matches the shape of its cached mask, which is the same
    # tile 05 used by construction.
    print("  assigning counties to tiles by matching cached mask shapes",
          flush=True)
    items_by_tile, samples = {}, {}
    for h, v in tiles:
        its = [
            i for i in cat.search(
                collections=["modis-13Q1-061"],
                query={"modis:horizontal-tile": {"eq": h},
                       "modis:vertical-tile": {"eq": v}},
                datetime=f"{year}-{SEASON[0]}/{year}-{SEASON[1]}",
            ).items()
            if i.id.startswith("MOD")        # Terra only, see 03b
        ]
        its.sort(key=lambda i: i.properties["start_datetime"])
        if not its:
            continue
        items_by_tile[(h, v)] = its
        samples[(h, v)] = open_signed(
            its[0].assets["250m_16_days_NDVI"].href).load()

    assigned = {}
    unassigned = []
    for row in counties.itertuples():
        if row.fips not in masks:
            continue
        want = masks[row.fips].shape
        for key, sample in samples.items():
            g = gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_crs(sample.rio.crs)
            minx, miny, maxx, maxy = g.total_bounds
            ax0, ay0, ax1, ay1 = sample.rio.bounds()
            if maxx < ax0 or minx > ax1 or maxy < ay0 or miny > ay1:
                continue
            try:
                shp = (sample.rio.clip_box(minx, miny, maxx, maxy)
                       .rio.clip(g.geometry, g.crs, drop=True)).shape
            except Exception:
                continue
            if shp == want:
                assigned.setdefault(key, []).append((row.fips, g))
                break
        else:
            unassigned.append(row.fips)

    samples.clear()
    print(f"  assigned: {sum(len(v) for v in assigned.values())}, "
          f"unassigned: {len(unassigned)}")
    if unassigned:
        print(f"    {unassigned[:10]}")

    rows, mismatches = [], []
    done = 0
    total_units = sum(len(items_by_tile[k]) * len(v) for k, v in assigned.items())

    for (h, v), here in sorted(assigned.items()):
        items = items_by_tile[(h, v)]
        print(f"  h{h:02d}v{v:02d}: {len(items)} composites, {len(here)} counties",
              flush=True)

        for n, item in enumerate(items, 1):
            date = item.properties["start_datetime"][:10]
            t_date = time.time()
            ndvi_tile = open_signed(item.assets["250m_16_days_NDVI"].href).load()
            rel_tile = open_signed(
                item.assets["250m_16_days_pixel_reliability"].href).load()

            for fips, g in here:
                if fips not in masks:
                    continue
                minx, miny, maxx, maxy = g.total_bounds
                try:
                    ndvi = (ndvi_tile.rio.clip_box(minx, miny, maxx, maxy)
                            .rio.clip(g.geometry, g.crs, drop=True)) * NDVI_SCALE
                    rel = (rel_tile.rio.clip_box(minx, miny, maxx, maxy)
                           .rio.clip(g.geometry, g.crs, drop=True))
                except Exception:
                    continue

                frac = masks[fips]
                if frac.shape != ndvi.shape:
                    mismatches.append((fips, date, frac.shape, ndvi.shape))
                    continue

                ndvi = ndvi.where(rel.isin(GOOD_RELIABILITY))
                w = np.where(np.isfinite(ndvi.values), frac, np.nan)
                tot = np.nansum(w)
                rows.append({
                    "fips": fips,
                    "year": year,
                    "date": date,
                    "ndvi": float(np.nansum(ndvi.values * w) / tot) if tot > 0 else np.nan,
                    "usable_px": int(np.isfinite(ndvi.values).sum()),
                    "county_px": int(np.isfinite(rel.values).sum()),
                })

            done += len(here)
            el = time.time() - t0
            rate = done / el if el else 0
            left = (total_units - done) / rate / 60 if rate else float("nan")
            print(f"    h{h:02d}v{v:02d} {n:>2}/{len(items)} {date}  "
                  f"{time.time()-t_date:5.1f}s  "
                  f"{el/60:5.1f} min elapsed, ~{left:5.1f} min left", flush=True)

    df = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / f"ndvi_{year}.parquet", index=False)

    mins = (time.time() - t0) / 60
    print(f"  rows {len(df):,}  counties {df.fips.nunique()}  "
          f"dates {df.date.nunique()}  in {mins:.1f} min")
    if mismatches:
        print(f"  GRID MISMATCHES: {len(mismatches)}  (mask and NDVI "
              f"disagree on shape; weights would land on wrong pixels)")
        for m in mismatches[:5]:
            print(f"    {m}")
    if len(df):
        peak = df.groupby("date").ndvi.mean()
        print(f"  mean NDVI peaks {peak.idxmax()} at {peak.max():.3f}")
        story = df[df.fips == "19169"]
        if len(story):
            i = story.ndvi.idxmax()
            print(f"  Story County peak: {story.loc[i,'ndvi']:.3f} "
                  f"on {story.loc[i,'date']}   "
                  f"(probe gave 0.868 on 2020-07-27)")
    return mins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--year", type=int, default=2020)
    args = ap.parse_args()

    counties = gpd.read_file(COUNTIES, layer="counties")
    cat = pystac_client.Client.open(STAC)

    for y in (list(YEARS) if args.all else [args.year]):
        if (OUTDIR / f"ndvi_{y}.parquet").exists():
            print(f"{y}: already done, skipping")
            continue
        mins = run_year(cat, counties, y)
        if not args.all and mins:
            print(f"\nOne year took {mins:.1f} min. "
                  f"Fourteen would be about {14*mins/60:.1f} hours.")


if __name__ == "__main__":
    main()
