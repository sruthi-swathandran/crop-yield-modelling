"""
Extract corn-weighted NDVI time series for every county, one year at
a time.

    python src/05_extract_ndvi.py           # one year, for timing
    python src/05_extract_ndvi.py --all     # every year 2008-2021

Writes data/ndvi/ndvi_YYYY.parquet, one row per county-composite.

WHY THE LOOP IS INVERTED
------------------------
04_probe_extraction.py took about two minutes for one county-year.
There are 742 counties and 14 years, which at that rate is a fortnight
of continuous running.

The cost is in the reads rather than the arithmetic, and neighbouring
counties read identical rasters: Story and Boone pull the same MODIS
tile fifteen times each.

So this iterates rasters and fetches counties, not the reverse. Six
MODIS tiles cover the Corn Belt, giving 84 tile-years and roughly
2,500 MODIS reads rather than 311,000.

The CDL cannot be handled the same way. At 30 m a full MODIS tile is
1.6 billion pixels, while a county bounding box is a handful of
3000 x 3000 CDL tiles. It stays per-county, and it is the remaining
cost.

CHECKPOINTING
-------------
Written per year. A restart skips years already on disk. A nine-hour
job that dies at hour eight and starts again is worse than no job.
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
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
OUTDIR = ROOT / "data" / "ndvi"

YEARS = range(2008, 2022)          # CDL coverage, confirmed in 03
SEASON = ("04-01", "10-31")
CORN = 1
NDVI_SCALE = 0.0001
GOOD_RELIABILITY = (0, 1)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def tile_of(item):
    return (item.properties["modis:horizontal-tile"],
            item.properties["modis:vertical-tile"])


def county_tiles(cat, counties, year):
    """Which MODIS tile each county sits in. Counties on a tile
    boundary appear under more than one and are handled by taking the
    mosaic of whatever covers them."""
    out = {}
    for row in counties.itertuples():
        items = list(cat.search(
            collections=["modis-13Q1-061"],
            bbox=list(row.geometry.bounds),
            datetime=f"{year}-07-01/{year}-07-20",
        ).items())
        out[row.fips] = sorted({tile_of(i) for i in items})
    return out


def corn_fraction_for(cat, geom4326, year, template):
    """Fraction of each MODIS pixel that is corn, from the 30 m CDL.

    Binary corn first, then average resampling onto the vegetation
    grid. Averaging a 0/1 layer turns a class map into a fraction,
    which is more honest than thresholding a 250 m pixel as corn or
    not when it routinely contains both.
    """
    items = [
        it for it in cat.search(
            collections=["usda-cdl"],
            bbox=list(geom4326.bounds),
            datetime=f"{year}-01-01/{year}-12-31",
        ).items()
        if "cropland" in it.assets          # see 04: the collection
    ]                                       # also holds frequency layers
    if not items:
        return None

    parts = []
    for it in items:
        arr = rioxarray.open_rasterio(it.assets["cropland"].href, masked=True).squeeze()
        g = gpd.GeoSeries([geom4326], crs="EPSG:4326").to_crs(arr.rio.crs)
        try:
            parts.append(arr.rio.clip(g.geometry, g.crs, drop=True))
        except Exception:
            continue                        # tile does not really intersect
    if not parts:
        return None

    cdl = parts[0] if len(parts) == 1 else merge_arrays(parts)
    corn = (cdl == CORN).astype("float32").rio.write_nodata(np.nan)
    return corn.rio.reproject_match(template, resampling=Resampling.average).fillna(0)


def run_year(cat, counties, year):
    t0 = time.time()
    print(f"\n=== {year} ===")

    tiles = county_tiles(cat, counties, year)
    by_tile = {}
    for fips, ts in tiles.items():
        for t in ts:
            by_tile.setdefault(t, []).append(fips)
    print(f"  MODIS tiles: {sorted(by_tile)}")
    print(f"  counties per tile: "
          f"{ {t: len(v) for t, v in sorted(by_tile.items())} }")

    # Progress accounting. Total units of work is composites times
    # counties, summed across tiles, which is knowable before starting
    # and makes an ETA possible rather than a guess.
    rows = []
    done_units = 0
    total_units = None
    t_start = time.time()

    for tile, fips_list in sorted(by_tile.items()):
        h, v = tile
        items = [
            i for i in cat.search(
                collections=["modis-13Q1-061"],
                query={"modis:horizontal-tile": {"eq": h},
                       "modis:vertical-tile": {"eq": v}},
                datetime=f"{year}-{SEASON[0]}/{year}-{SEASON[1]}",
            ).items()
            if i.id.startswith("MOD")        # Terra only, see 03b
        ]
        items.sort(key=lambda i: i.properties["start_datetime"])
        print(f"  tile h{h:02d}v{v:02d}: {len(items)} composites, "
              f"{len(fips_list)} counties")

        sub = counties[counties.fips.isin(fips_list)]
        fractions = {}

        if total_units is None:
            total_units = sum(
                len(items) * len(v) for v in by_tile.values()
            )

        for n, item in enumerate(items, 1):
            date = item.properties["start_datetime"][:10]
            t_date = time.time()
            ndvi_full = rioxarray.open_rasterio(
                item.assets["250m_16_days_NDVI"].href, masked=True).squeeze()
            rel_full = rioxarray.open_rasterio(
                item.assets["250m_16_days_pixel_reliability"].href,
                masked=True).squeeze()

            for row in sub.itertuples():
                g = gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_crs(ndvi_full.rio.crs)
                try:
                    ndvi = ndvi_full.rio.clip(g.geometry, g.crs, drop=True) * NDVI_SCALE
                    rel = rel_full.rio.clip(g.geometry, g.crs, drop=True)
                except Exception:
                    continue
                ndvi = ndvi.where(rel.isin(GOOD_RELIABILITY))

                if row.fips not in fractions:
                    fractions[row.fips] = corn_fraction_for(cat, row.geometry, year, ndvi)
                frac = fractions[row.fips]
                if frac is None:
                    continue

                w = frac.where(np.isfinite(ndvi))
                tot = float(w.sum())
                rows.append({
                    "fips": row.fips,
                    "year": year,
                    "date": date,
                    "ndvi": float((ndvi * w).sum() / tot) if tot > 0 else np.nan,
                    "corn_fraction": float(w.mean()) if tot > 0 else np.nan,
                    "usable_pct": float(np.isfinite(ndvi).sum()) / max(int(np.isfinite(rel).sum()), 1),
                })

            done_units += len(sub)
            elapsed = time.time() - t_start
            rate = done_units / elapsed if elapsed else 0
            remaining = (total_units - done_units) / rate / 60 if rate else 0
            print(f"    h{h:02d}v{v:02d} {n:>2}/{len(items)}  {date}  "
                  f"{len(sub):>3} counties in {time.time() - t_date:5.1f}s  "
                  f"| {done_units:>5}/{total_units} units, "
                  f"{elapsed/60:5.1f} min elapsed, ~{remaining:5.1f} min left",
                  flush=True)

    df = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / f"ndvi_{year}.parquet", index=False)

    mins = (time.time() - t0) / 60
    print(f"  rows {len(df):,}  counties {df.fips.nunique()}  "
          f"dates {df.date.nunique()}  in {mins:.1f} min")
    if len(df):
        peak = df.groupby("date").ndvi.mean()
        print(f"  mean NDVI peaks on {peak.idxmax()} at {peak.max():.3f}")
    return mins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="run every year rather than one")
    ap.add_argument("--year", type=int, default=2020)
    args = ap.parse_args()

    counties = gpd.read_file(COUNTIES, layer="counties")
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)

    years = list(YEARS) if args.all else [args.year]
    for y in years:
        path = OUTDIR / f"ndvi_{y}.parquet"
        if path.exists():
            print(f"{y}: already done, skipping")
            continue
        mins = run_year(cat, counties, y)
        if not args.all:
            print(f"\nOne year took {mins:.1f} minutes. "
                  f"Fourteen would take about {14 * mins / 60:.1f} hours.")


if __name__ == "__main__":
    main()
