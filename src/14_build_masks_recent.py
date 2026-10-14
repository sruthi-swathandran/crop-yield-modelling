"""
Corn masks for 2022 to 2024, from the Source Cooperative copy of the
Cropland Data Layer.

    conda activate cropyield-cdl
    python src/14_build_masks_recent.py --verify
    python src/14_build_masks_recent.py

Writes data/masks/corn_2022.npz .. corn_2024.npz, the same format as 05,
so 06 onward needs no changes.

WHY A SECOND SOURCE
-------------------
The Planetary Computer's copy stops at 2021. 14a_probe_recent.py
rebuilt Story County's 2021 mask from Source Cooperative and it matched
05's to the last pixel. One county is one county, so --verify repeats
that for 24 counties spread across all eight states before anything is
built. Run it first.

WHAT IS REUSED
--------------
The MODIS side comes straight from 05: one NDVI layer per tile held in
memory, clip_box then clip for the county's template grid. Only the
reading of the corn map changes. The window is cut from the national
30 m grid (EPSG:5070) by coordinates, which fetches a few hundred
kilobytes per county rather than a national file.

2024 is published at 10 m. The 30 m layer used here is NASS's own
resampling of it, included in the store for continuity, so every mask
in the project comes from 30 m data. The 10 m layer is for the
Sentinel-2 extension.

WHY NOT 2025
------------
The Planetary Computer's MOD13Q1 stops at 26 June 2025, so there is no
2025 season to extract. Getting it would mean NASA's own archive, which
needs an Earthdata account. 2025 would also be the most drifted Terra
year, so it is left out rather than chased.
"""

import argparse
import importlib
import os
import time
from pathlib import Path

# GDAL retries a failed HTTP range request itself before giving up.
# The first --verify run died on a truncated response (61 KB of a 354 KB
# block) while loading a MODIS tile.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

import warnings

# Clipping the integer corn map marks pixels outside the county as NaN,
# and casting NaN back to an integer warns once per county. Those pixels
# count as not-corn either way; --verify showed the masks match 05's
# exactly with the warning present. Silenced so real problems stand out.
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

import geopandas as gpd
import icechunk
import numpy as np
import pystac_client
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import shape

from runlog import tee_to

m05 = importlib.import_module("05_build_corn_masks")

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
OUTDIR = ROOT / "data" / "masks"
YEARS = [2022, 2023, 2024]
PAD = 60          # metres of margin around the county window
N_VERIFY = 24


def with_retries(fn, *args, tries=4, label=""):
    """Run fn, retrying on any error with a growing pause. 05 retried
    opening a file but not reading it, and a read is where a truncated
    response shows up."""
    for attempt in range(1, tries + 1):
        try:
            return fn(*args)
        except Exception as e:                       # noqa: BLE001
            if attempt == tries:
                raise
            wait = 5 * attempt
            print(f"    {label} read failed ({type(e).__name__}), "
                  f"retry {attempt}/{tries - 1} in {wait} s", flush=True)
            time.sleep(wait)


def load_tile_grids(cat, counties, year):
    """05's tile loader, with the download inside the retry. Signing
    happens inside the retried call, so a retry also gets a fresh token."""
    items = [i for i in cat.search(
        collections=["modis-13Q1-061"], bbox=list(counties.total_bounds),
        datetime=f"{year}-07-01/{year}-07-20").items() if i.id.startswith("MOD")]
    grids = {}
    for it in items:
        key = (it.properties["modis:horizontal-tile"],
               it.properties["modis:vertical-tile"])
        if key in grids:
            continue
        print(f"    loading tile h{key[0]:02d}v{key[1]:02d} ...", flush=True)
        href = it.assets["250m_16_days_NDVI"].href
        grids[key] = with_retries(lambda h: m05.open_signed(h).load(), href,
                                  label=f"tile h{key[0]:02d}v{key[1]:02d}")
    return grids


def open_cdl():
    storage = icechunk.s3_storage(
        bucket="chill", prefix="usda-cropland-data-layer/v0.1.0.icechunk",
        endpoint_url="https://data.source.coop", region="us-east-1",
        anonymous=True, force_path_style=True,
    )
    session = icechunk.Repository.open(storage).readonly_session("main")
    return xr.open_zarr(session.store, group="30m", chunks=None)


def corn_fraction(ds, geom, year, template, pad=PAD):
    g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:5070")
    minx, miny, maxx, maxy = g.total_bounds
    cdl = (ds.crop_type.sel(year=year,
                            x=slice(minx - pad, maxx + pad),
                            y=slice(maxy + pad, miny - pad))   # y runs north to south
           .load().rio.write_crs("EPSG:5070"))
    cdl = cdl.rio.clip(g.geometry, g.crs, drop=True)
    corn = (cdl == m05.CORN).astype("float32").rio.write_nodata(np.nan)
    frac = corn.rio.reproject_match(template, resampling=Resampling.average).fillna(0)
    return frac.values.astype("float32")


def verify(cat, ds, counties):
    """Rebuild 2021 for counties across every state and compare with
    the masks 05 made from the Planetary Computer."""
    print(f"--- verify: 2021 rebuilt for {N_VERIFY} counties ---")
    old = np.load(OUTDIR / "corn_2021.npz")
    # groupby().sample keeps every column; groupby().apply drops the
    # grouping column in pandas 3, which broke the first version.
    sample = (counties[counties.fips.isin(old.files)]
              .groupby("state_alpha").sample(n=N_VERIFY // 8, random_state=0))
    grids = load_tile_grids(cat, counties, 2021)
    worst = 0.0
    for row in sample.itertuples():
        template = m05.template_for(grids, row.geometry)
        new = with_retries(corn_fraction, ds, row.geometry, 2021, template,
                           label=row.fips)
        ref = old[row.fips]
        if new.shape != ref.shape:
            print(f"  {row.fips} {row.county_name:<16} SHAPE {new.shape} vs {ref.shape}")
            worst = np.inf
            continue
        d = float(np.abs(new - ref).max())
        worst = max(worst, d)
        print(f"  {row.fips} {row.state_alpha} {row.county_name[:16]:<16} "
              f"mean {new.mean():.4f} vs {ref.mean():.4f}   largest difference {d:.5f}")
    ok = worst < 0.01
    print(f"\n  largest difference across all: {worst:.5f}   "
          f"verdict: {'MATCH' if ok else 'DIFFERENT, do not build'}")
    return ok


def pc_tile_count(cat, counties, year=2021):
    """How many Planetary Computer CDL tiles each county touched in 05,
    using the same footprint matching 05 used."""
    items = [it for it in cat.search(collections=["usda-cdl"],
                                     bbox=list(counties.total_bounds),
                                     datetime=f"{year}-01-01/{year}-12-31").items()
             if "cropland" in it.assets]
    fp = gpd.GeoDataFrame(geometry=[shape(it.geometry) for it in items],
                          crs="EPSG:4326")
    return {r.fips: int(fp.intersects(r.geometry).sum()) for r in counties.itertuples()}


def verify_all(cat, ds, counties):
    """Every county, 2021, against 05. Prints only the ones that differ,
    with what is needed to tell a seam artefact from a real disagreement."""
    print("--- verify, every county, 2021 ---")
    old = np.load(OUTDIR / "corn_2021.npz")
    tiles = pc_tile_count(cat, counties)
    grids = load_tile_grids(cat, counties, 2021)
    rows, t0 = [], time.time()
    todo = counties[counties.fips.isin(old.files)]
    for n, row in enumerate(todo.itertuples(), 1):
        template = m05.template_for(grids, row.geometry)
        new = with_retries(corn_fraction, ds, row.geometry, 2021, template,
                           label=row.fips)
        ref = old[row.fips]
        if new.shape != ref.shape:
            rows.append((row.fips, row.state_alpha, row.county_name, tiles[row.fips],
                         -1, np.nan, np.nan, "shape differs"))
            continue
        d = np.abs(new - ref)
        bad = d > 0.01
        if bad.any():
            r, c = np.nonzero(bad)
            share = float(d.sum() / max(ref.sum(), 1e-9))
            pattern = (f"{len(np.unique(r))} rows x {len(np.unique(c))} cols")
            rows.append((row.fips, row.state_alpha, row.county_name, tiles[row.fips],
                         int(bad.sum()), float(d.max()), share, pattern))
        if n % 50 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"    {n:>3}/{len(todo)}  differing so far {len(rows)}   "
                  f"{el/60:4.1f} min, ~{(len(todo) - n) * el / n / 60:4.1f} min left",
                  flush=True)

    import pandas as pd
    res = pd.DataFrame(rows, columns=["fips", "state", "county", "pc_tiles",
                                      "pixels", "max_diff", "weight_share", "pattern"])
    multi = sum(1 for f in todo.fips if tiles[f] > 1)
    print(f"\n  counties checked {len(todo)}, differing {len(res)}")
    print(f"  counties spanning more than one PC tile: {multi}")
    if len(res):
        print(f"  of the differing, spanning more than one PC tile: "
              f"{(res.pc_tiles > 1).sum()}")
        print(f"  largest share of a county's corn weight that changes: "
              f"{res.weight_share.max():.5f}")
        print(res.sort_values("weight_share", ascending=False)
              .head(25).to_string(index=False))
    res.to_csv(ROOT / "results" / "mask_source_check_2021.csv", index=False)


def diagnose(cat, ds, counties):
    """For the counties --verify-all flagged: are the differences at the
    edge of the corn-map window?

    The hypothesis. A 250 m MODIS pixel is averaged from the 30 m corn
    pixels under it. If the window read from the corn map stops partway
    across a MODIS pixel, the average covers only the part inside the
    window, and a sliver that happens to be all corn reads as 1.0. 05's
    window was the county's bounding box cut from each Planetary Computer
    tile; 14's adds 60 m. Different windows, different slivers, and a
    straight window edge in Albers runs diagonally across the MODIS
    grid, which would explain "25 rows x 25 cols" for 25 pixels.

    If that is right, the differing pixels should change again when the
    window is widened to 500 m, which covers every MODIS pixel touching
    the county completely. And what matters for the NDVI is only the
    share of weight inside the county, since 06 weights only pixels
    whose NDVI survives the clip to the county outline.
    """
    import pandas as pd
    flagged = pd.read_csv(ROOT / "results" / "mask_source_check_2021.csv",
                          dtype={"fips": str})
    print(f"--- diagnose {len(flagged)} flagged counties ---")
    old = np.load(OUTDIR / "corn_2021.npz")
    grids = load_tile_grids(cat, counties, 2021)
    rows = []
    for n, f in enumerate(flagged.fips, 1):
        geom = counties.loc[counties.fips == f, "geometry"].iloc[0]
        template = m05.template_for(grids, geom)
        inside = np.isfinite(template.values)
        ref = old[f]
        n60 = with_retries(corn_fraction, ds, geom, 2021, template, 60, label=f)
        n500 = with_retries(corn_fraction, ds, geom, 2021, template, 500, label=f)
        bad = np.abs(n60 - ref) > 0.01
        moves = np.abs(n500 - n60) > 0.01
        rows.append({
            "fips": f,
            "differing": int(bad.sum()),
            "inside_county": int((bad & inside).sum()),
            "move_with_wider_window": int((bad & moves).sum()),
            "weight_share_inside": float(np.abs(n60 - ref)[inside].sum()
                                         / max(ref[inside].sum(), 1e-9)),
            "weight_share_inside_500m": float(np.abs(n500 - ref)[inside].sum()
                                              / max(ref[inside].sum(), 1e-9)),
        })
        if n % 10 == 0 or n == len(flagged):
            print(f"    {n}/{len(flagged)}", flush=True)
    d = pd.DataFrame(rows)
    tot = d.differing.sum()
    print(f"\n  differing pixels in total: {tot}")
    print(f"  inside the county outline: {d.inside_county.sum()} "
          f"({100 * d.inside_county.sum() / max(tot, 1):.0f}%)")
    print(f"  that change again when the window widens to 500 m: "
          f"{d.move_with_wider_window.sum()} "
          f"({100 * d.move_with_wider_window.sum() / max(tot, 1):.0f}%)")
    print(f"  largest share of in-county corn weight that differs from 05: "
          f"{d.weight_share_inside.max():.5f} (60 m window), "
          f"{d.weight_share_inside_500m.max():.5f} (500 m window)")
    print(d.sort_values("weight_share_inside", ascending=False).head(15)
          .round(5).to_string(index=False))
    d.to_csv(ROOT / "results" / "mask_source_diagnosis_2021.csv", index=False)


def run_year(cat, ds, counties, year):
    out = OUTDIR / f"corn_{year}.npz"
    if out.exists():
        print(f"{year}: already built, skipping")
        return
    t0 = time.time()
    print(f"\n=== {year} ===", flush=True)
    grids = load_tile_grids(cat, counties, year)
    print(f"  tiles loaded: {sorted(grids)}")
    masks, skipped = {}, []
    n_all = len(counties)
    for n, row in enumerate(counties.itertuples(), 1):
        template = m05.template_for(grids, row.geometry)
        if template is None:
            skipped.append(row.fips)
            continue
        try:
            masks[row.fips] = with_retries(corn_fraction, ds, row.geometry,
                                           year, template, label=row.fips)
        except Exception as e:                       # noqa: BLE001
            print(f"    {row.fips} failed: {e}")
            skipped.append(row.fips)
            continue
        if n <= 5 or n % 10 == 0 or n == n_all:
            el = time.time() - t0
            print(f"    {n:>3}/{n_all}  {row.county_name[:18]:<18} "
                  f"{el/60:5.1f} min elapsed, ~{(n_all - n) * el / n / 60:5.1f} min left",
                  flush=True)
    np.savez_compressed(out, **masks)
    frs = [float(v.mean()) for v in masks.values()]
    print(f"  masks written: {len(masks):,}  skipped: {len(skipped)}  "
          f"in {(time.time() - t0)/60:.1f} min")
    print(f"  corn fraction across counties: mean {np.mean(frs):.3f}")
    if skipped:
        print(f"  skipped: {skipped[:20]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="compare a 2021 rebuild with 05's masks, build nothing")
    ap.add_argument("--verify-all", action="store_true",
                    help="the same for every county, with seam diagnostics")
    ap.add_argument("--diagnose", action="store_true",
                    help="test the window-edge explanation on flagged counties")
    args = ap.parse_args()
    tee_to("masks_recent")
    counties = gpd.read_file(COUNTIES, layer="counties")
    cat = pystac_client.Client.open(m05.STAC)
    ds = open_cdl()
    if args.diagnose:
        diagnose(cat, ds, counties)
        return
    if args.verify_all:
        verify_all(cat, ds, counties)
        return
    if args.verify:
        verify(cat, ds, counties)
        return
    for y in YEARS:
        run_year(cat, ds, counties, y)


if __name__ == "__main__":
    main()
