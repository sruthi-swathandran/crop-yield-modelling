"""
Can Sentinel-2 over Iowa be read fast enough? Three measurements that
decide the scope before anything is built.

    python src/17b_probe_s2_cost.py

Runs in the cropyield environment, about ten minutes.

17a found one county read from one scene takes 7.1 s, which puts
eight years at 20 m near a day and a half. Most of that time is
probably round trips between here and the storage account in West
Europe rather than bandwidth, and round trips can be overlapped.

1. DUPLICATES
-------------
2022 and 2023 list about 5,500 scenes against about 3,300 in other
years, under two processing baselines each. ESA reprocessed them and
the catalogue kept both. Keep one item per tile and sensing time, the
newest baseline, and count what that removes.

2. DATES NEEDED
---------------
For every Iowa county and every 16-day MODIS window, how many distinct
sensing dates have a scene under 50% cloud covering it. If most windows
have at least one, a single good date per county-window is enough, and
that sets the volume.

3. READ SPEED
-------------
The same eight counties from one scene read three ways:
  sequential, default GDAL settings
  sequential, GDAL tuned for cloud files (no directory listing, merged
    range requests, HTTP/2 multiplexing, a block cache)
  eight counties at once on threads, tuned settings
The fastest becomes the extraction's method, and its rate the basis of
the time estimate.
"""

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401
from shapely.geometry import shape

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
FIRST_DOY, N_COMP = 81, 14
YEARS = range(2017, 2025)

TUNED = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "100000000",
}


def dedupe(items):
    """One item per MGRS tile and sensing time, the newest baseline."""
    best = {}
    for it in items:
        key = (it.properties.get("s2:mgrs_tile"), it.properties["datetime"][:19])
        b = str(it.properties.get("s2:processing_baseline", "0"))
        if key not in best or b > str(best[key].properties.get("s2:processing_baseline", "0")):
            best[key] = it
    return list(best.values())


def county_read(it, geom):
    out = {}
    for asset, ov, masked in (("B04", 0, True), ("B08", 0, True), ("SCL", None, False)):
        kw = {"overview_level": ov} if ov is not None else {}
        a = rioxarray.open_rasterio(pc.sign(it.assets[asset].href), masked=masked, **kw).squeeze()
        g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(a.rio.crs)
        out[asset] = a.rio.clip_box(*g.total_bounds).load()
    return sum(v.size for v in out.values())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-only", action="store_true",
                    help="skip the per-year metadata tables, measure read speed only")
    speed_only = ap.parse_args().speed_only
    tee_to("probe_s2_cost")
    counties = gpd.read_file(COUNTIES, layer="counties")
    ia = counties[counties.state_alpha == "IA"].reset_index(drop=True)
    cat = pystac_client.Client.open(STAC)

    # --- 1 and 2: metadata only ---------------------------------------
    print("--- 1. duplicates, and 2. dates per county-window (scenes under 50% cloud) ---")
    print("  year  items  after dedupe  removed   windows with >=1 date   median dates")
    keep_items = {}
    for y in ([2020] if speed_only else YEARS):
        its = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(ia.total_bounds),
                              datetime=f"{y}-03-22/{y}-10-31").items())
        d = dedupe(its)
        keep_items[y] = d
        clear = [i for i in d if i.properties.get("eo:cloud_cover", 100) < 50]
        fp = gpd.GeoDataFrame(
            {"date": [pd.Timestamp(i.properties["datetime"][:10]) for i in clear]},
            geometry=[shape(i.geometry) for i in clear], crs="EPSG:4326")
        fp["window"] = (fp.date.dt.dayofyear - FIRST_DOY) // 16 + 1
        fp = fp[fp.window.between(1, N_COMP)]
        hits = gpd.sjoin(ia[["fips", "geometry"]], fp, predicate="intersects")
        per = hits.groupby(["fips", "window"]).date.nunique()
        full = per.reindex(pd.MultiIndex.from_product([ia.fips, range(1, N_COMP + 1)]),
                           fill_value=0)
        print(f"  {y}  {len(its):>5}  {len(d):>12}  {len(its) - len(d):>7}   "
              f"{100 * (full >= 1).mean():>19.0f}%   {full.median():>12.0f}", flush=True)

    # --- 3. read speed ------------------------------------------------
    # The first version required a scene that wholly contained eight
    # counties and found none: Sentinel-2 granules are often cut by the
    # edge of the swath. For timing, partial cover is fine, so take the
    # clear July 2020 scene that overlaps the most Iowa land, and its
    # eight counties with the largest overlap.
    print("\n--- 3. read speed: eight counties from one July 2020 scene ---")
    its = [i for i in keep_items[2020]
           if i.properties.get("eo:cloud_cover", 100) < 5
           and i.properties["datetime"][5:7] == "07"]
    ia_eq = ia.to_crs("EPSG:5070")
    best, chosen = 0, None
    for it in its:
        fp = gpd.GeoSeries([shape(it.geometry)], crs="EPSG:4326").to_crs("EPSG:5070").iloc[0]
        ov = ia_eq.geometry.intersection(fp).area
        if ov.sum() > best:
            best, chosen = ov.sum(), it
            sample = ia.loc[ov.sort_values(ascending=False).index[:8]]
    if chosen is None:
        raise SystemExit("no clear July 2020 scene over Iowa")
    print(f"  scene {chosen.id} (cloud {chosen.properties['eo:cloud_cover']:.1f}%), "
          f"counties {list(sample.fips)}")

    def timed(label, parallel):
        t0 = time.time()
        if parallel:
            with ThreadPoolExecutor(max_workers=8) as ex:
                px = sum(ex.map(lambda g: county_read(chosen, g), sample.geometry))
        else:
            px = sum(county_read(chosen, g) for g in sample.geometry)
        dt = time.time() - t0
        print(f"  {label:<38} {dt:6.1f} s   {dt / len(sample):5.1f} s per county   "
              f"{px / dt / 1e6:5.2f} M pixels/s", flush=True)
        return dt / len(sample)

    timed("sequential, default GDAL", False)
    os.environ.update(TUNED)
    seq = timed("sequential, tuned GDAL", False)
    par = timed("8 counties in parallel, tuned GDAL", True)

    # --- the estimate -------------------------------------------------
    best = min(seq, par)
    reads = 99 * N_COMP * len(YEARS) * 1.5      # one date per window, ~1.5 tiles per county
    print("\n--- estimate, one date per county-window, 20 m ---")
    print(f"  county reads: about {reads:,.0f}")
    print(f"  at {best:.1f} s per county read: about {reads * best / 3600:.0f} hours")
    print(f"  at 40 m (a quarter of the pixels): roughly {reads * best / 3600 / 3:.0f} hours, "
          f"since part of each read is fixed overhead")


if __name__ == "__main__":
    main()
