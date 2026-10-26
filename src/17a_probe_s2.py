"""
What would Sentinel-2 over Iowa, 2017 to 2024, cost to extract? Measured
before anything is built.

The first plan stopped at 2021 because both the corn map and the MODIS
extraction did. Both now reach 2024, and the forest test found no
measurable drift in MODIS after 2021, so all eight seasons are usable.
2024 also has the 10 m corn map.

    python src/17a_probe_s2.py

Runs in the cropyield environment, in a few minutes.

1. HOW MANY SCENES
------------------
Sentinel-2 Level-2A items over Iowa each season (April to October), by
cloud cover, and how many MGRS tiles cover the state. Sentinel-2B joined
in mid 2017, so 2017 should have fewer scenes than later years.

2. ONE COUNTY, ONE SCENE, TIMED
-------------------------------
Story County on a clear July 2020 scene: red (B04) and near infrared
(B08) read at 20 m from their overviews, and the scene classification
layer (SCL) at its native 20 m. Timing that read is what turns the
scene count into hours.

The corn-weighted NDVI from that scene is compared with MODIS for the
same fortnight (06 found Story County at 0.868 on the composite
starting 27 July 2020). They should be close but not equal: a single
clear day against a 16-day best-pixel composite, and 20 m pixels that
are mostly pure corn against 250 m pixels that are mostly mixed. If
Sentinel-2 reads noticeably higher, that is the mixed-pixel dilution
the extension is meant to measure.

3. A TRAP CHECKED IN ADVANCE
----------------------------
From processing baseline 04.00 (January 2022) ESA added an offset of
1000 to Level-2A reflectance values. 2022 to 2024 should carry it and
earlier years should not, unless reprocessed. The probe prints the
baselines it finds per year rather than assuming, because an
uncorrected offset would bias NDVI quietly.
"""

import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401
from rasterio.enums import Resampling

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
STORY = "19169"
CORN = 1
GOOD_SCL = (4, 5)          # vegetation, not vegetated; clouds and shadows excluded


def main() -> None:
    tee_to("probe_s2")
    counties = gpd.read_file(COUNTIES, layer="counties")
    ia = counties[counties.state_alpha == "IA"]
    story = counties[counties.fips == STORY].geometry.iloc[0]
    cat = pystac_client.Client.open(STAC)
    print(f"Iowa counties: {len(ia)}   bbox {np.round(ia.total_bounds, 2)}")

    # --- 1. scene counts ----------------------------------------------
    print("\n--- 1. Sentinel-2 L2A over Iowa, 1 April to 31 October ---")
    print("  year  scenes  <20% cloud  <50% cloud  tiles  baselines")
    for y in range(2017, 2025):
        its = list(cat.search(collections=["sentinel-2-l2a"],
                              bbox=list(ia.total_bounds),
                              datetime=f"{y}-04-01/{y}-10-31").items())
        cc = np.array([i.properties.get("eo:cloud_cover", 100) for i in its])
        tiles = {i.properties.get("s2:mgrs_tile") for i in its}
        base = Counter(i.properties.get("s2:processing_baseline") for i in its)
        print(f"  {y}  {len(its):>6}  {(cc < 20).sum():>10}  {(cc < 50).sum():>10}"
              f"  {len(tiles):>5}  {dict(base)}", flush=True)

    # --- 2. one county, one scene -------------------------------------
    print("\n--- 2. Story County, clearest scene 15 July to 10 August 2020 ---")
    its = list(cat.search(collections=["sentinel-2-l2a"], intersects=story.__geo_interface__,
                          datetime="2020-07-15/2020-08-10").items())
    its.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    it = its[0]
    print(f"  {it.id}   cloud {it.properties['eo:cloud_cover']:.1f}%   "
          f"baseline {it.properties.get('s2:processing_baseline')}")

    def read(asset, overview, masked=True):
        # Masked, so Sentinel-2's no-data value of 0 becomes NaN. Unmasked,
        # the "inside the county" test below would count every pixel.
        href = pc.sign(it.assets[asset].href)
        kw = {"overview_level": overview} if overview is not None else {}
        a = rioxarray.open_rasterio(href, masked=masked, **kw).squeeze()
        g = gpd.GeoSeries([story], crs="EPSG:4326").to_crs(a.rio.crs)
        t0 = time.time()
        sub = a.rio.clip_box(*g.total_bounds).load()
        return sub, time.time() - t0, g

    red, t_red, g = read("B04", 0)       # overview 0 of a 10 m band is 20 m
    nir, t_nir, _ = read("B08", 0)
    scl, t_scl, _ = read("SCL", None, masked=False)   # native 20 m, class codes
    print(f"  read times: B04 {t_red:.1f}s  B08 {t_nir:.1f}s  SCL {t_scl:.1f}s   "
          f"shapes {red.shape} {nir.shape} {scl.shape}")
    print(f"  resolutions: B04 {abs(red.rio.resolution()[0]):.0f} m, "
          f"B08 {abs(nir.rio.resolution()[0]):.0f} m, SCL {abs(scl.rio.resolution()[0]):.0f} m")
    if red.shape != scl.shape:
        scl = scl.rio.reproject_match(red, resampling=Resampling.nearest)

    offset = 1000 if str(it.properties.get("s2:processing_baseline", "0")) >= "04.00" else 0
    r = (red.values.astype("float64") - offset) * 1e-4
    n = (nir.values.astype("float64") - offset) * 1e-4
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (n - r) / (n + r)

    # Corn fraction on this 20 m grid, from the 2020 corn map.
    cdl_items = [c for c in cat.search(collections=["usda-cdl"], intersects=story.__geo_interface__,
                                       datetime="2020-01-01/2020-12-31").items()
                 if "cropland" in c.assets]
    cdl = rioxarray.open_rasterio(pc.sign(cdl_items[0].assets["cropland"].href),
                                  masked=True).squeeze()
    gc = gpd.GeoSeries([story], crs="EPSG:4326").to_crs(cdl.rio.crs)
    minx, miny, maxx, maxy = gc.total_bounds
    cdl = (cdl.rio.clip_box(minx - 500, miny - 500, maxx + 500, maxy + 500)
           .rio.clip(gc.geometry, gc.crs, drop=True).load())
    corn = (cdl == CORN).astype("float32").rio.write_nodata(np.nan)
    frac = corn.rio.reproject_match(red, resampling=Resampling.average).fillna(0).values

    inside = red.rio.clip(g.geometry, g.crs, drop=False).notnull().values
    ok = inside & np.isin(scl.values, GOOD_SCL) & np.isfinite(ndvi)
    w = np.where(ok, frac, 0)
    print(f"  pixels in county {inside.sum():,}, clear {ok.sum():,} "
          f"({100 * ok.sum() / max(inside.sum(), 1):.0f}%)")
    print(f"  share of 20 m pixels at least 90% corn: "
          f"{100 * (frac[inside] >= 0.9).mean():.0f}%  "
          f"(MODIS 250 m pixels at least 90% corn in this county: almost none)")
    # Clouded pixels carry NaN NDVI and zero weight, and NaN times 0 is
    # still NaN, which turned the first run's result into nan.
    num = np.sum(np.where(ok, ndvi * frac, 0.0))
    print(f"  corn-weighted NDVI, Sentinel-2 20 m: {num / np.sum(w):.3f}")
    print("  MODIS composite starting 27 July 2020 (from 06): 0.868")

    # --- 3. what the full job would cost ------------------------------
    per_read = t_red + t_nir + t_scl
    print("\n--- 3. rough cost ---")
    print(f"  one county-scene read: {per_read:.1f} s")
    print("  Multiply by Iowa counties per scene and scenes kept to estimate the")
    print("  full run; the estimate is worked out from these numbers next.")


if __name__ == "__main__":
    main()
