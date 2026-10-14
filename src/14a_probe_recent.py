"""
Before building anything for 2022 to 2025, check the three things it
depends on.

    conda activate cropyield-cdl
    python src/14a_probe_recent.py

Runs in its own environment, built from environment-cdl.yml. The
Source Cooperative store is Icechunk format 2, which needs icechunk 2.x
and therefore Python 3.12; the main environment stays on 3.11 so the
published results reproduce exactly.

1. A NEW SOURCE FOR THE CORN MAP
--------------------------------
The Planetary Computer's Cropland Data Layer stops at 2021. Source
Cooperative hosts the full 2008 to 2025 record as an Icechunk Zarr
store, 30 m throughout and 10 m from 2024, described as bit-identical
to the official USDA rasters.

"Described as" is not enough to swap a data source halfway through a
project. So Story County's 2021 mask is rebuilt from the new source,
by the same method 05 used, and compared with the mask 05 saved from
the Planetary Computer. If they match, 2022 to 2025 can come from the
new source without changing what a mask means.

2. MODIS AFTER 2021
-------------------
Count the Terra composites in each season from 2022 to 2025 on Story
County's tile. Expect 14, or 15 in a leap year.

3. A FIRST LOOK AT TERRA'S DRIFT
--------------------------------
MOD13Q1 records the sun zenith angle for every pixel. At a fixed place
and date, the sun is lower in the sky earlier in the morning, so if
Terra has been crossing earlier since 2020 the angle should rise. One
composite per year (day 209, late July) over Story County, 2008 to
2025. This is a preview; the proper drift check comes later.
"""

import time
from pathlib import Path

import geopandas as gpd
import icechunk
import numpy as np
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
MASKS = ROOT / "data" / "masks"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

STORY = "19169"
TILE = (11, 4)
CORN = 1


def open_cdl_store():
    storage = icechunk.s3_storage(
        bucket="chill",
        prefix="usda-cropland-data-layer/v0.1.0.icechunk",
        endpoint_url="https://data.source.coop",
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, group="30m", chunks=None)


def open_signed(href):
    return rioxarray.open_rasterio(pc.sign(href), masked=True).squeeze()


def modis_items(cat, year, start="04-01", end="10-31"):
    its = [i for i in cat.search(
        collections=["modis-13Q1-061"],
        query={"modis:horizontal-tile": {"eq": TILE[0]},
               "modis:vertical-tile": {"eq": TILE[1]}},
        datetime=f"{year}-{start}/{year}-{end}",
    ).items() if i.id.startswith("MOD")]
    return sorted(its, key=lambda i: i.properties["start_datetime"])


def main() -> None:
    tee_to("probe_recent")
    counties = gpd.read_file(COUNTIES, layer="counties")
    story = counties[counties.fips == STORY].geometry.iloc[0]
    cat = pystac_client.Client.open(STAC)

    # --- 1. the new corn map source --------------------------------
    print("--- 1. Cropland Data Layer from Source Cooperative ---")
    t0 = time.time()
    ds = open_cdl_store()
    years = ds.year.values.tolist()
    print(f"  opened in {time.time() - t0:.1f} s")
    print(f"  years: {years[0]} to {years[-1]} ({len(years)})")
    print(f"  grid: {ds.sizes}   crs attr: {ds.crop_type.attrs.get('grid_mapping', 'n/a')}")
    print(f"  class 1 is: {ds.crop_type.attrs['class_names'].get('1')}")

    g5070 = gpd.GeoSeries([story], crs="EPSG:4326").to_crs("EPSG:5070")
    minx, miny, maxx, maxy = g5070.total_bounds
    t0 = time.time()
    cdl = (ds.crop_type.sel(year=2021, x=slice(minx - 60, maxx + 60),
                            y=slice(maxy + 60, miny - 60))
           .load().rio.write_crs("EPSG:5070"))
    print(f"  Story County 2021 window {cdl.shape} read in {time.time() - t0:.1f} s")
    cdl = cdl.rio.clip(g5070.geometry, g5070.crs, drop=True)

    # The MODIS template exactly as 05 built it: one composite of the
    # county's tile, clip_box then clip.
    item = modis_items(cat, 2021, "07-01", "07-20")[0]
    tile = open_signed(item.assets["250m_16_days_NDVI"].href)
    gs = gpd.GeoSeries([story], crs="EPSG:4326").to_crs(tile.rio.crs)
    template = (tile.rio.clip_box(*gs.total_bounds)
                .rio.clip(gs.geometry, gs.crs, drop=True).load())

    corn = (cdl == CORN).astype("float32").rio.write_nodata(np.nan)
    frac = corn.rio.reproject_match(template, resampling=Resampling.average).fillna(0)
    new = frac.values.astype("float32")
    old = np.load(MASKS / "corn_2021.npz")[STORY]

    print(f"\n  Story County 2021 corn mask, new source against 05's:")
    print(f"    shape         {new.shape}  against {old.shape}")
    if new.shape == old.shape:
        d = np.abs(new - old)
        print(f"    mean fraction {new.mean():.4f}  against {old.mean():.4f}")
        print(f"    largest pixel difference {d.max():.4f}, "
              f"pixels differing by more than 0.01: {(d > 0.01).sum()} of {d.size}")
        ok = d.max() < 0.01
        print(f"    verdict: {'MATCH' if ok else 'DIFFERENT, investigate before using'}")
    else:
        print("    verdict: SHAPES DIFFER, investigate before using")

    # --- 2. MODIS after 2021 --------------------------------------
    print("\n--- 2. Terra composites per season, tile "
          f"h{TILE[0]:02d}v{TILE[1]:02d} ---")
    for y in range(2020, 2026):
        its = modis_items(cat, y)
        first = its[0].properties["start_datetime"][:10] if its else "-"
        last = its[-1].properties["start_datetime"][:10] if its else "-"
        print(f"  {y}: {len(its):>2} composites   {first} to {last}")

    # --- 3. sun zenith angle, a preview of the drift ---------------
    print("\n--- 3. Sun zenith angle over Story County, composite starting day 209 ---")
    print("  year   sun zenith (deg)   view zenith (deg)")
    for y in range(2008, 2026):
        its = [i for i in modis_items(cat, y, "07-20", "08-05")
               if i.properties["start_datetime"][5:10] in ("07-27", "07-28")]
        if not its:
            print(f"  {y}   no day-209 composite found")
            continue
        it = its[0]
        vals = []
        for asset in ["250m_16_days_sun_zenith_angle", "250m_16_days_view_zenith_angle"]:
            a = open_signed(it.assets[asset].href)
            sub = a.rio.clip_box(*gs.total_bounds).load() * 0.01
            vals.append(float(np.nanmean(sub.values)))
        print(f"  {y}   {vals[0]:8.2f}            {vals[1]:8.2f}")
    print("  A steady rise from about 2020 would be the earlier overpass.")


if __name__ == "__main__":
    main()
