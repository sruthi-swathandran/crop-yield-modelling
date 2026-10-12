"""
Fetch county boundaries from the US Census cartographic boundary files.

    python src/02_fetch_boundaries.py

Writes data/counties.gpkg, one polygon per county, for the states the
yield data covers.

WHY THE CARTOGRAPHIC FILE AND NOT TIGER/LINE
--------------------------------------------
TIGER/Line gives the legal boundaries at full precision, around 800 MB
for US counties, with coastlines following every inlet.

These polygons exist to clip a 250 metre MODIS raster. At that pixel
size, detail finer than a few hundred metres cannot change which pixels
fall inside a county. The 1:500,000 cartographic file is about 10 MB
and is accurate well beyond what the analysis can use.

Picking the precision the question needs rather than the highest
available is the same decision as using MODIS over Sentinel-2.

WHY GEOPACKAGE AND NOT SHAPEFILE
--------------------------------
Shapefile truncates field names at ten characters, has no real null,
stores the CRS in a separate file that gets lost, and is four files
pretending to be one. GeoPackage is a single SQLite file with none of
those problems.

The Olympic project inherited a shapefile and spent an afternoon on its
defects. Where there is a choice, this is the choice.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
YIELDS = ROOT / "data" / "nass_corn_yields.csv"
OUT = ROOT / "data" / "counties.gpkg"

URL = (
    "https://www2.census.gov/geo/tiger/GENZ2023/shp/"
    "cb_2023_us_county_500k.zip"
)

STATE_FIPS = {
    "19": "IA", "17": "IL", "18": "IN", "31": "NE",
    "27": "MN", "39": "OH", "29": "MO", "46": "SD",
}


def main() -> None:
    print(f"reading {URL}")
    gdf = gpd.read_file(URL)
    print(f"  national counties: {len(gdf):,}")
    print(f"  CRS as published : {gdf.crs}")
    print(f"  columns          : {list(gdf.columns)}")

    gdf = gdf[gdf.STATEFP.isin(STATE_FIPS)].copy()
    gdf["fips"] = gdf.GEOID
    gdf["state_alpha"] = gdf.STATEFP.map(STATE_FIPS)

    # EPSG:4326, matching the MODIS products and avoiding a
    # reprojection inside the extraction loop. Areas computed from
    # this CRS would be wrong, which is why none are.
    gdf = gdf.to_crs("EPSG:4326")

    keep = ["fips", "state_alpha", "NAME", "ALAND", "AWATER", "geometry"]
    gdf = gdf[keep].rename(columns={"NAME": "county_name"})

    print(f"\n  counties in the eight states: {len(gdf):,}")
    print(gdf.groupby("state_alpha").size().to_string())

    # --- the check that matters -----------------------------------
    # Every county with a yield must have a polygon, or it silently
    # drops out of the whole pipeline at the extraction step.
    yields = pd.read_csv(YIELDS, dtype={"fips": str})
    have = set(gdf.fips)
    want = set(yields.fips)

    missing = sorted(want - have)
    print(f"\n  counties with a yield but no geometry: {len(missing)}")
    if missing:
        print(yields[yields.fips.isin(missing)]
              .groupby(["state_alpha", "county_name"]).size()
              .to_string())

    extra = sorted(have - want)
    print(f"  counties with geometry but no yield : {len(extra)}")

    # Geometry validity. An invalid polygon breaks clipping in ways
    # that produce wrong answers rather than errors.
    invalid = gdf[~gdf.geometry.is_valid]
    print(f"  invalid geometries: {len(invalid)}")
    if len(invalid):
        gdf.loc[~gdf.geometry.is_valid, "geometry"] = gdf.loc[
            ~gdf.geometry.is_valid, "geometry"
        ].buffer(0)
        print("  repaired with buffer(0)")

    print(f"\n  bounding box: {[round(v, 2) for v in gdf.total_bounds]}")

    OUT.parent.mkdir(exist_ok=True)
    gdf.to_file(OUT, layer="counties", driver="GPKG")
    print(f"\nwritten: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
