"""
Recompute the 2021 county-windows that used a scene whose offset label
was wrong.

    python src/17e_repair_offset.py

Runs in the cropyield environment, a few minutes. Safe to run while 17
is still extracting later years; it only touches 2021.

17d found a 2021 scene labelled processing baseline 04.00 whose data
has no reflectance offset, so 17 subtracted 1000 that was never there.
17 now decides from the data (detect_offset). This repairs what the
first version already wrote:

  1. find the 2021 scenes labelled 04.00 or later (acquired before the
     baseline came into use in January 2022, so any such label comes
     from reprocessing)
  2. find the county-windows whose chosen date is one of their dates
     and whose county they cover
  3. recompute those with the fixed offset handling, keeping the same
     date. The date was chosen on cloud metadata and the cloud layer,
     neither of which the offset affects, so only NDVI can change.

The original file is kept as s2_ndvi_2021_before_repair.parquet, and
every changed row is printed with its old and new value.
"""

import importlib
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import rioxarray  # noqa: F401
from rasterio.features import geometry_mask

from runlog import tee_to

m17 = importlib.import_module("17_extract_s2")

ROOT = Path(__file__).resolve().parent.parent
YEAR = 2021


def main() -> None:
    tee_to("repair_offset")
    counties = gpd.read_file(ROOT / "data" / "counties.gpkg", layer="counties")
    ia = counties[counties.state_alpha == "IA"].sort_values("fips").reset_index(drop=True)
    cat = pystac_client.Client.open(m17.STAC)

    sc = m17.scenes_for_year(cat, ia.total_bounds, YEAR)
    relabelled = sc[sc.baseline >= "04.00"]
    print(f"{YEAR} scenes under {m17.MAX_CLOUD}% cloud labelled 04.00 or later: "
          f"{len(relabelled)}")
    for s in relabelled.itertuples():
        print(f"  {s.item.id}   {s.date.date()}")

    path = m17.S2DIR / f"s2_ndvi_{YEAR}.parquet"
    df = pd.read_parquet(path)
    affected = []
    for s in relabelled.itertuples():
        d = s.date.date().isoformat()
        hit = ia[ia.intersects(s.geometry)].fips
        affected += list(df[(df.date == d) & df.fips.isin(hit)].index)
    affected = sorted(set(affected))
    print(f"\ncounty-windows that used one of those dates: {len(affected)}")
    if not affected:
        return

    geo = ia.set_index("fips").geometry
    changes = []
    for i in affected:
        row = df.loc[i]
        geom = geo[row.fips]
        corn = rioxarray.open_rasterio(m17.CORN30 / str(YEAR) / f"{row.fips}.tif").squeeze()
        corn = corn.astype("float32").rio.write_nodata(np.nan)
        g5070 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:5070")
        inside30 = ~geometry_mask(g5070.geometry, out_shape=corn.shape,
                                  transform=corn.rio.transform())
        corn_total = float(np.nansum(np.where(inside30, corn.values, 0))) * 2.25
        cand = sc[(sc.date == pd.Timestamp(row.date)) & sc.intersects(geom)]
        v, clear, nsc = m17.county_on_date(cand, geom, corn, corn_total)
        changes.append((row.fips, row.window, row.date, row.ndvi, v, row.clear_share, clear))
        df.loc[i, ["ndvi", "clear_share", "scenes"]] = [v, clear, nsc]

    out = pd.DataFrame(changes, columns=["fips", "window", "date", "ndvi_before",
                                         "ndvi_after", "clear_before", "clear_after"])
    print(out.round(3).to_string(index=False))
    moved = (out.ndvi_after - out.ndvi_before).abs() > 0.001
    print(f"\n  rows whose NDVI changed: {moved.sum()} of {len(out)}")
    print(f"  NDVI above 1 after repair: {(df.ndvi > 1).sum()}")

    shutil.copy(path, path.with_name(f"s2_ndvi_{YEAR}_before_repair.parquet"))
    df.to_parquet(path, index=False)
    out.to_csv(ROOT / "results" / f"s2_offset_repair_{YEAR}.csv", index=False)
    print(f"\nwritten: {path} (original kept alongside)")


if __name__ == "__main__":
    main()
