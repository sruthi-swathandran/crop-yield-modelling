"""
Save each Iowa county's 30 m corn map, 2017 to 2024, as a small GeoTIFF
for the Sentinel-2 extraction to read from disk.

    conda activate cropyield-cdl
    python src/17c_export_corn30.py

Writes data/s2/corn30/YEAR/FIPS.tif, uint8, 1 = corn, EPSG:5070.
About 35 minutes. Finished files are skipped on a rerun.

WHY A SEPARATE STEP
-------------------
The extraction runs for about two days in the main environment, which
cannot read the Source Cooperative store (that needs Python 3.12 and
icechunk 2). Exporting the masks first keeps the long run in one
environment and off a second network source.

All eight years come from Source Cooperative, which 14 verified against
the Planetary Computer's 2021 masks. One source for every year means
no year's mask differs from another's in how it was made. 2024 uses the
30 m layer, NASS's own resampling of its 10 m map; the 10 m layer is
for a separate test.

The window is the county's bounding box plus 500 m and is not clipped
to the outline. The extraction clips on the Sentinel-2 grid instead, so
pixels on the boundary get their corn fraction from real neighbours.
"""

import importlib
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401

from runlog import tee_to

m14 = importlib.import_module("14_build_masks_recent")

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
OUT = ROOT / "data" / "s2" / "corn30"
YEARS = range(2017, 2025)
MARGIN = 500


def main() -> None:
    tee_to("corn30")
    counties = gpd.read_file(COUNTIES, layer="counties")
    ia = counties[counties.state_alpha == "IA"].sort_values("fips")
    ds = m14.open_cdl()
    total, done, t0 = len(ia) * len(YEARS), 0, time.time()
    for y in YEARS:
        (OUT / str(y)).mkdir(parents=True, exist_ok=True)
        for row in ia.itertuples():
            done += 1
            path = OUT / str(y) / f"{row.fips}.tif"
            if path.exists():
                continue
            g = gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_crs("EPSG:5070")
            minx, miny, maxx, maxy = g.total_bounds

            def read():
                return (ds.crop_type.sel(year=y, x=slice(minx - MARGIN, maxx + MARGIN),
                                         y=slice(maxy + MARGIN, miny - MARGIN))
                        .load())
            cdl = m14.with_retries(read, label=f"{y} {row.fips}")
            corn = (cdl == 1).astype("uint8").rio.write_crs("EPSG:5070")
            corn.rio.to_raster(path, compress="deflate")
        el = time.time() - t0
        print(f"  {y}: {len(ia)} counties   {el/60:5.1f} min, "
              f"~{(total - done) * el / max(done, 1) / 60:5.1f} min left", flush=True)

    # Check: every file present, and corn shares plausible for Iowa.
    shares = []
    for y in YEARS:
        for row in ia.itertuples():
            a = rioxarray.open_rasterio(OUT / str(y) / f"{row.fips}.tif").values
            shares.append(a.mean())
    print(f"\n  files: {len(shares)} of {total}")
    print(f"  corn share of county windows: median {np.median(shares):.2f}, "
          f"range {np.min(shares):.2f} to {np.max(shares):.2f}")


if __name__ == "__main__":
    main()
