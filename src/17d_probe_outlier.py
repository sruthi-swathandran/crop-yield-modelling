"""
Why did two counties on 25 September 2021 come out with NDVI above 1?

    python src/17d_probe_outlier.py

Runs in the cropyield environment, about a minute.

Davis (19051) and Wapello (19179) read 1.04 and 1.01 on that date,
against MODIS 0.63 and 0.57. Every other Sentinel-2 date sits within
about 0.1 of its MODIS fortnight. NDVI above 1 needs negative red
reflectance, which is what subtracting the 1000 offset from data that
never carried it would produce.

The suspicion: a 2021 scene labelled with processing baseline 04.00 or
later, so 17 removed the offset, whose values do not actually include
it. Data with the offset has no value below 1000 over land; data
without it has many. So for each scene of that date over the two
counties, print the label and the share of red values under 1000.
"""

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401

from runlog import tee_to

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def main() -> None:
    tee_to("probe_outlier")
    counties = gpd.read_file("data/counties.gpkg", layer="counties")
    two = counties[counties.fips.isin(["19051", "19179"])]
    cat = pystac_client.Client.open(STAC)
    its = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(two.total_bounds),
                          datetime="2021-09-25/2021-09-25").items())
    print(f"scenes on 2021-09-25 over the two counties: {len(its)}")
    for it in sorted(its, key=lambda i: i.id):
        a = rioxarray.open_rasterio(pc.sign(it.assets["B04"].href), overview_level=1,
                                    masked=True).squeeze()
        g = two.to_crs(a.rio.crs)
        v = a.rio.clip_box(*g.total_bounds).load().values
        v = v[np.isfinite(v)]
        print(f"  {it.id}")
        print(f"    baseline label {it.properties.get('s2:processing_baseline')}   "
              f"red DN: min {v.min():.0f}, median {np.median(v):.0f}, "
              f"share under 1000 {100 * (v < 1000).mean():.0f}%")
    print("\n  Labelled 04.00 or later with many values under 1000 means the label")
    print("  promises an offset the data does not have.")


if __name__ == "__main__":
    main()
