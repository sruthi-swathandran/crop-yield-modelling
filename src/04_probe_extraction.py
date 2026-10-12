"""
One county, one year, end to end. Validate the extraction before it
runs over 10,000 county-years.

    python src/04_probe_extraction.py

Writes results/probe_story_county_2020.png

Story County, Iowa, 2020. Central Corn Belt, unambiguous corn country,
and NASS reports a yield for it so the result can be checked.

THE TEST
--------
If the extracted curve looks like corn, green-up through May and June,
a peak in late July, senescence through September, then the whole
chain is sound: tile lookup, cloud masking, reprojection, crop
weighting and zonal statistics. If it doesn't, one of those is wrong
and the curve shows which.

Running this before the full loop is the same decision as loading the
smallest taxi file first.

THE REPROJECTION, WHICH IS THE PART MOST LIKELY TO BE WRONG
-----------------------------------------------------------
Three coordinate systems are in play. County boundaries are EPSG:4326.
The CDL is a US Albers equal-area projection at 30 m. MODIS is on a
sinusoidal grid at 231.66 m. No two of them match.

Everything is brought onto the MODIS grid, because that is the grid the
measurement lives on and resampling the measurement would invent
values it never recorded.

The crop mask is converted to a binary corn layer first, then
reprojected with average resampling. Averaging zeros and ones gives
the fraction of each MODIS pixel that is corn, which is more honest
than thresholding a pixel as corn or not-corn when a 250 m pixel
routinely contains both.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
OUT = ROOT / "results" / "probe_story_county_2020.png"

FIPS = "19169"        # Story County, Iowa
YEAR = 2020
SEASON = (f"{YEAR}-04-01", f"{YEAR}-10-31")

CORN = 1              # CDL class code for corn
NDVI_SCALE = 0.0001   # MOD13Q1 stores NDVI as int16

# pixel_reliability: 0 good, 1 marginal, 2 snow or ice, 3 cloud,
# -1 not produced. Keeping 0 and 1 is the usual compromise; keeping
# only 0 throws away most of a cloudy season.
GOOD_RELIABILITY = (0, 1)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def main() -> None:
    counties = gpd.read_file(COUNTIES, layer="counties")
    county = counties[counties.fips == FIPS]
    if county.empty:
        raise SystemExit(f"{FIPS} not found in {COUNTIES}")
    name = county.county_name.iloc[0]
    geom4326 = county.geometry.iloc[0]
    bbox = list(county.total_bounds)
    print(f"county: {name}, Iowa  ({FIPS})")
    print(f"bbox  : {[round(v, 3) for v in bbox]}")

    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)

    # --- vegetation index ----------------------------------------
    items = [
        i for i in cat.search(
            collections=["modis-13Q1-061"], bbox=bbox,
            datetime=f"{SEASON[0]}/{SEASON[1]}",
        ).items()
        if i.id.startswith("MOD")          # Terra only, see 03b
    ]
    items.sort(key=lambda i: i.properties["start_datetime"])
    print(f"Terra composites: {len(items)}")

    dates, means, coverage = [], [], []
    corn_fraction = None

    for n, item in enumerate(items):
        ndvi = rioxarray.open_rasterio(
            item.assets["250m_16_days_NDVI"].href, masked=True
        ).squeeze()
        rel = rioxarray.open_rasterio(
            item.assets["250m_16_days_pixel_reliability"].href, masked=True
        ).squeeze()

        # Clip to the county, in the raster's own CRS.
        geom = gpd.GeoSeries([geom4326], crs="EPSG:4326").to_crs(ndvi.rio.crs)
        ndvi = ndvi.rio.clip(geom.geometry, geom.crs, drop=True)
        rel = rel.rio.clip(geom.geometry, geom.crs, drop=True)

        ndvi = ndvi * NDVI_SCALE

        # Diagnostic. The first run reported 48% usable pixels on
        # almost every date, which cloud cover cannot do: March and
        # July are not equally cloudy. A constant means the loss is
        # geometric, or that ndvi and rel are not aligned and xarray
        # is silently dropping the half that doesn't overlap.
        before = int(np.isfinite(ndvi).sum())
        masked = ndvi.where(rel.isin(GOOD_RELIABILITY))
        after = int(np.isfinite(masked).sum())
        if n == 0:
            print(f"    ndvi shape {ndvi.shape}, rel shape {rel.shape}, "
                  f"same coords: {ndvi.shape == rel.shape}")
        print(f"    array {ndvi.size:>6}  inside county {before:>6}"
              f"  after reliability {after:>6}"
              f"  ({100*after/max(before,1):5.1f}% of county kept)")
        ndvi = masked

        # The crop mask only needs building once: it is the same for
        # every composite in a year.
        if corn_fraction is None:
            corn_fraction = build_corn_fraction(cat, bbox, geom, ndvi)

        weights = corn_fraction.where(np.isfinite(ndvi))
        total = float(weights.sum())
        wmean = float((ndvi * weights).sum() / total) if total > 0 else np.nan

        dates.append(item.properties["start_datetime"][:10])
        means.append(wmean)
        coverage.append(float(np.isfinite(ndvi).sum()) / ndvi.size)
        print(f"  {dates[-1]}  NDVI {wmean:5.3f}   "
              f"usable pixels {100*coverage[-1]:5.1f}%")

    # --- plot -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    ax.plot(dates, means, marker="o", color="#2b7a3d")
    ax.set_ylabel("NDVI, corn-weighted county mean")
    ax.set_title(f"{name} County, Iowa, {YEAR}: MODIS Terra 16-day composites",
                 loc="left")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print(f"\nwritten: {OUT}")

    # --- what the curve should look like --------------------------
    peak_i = int(np.nanargmax(means))
    print(f"\npeak NDVI {means[peak_i]:.3f} on {dates[peak_i]}")
    print(f"first composite {means[0]:.3f}, last {means[-1]:.3f}")
    # In-county only. Averaging across the whole clipped array reports
    # 0.223, which is not wrong so much as answering a different
    # question: the bounding box is twice the county.
    #
    # Sinusoidal projection puts x at R * lambda * cos(phi), so a single
    # meridian shifts east-west as latitude changes. Story County spans
    # 0.346 degrees of latitude, which displaces the same longitude line
    # by about 41 km in x, while the county is only 38.5 km wide. The
    # bounding box therefore has to be roughly 80 km across to hold a
    # 38.5 km county, and the county fills 48% of it.
    last = (dates[-1], means[-1])
    in_county = corn_fraction.where(np.isfinite(ndvi))
    print(f"corn fraction, in-county: mean {float(in_county.mean()):.3f}, "
          f"max {float(in_county.max()):.3f}")
    print(f"corn fraction, whole clipped array: {float(corn_fraction.mean()):.3f} "
          f"(diluted by the 52% outside the county)")


def build_corn_fraction(cat, bbox, geom, ndvi_template):
    """Fraction of each MODIS pixel that is corn, from the 30 m CDL."""
    found = list(cat.search(
        collections=["usda-cdl"], bbox=bbox,
        datetime=f"{YEAR}-01-01/{YEAR}-12-31",
    ).items())

    # usda-cdl is not one product. It carries the annual cropland
    # layer alongside a "cultivated" layer and crop frequency layers,
    # all under the same collection id and with different asset names.
    # Taking every item returned and reaching for assets["cropland"]
    # fails on the ones that have no such asset.
    print(f"  CDL items returned: {len(found)}")
    for it in found:
        print(f"    {it.id:<48} assets={sorted(it.assets)}")

    cdl_items = [it for it in found if "cropland" in it.assets]
    print(f"  items carrying a cropland asset: {len(cdl_items)}")
    if not cdl_items:
        raise SystemExit("no cropland asset found; the collection has changed")

    parts = []
    for it in cdl_items:
        arr = rioxarray.open_rasterio(it.assets["cropland"].href, masked=True).squeeze()
        g = geom.to_crs(arr.rio.crs)
        try:
            arr = arr.rio.clip(g.geometry, g.crs, drop=True)
        except Exception:
            continue      # tile does not actually intersect the county
        parts.append(arr)

    cdl = parts[0] if len(parts) == 1 else merge_parts(parts)

    # Binary corn, then average onto the MODIS grid. Averaging a 0/1
    # layer is what turns a class map into a fraction.
    corn = (cdl == CORN).astype("float32")
    corn = corn.rio.write_nodata(np.nan)
    frac = corn.rio.reproject_match(ndvi_template, resampling=Resampling.average)
    return frac.fillna(0)


def merge_parts(parts):
    from rioxarray.merge import merge_arrays
    return merge_arrays(parts)


if __name__ == "__main__":
    main()
