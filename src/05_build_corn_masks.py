"""
Build and cache the corn fraction for every county-year.

    python src/05_build_corn_masks.py --year 2020
    python src/05_build_corn_masks.py --all

Writes data/masks/corn_YYYY.npz, one array per county, on that
county's MODIS grid.

WHY THIS IS ITS OWN SCRIPT
--------------------------
The first attempt built masks inside the NDVI loop. The first composite
date of the first tile took 2,207 seconds against 118 for every date
after it, because the first one paid for 195 counties' crop masks and
the rest reused them.

11 seconds per county, 10,388 county-years: 32 hours of crop masks
alone, redone from scratch on any restart.

A mask depends on the county and the CDL year and nothing else. It is
the same for all fifteen composites, so it belongs outside the loop
over composites, and it is worth keeping on disk.

WHERE THE 11 SECONDS WENT
-------------------------
Two network costs per county, both avoidable.

A STAC search, which is a round trip to ask which CDL tiles cover one
county. Searched once per year for the whole study area instead, then
matched locally against each county's bounds.

And several 30 m COGs opened lazily, so every clip re-reads over HTTP.
Each CDL tile is now read once into memory and clipped from the array.

A KNOWN EDGE ARTEFACT, MEASURED AND LEFT IN PLACE
-------------------------------------------------
The corn map is read with clip_box on the county's bounding box and no
margin. A MODIS pixel lying across that box edge is averaged from only
the 30 m pixels inside it, so a sliver can read as 0 or 1 when the
whole pixel would not. 14_build_masks_recent.py rebuilt 2021 from a
second source with a margin and compared every county: 71 of 742
differed, 97% of the differing pixels lie outside the county outline
(which 06 never weights), widening the margin from 60 m to 500 m changed
none of them, and the largest share of in-county corn weight affected
was 0.031%. That moves a county's NDVI by well under 0.001. The masks
are kept as built so that published results reproduce; masks from 2022
on use the margin.

SIGNING
-------
Tokens are stamped on at read time rather than at search time. The
first run died with HTTP 403 forty-three minutes in, because the
signature applied during the catalogue search had been rotated before
the loop reached that asset.
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401
from rasterio.enums import Resampling
from shapely.geometry import box, shape

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
OUTDIR = ROOT / "data" / "masks"

YEARS = range(2008, 2022)
CORN = 1
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def open_signed(href, retries=3):
    """Sign immediately before opening, and re-sign on failure.

    planetary_computer tokens are short-lived. Signing once during a
    catalogue search and reading an hour later returns HTTP 403, which
    is what killed the first version of the extraction.
    """
    last = None
    for attempt in range(retries):
        try:
            return rioxarray.open_rasterio(pc.sign(href), masked=True).squeeze()
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def load_tile_grids(cat, counties, year):
    """One NDVI layer per MODIS tile, held in memory.

    The first version did a STAC search and an NDVI download per
    county, 742 of each, to discover five distinct answers. Every
    county inside a MODIS tile shares that tile's grid, so the grid
    is a property of the tile and not of the county.
    """
    items = [
        i for i in cat.search(
            collections=["modis-13Q1-061"],
            bbox=list(counties.total_bounds),
            datetime=f"{year}-07-01/{year}-07-20",
        ).items()
        if i.id.startswith("MOD")
    ]
    grids = {}
    for it in items:
        key = (it.properties["modis:horizontal-tile"],
               it.properties["modis:vertical-tile"])
        if key in grids:
            continue
        print(f"    loading tile h{key[0]:02d}v{key[1]:02d} ...", flush=True)
        grids[key] = open_signed(it.assets["250m_16_days_NDVI"].href).load()
    return grids


def template_for(grids, geom):
    """Clip whichever loaded tile covers this county.

    clip_box first, then clip. rio.clip rasterises the polygon mask
    across the entire array, so running it against a 4800 x 4800 tile
    costs 23 million pixels to extract a county covering 27,000 of
    them, and the first version did that up to six times per county
    while working out which tile applied. clip_box is array slicing.
    """
    g4326 = gpd.GeoSeries([geom], crs="EPSG:4326")
    for arr in grids.values():
        g = g4326.to_crs(arr.rio.crs)
        minx, miny, maxx, maxy = g.total_bounds
        ax0, ay0, ax1, ay1 = arr.rio.bounds()
        if maxx < ax0 or minx > ax1 or maxy < ay0 or miny > ay1:
            continue                                # cheap rejection
        try:
            sub = arr.rio.clip_box(minx, miny, maxx, maxy)
            clipped = sub.rio.clip(g.geometry, g.crs, drop=True)
        except Exception:
            continue
        if clipped.size and bool(np.isfinite(clipped).any()):
            return clipped
    return None


def run_year(cat, counties, year):
    t0 = time.time()
    out_path = OUTDIR / f"corn_{year}.npz"
    print(f"\n=== {year} ===")

    # One search for the whole study area rather than one per county.
    bounds = counties.total_bounds
    cdl_items = [
        it for it in cat.search(
            collections=["usda-cdl"],
            bbox=list(bounds),
            datetime=f"{year}-01-01/{year}-12-31",
        ).items()
        if "cropland" in it.assets          # the collection also holds
    ]                                       # frequency layers, see 04
    print(f"  CDL tiles covering the study area: {len(cdl_items)}")

    # Index by the item's real footprint, not its bbox. A CDL tile is
    # a square in Albers, so in lat/lon it is a rotated quadrilateral
    # and its bounding box is materially larger than the tile itself.
    # Matching on the box selected tiles a county does not touch, and
    # every false match opened a 30 m COG for nothing.
    footprints = gpd.GeoDataFrame(
        {"item": cdl_items},
        geometry=[shape(it.geometry) if it.geometry else box(*it.bbox)
                  for it in cdl_items],
        crs="EPSG:4326",
    )

    print("  loading one NDVI layer per MODIS tile for the target grid")
    grids = load_tile_grids(cat, counties, year)
    print(f"  tiles loaded: {sorted(grids)}")

    masks, skipped = {}, []
    for n, row in enumerate(counties.itertuples(), 1):
        hits = footprints[footprints.intersects(row.geometry)]
        if hits.empty:
            skipped.append(row.fips)
            continue

        template = template_for(grids, row.geometry)
        if template is None:
            skipped.append(row.fips)
            continue

        parts = []
        for it in hits.item:
            arr = open_signed(it.assets["cropland"].href)
            g = gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_crs(arr.rio.crs)
            minx, miny, maxx, maxy = g.total_bounds
            try:
                # clip_box on a lazily-opened COG is a windowed read:
                # it fetches only the bytes covering the county rather
                # than the whole 3000 x 3000 tile.
                sub = arr.rio.clip_box(minx, miny, maxx, maxy)
                clipped = sub.rio.clip(g.geometry, g.crs, drop=True).load()
            except Exception:
                continue                    # footprint touches, raster does not
            parts.append(clipped)

        if not parts:
            skipped.append(row.fips)
            continue

        if len(parts) == 1:
            cdl = parts[0]
        else:
            from rioxarray.merge import merge_arrays
            cdl = merge_arrays(parts)

        # Binary corn, then average onto the MODIS grid. Averaging a
        # 0/1 layer gives the fraction of each 250 m pixel that is
        # corn, which beats thresholding a pixel that contains both.
        corn = (cdl == CORN).astype("float32").rio.write_nodata(np.nan)
        frac = corn.rio.reproject_match(
            template, resampling=Resampling.average
        ).fillna(0)

        masks[row.fips] = frac.values.astype("float32")

        # Print the first five individually, then every ten. Ten
        # minutes of silence on the previous run told us nothing about
        # whether it was slow or hung.
        if n <= 5 or n % 10 == 0 or n == len(counties):
            el = time.time() - t0
            rate = n / el
            print(f"    {n:>3}/{len(counties)}  {row.county_name[:18]:<18} "
                  f"cdl tiles {len(hits)}  "
                  f"{el/60:5.1f} min elapsed, "
                  f"~{(len(counties)-n)/rate/60:5.1f} min left", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **masks)

    mins = (time.time() - t0) / 60
    frs = [float(v.mean()) for v in masks.values()]
    print(f"  masks written: {len(masks):,}  skipped: {len(skipped)}")
    print(f"  corn fraction across counties: "
          f"mean {np.mean(frs):.3f}, min {np.min(frs):.3f}, max {np.max(frs):.3f}")
    print(f"  {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)  in {mins:.1f} min")
    if skipped:
        print(f"  skipped fips: {skipped[:20]}")
    return mins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--year", type=int, default=2020)
    args = ap.parse_args()

    counties = gpd.read_file(COUNTIES, layer="counties")
    cat = pystac_client.Client.open(STAC)      # unsigned; signing is per-read

    for y in (list(YEARS) if args.all else [args.year]):
        if (OUTDIR / f"corn_{y}.npz").exists():
            print(f"{y}: already built, skipping")
            continue
        mins = run_year(cat, counties, y)
        if not args.all:
            print(f"\nOne year took {mins:.1f} min. "
                  f"Fourteen would be about {14*mins/60:.1f} hours.")


if __name__ == "__main__":
    main()
