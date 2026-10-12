"""
Ask the Planetary Computer catalogue what it actually holds for the
Cropland Data Layer and MODIS vegetation indices.

    python src/03_probe_rasters.py

Nothing is downloaded. This reports collection ids, temporal coverage,
asset names, resolutions and projections, so that the extraction script
can be written against what exists rather than against what I remember.

The questions it answers:

  - what are the collections called
  - what years does each cover, which sets the period of the project
  - what are the assets called inside an item
  - what CRS and pixel size does each use, which decides how the 30 m
    crop mask gets onto the 250 m vegetation grid
  - how many MODIS items cover one county in one growing season
"""

import pystac_client
import planetary_computer as pc

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Story County, Iowa. Central Corn Belt, unambiguous corn country.
STORY_COUNTY_BBOX = [-93.70, 41.86, -93.32, 42.21]
YEAR = 2020


def item_date(item) -> str:
    """These collections are interval products: item.datetime is None
    and the date lives in start_datetime. Reaching for .datetime first
    is how the first run of this script crashed."""
    if item.datetime is not None:
        return item.datetime.date().isoformat()
    s = item.properties.get("start_datetime")
    return s[:10] if s else "unknown"


def main() -> None:
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)

    print("=== collections mentioning cdl, cropland or modis ===")
    candidates = []
    for c in cat.get_collections():
        hay = f"{c.id} {c.title or ''}".lower()
        if any(k in hay for k in ("cdl", "cropland", "modis")):
            candidates.append(c)
            extent = c.extent.temporal.intervals[0]
            lo = extent[0].date() if extent[0] else "?"
            hi = extent[1].date() if extent[1] else "open"
            print(f"  {c.id:<30} {lo} to {hi}   {c.title}")

    if not candidates:
        raise SystemExit("nothing matched; the catalogue has changed")

    # --- the crop mask -------------------------------------------
    print("\n=== crop mask: searching for a CDL item over Story County ===")
    cdl_ids = [c.id for c in candidates if "cdl" in c.id.lower() or "cropland" in c.id.lower()]
    print(f"  candidate collections: {cdl_ids}")

    for cid in cdl_ids:
        try:
            search = cat.search(
                collections=[cid],
                bbox=STORY_COUNTY_BBOX,
                datetime=f"{YEAR}-01-01/{YEAR}-12-31",
            )
            items = list(search.items())
        except Exception as e:
            print(f"  {cid}: search failed, {e}")
            continue

        print(f"\n  {cid}: {len(items)} item(s) for {YEAR}")
        if not items:
            continue
        it = items[0]
        print(f"    id        : {it.id}")
        print(f"    date      : {item_date(it)}")
        print(f"    assets    : {sorted(it.assets)}")
        print(f"    properties of interest:")
        for k in ("proj:epsg", "proj:shape", "proj:transform", "gsd"):
            if k in it.properties:
                print(f"      {k}: {it.properties[k]}")
        cropland = it.assets["cropland"]
        print(f"    cropland proj:epsg   : "
              f"{cropland.extra_fields.get('proj:epsg', 'absent')}")
        print(f"    years available      : ", end="")
        allyears = sorted({
            item_date(i)[:4]
            for i in cat.search(collections=[cid], bbox=STORY_COUNTY_BBOX).items()
        })
        print(allyears)

    # --- the vegetation index ------------------------------------
    print("\n=== vegetation index: MODIS items over one growing season ===")
    modis_ids = [c.id for c in candidates if "modis" in c.id.lower()]
    print(f"  candidate collections: {modis_ids}")

    # 13Q1 is the 16-day vegetation index product at 250 m.
    preferred = [c for c in modis_ids if "13Q1" in c or "13q1" in c]
    print(f"  looking for a 13Q1 product: {preferred or 'none found, listing all'}")

    for cid in (preferred or modis_ids[:3]):
        try:
            search = cat.search(
                collections=[cid],
                bbox=STORY_COUNTY_BBOX,
                datetime=f"{YEAR}-04-01/{YEAR}-10-31",
            )
            items = list(search.items())
        except Exception as e:
            print(f"  {cid}: search failed, {e}")
            continue

        print(f"\n  {cid}: {len(items)} item(s) between April and October {YEAR}")
        if not items:
            continue
        it = items[0]
        print(f"    id        : {it.id}")
        print(f"    datetime  : {it.datetime}")
        print(f"    assets    :")
        for name, asset in sorted(it.assets.items()):
            print(f"      {name:<34} {asset.media_type}")
        print(f"    properties of interest:")
        for k in ("proj:epsg", "proj:shape", "proj:transform", "gsd",
                  "modis:vertical-tile", "modis:horizontal-tile"):
            if k in it.properties:
                print(f"      {k}: {it.properties[k]}")

        # item.datetime is None on both collections: these are interval
        # products carrying start_datetime and end_datetime instead of
        # an instant. Reading .datetime.date() crashes on exactly the
        # products this project depends on.
        dates = sorted({item_date(i) for i in items})
        print(f"    composite start dates ({len(dates)}): {dates}")

        tiles = sorted({
            (i.properties.get("modis:horizontal-tile"),
             i.properties.get("modis:vertical-tile"))
            for i in items
        })
        print(f"    MODIS tiles covering this one county: {tiles}")
        print(f"    items = {len(dates)} dates x {len(tiles)} tiles "
              f"= {len(dates) * len(tiles)} (actual {len(items)})")

        # The CRS is not in item properties for these collections, so
        # read it off the asset. This decides which grid everything
        # gets reprojected onto.
        asset = it.assets["250m_16_days_NDVI"]
        print(f"    proj:epsg in asset   : "
              f"{asset.extra_fields.get('proj:epsg', 'absent')}")
        print(f"    proj:wkt2 present    : "
              f"{'proj:wkt2' in asset.extra_fields or 'proj:wkt2' in it.properties}")


if __name__ == "__main__":
    main()
