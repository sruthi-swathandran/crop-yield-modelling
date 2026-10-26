"""
Corn-weighted Sentinel-2 NDVI at 20 m for every Iowa county, 2017 to
2024, on the same 16-day windows as MOD13Q1.

    python src/17c_export_corn30.py      (cropyield-cdl, once, first)
    python src/17_extract_s2.py          (cropyield)

Writes data/s2/s2_ndvi_YEAR.parquet, one row per county and window.
Progress is saved after every county in data/s2/partial_YEAR.csv; a
rerun carries on from the last county finished. About 50 hours in all.

WHY 20 m, AND HOW
-----------------
Sentinel-2's red and near infrared are 10 m bands, read here from their
20 m overviews; the scene classification layer is native 20 m. 17b
measured the connection at about 0.8 million pixels a second whichever
way the files are read, so time scales with pixels, and 10 m would take
four times as long for a corn map that is 30 m anyway.

ONE DATE PER COUNTY PER WINDOW
------------------------------
MOD13Q1 gives one value per 16-day window, chosen pixel by pixel as the
best of that window. The closest affordable equivalent here is the
clearest date in the window for each county:

  1. scenes over the county in the window, after removing duplicates
     and anything over 50% cloud by the scene's own metadata
  2. dates ranked by the mean cloud cover of their scenes
  3. the best date read, and the share of the county's corn that is
     clear (scene classification 4 or 5) computed
  4. if that share is under 30%, the next date is tried, up to three;
     the clearest of those attempted is kept, with its share recorded

So the Sentinel-2 value is one day's view and the MODIS value is a
pixel-wise best of sixteen days. That difference is part of what is
being compared, and is stated with the results.

WEIGHTING
---------
As for MODIS: each 20 m pixel counts in proportion to its corn
fraction, averaged from the 30 m corn map onto the Sentinel-2 grid, and
only clear pixels inside the county count. The share of the county's
corn weight that was clear is saved with every value.

THREE TRAPS, HANDLED
--------------------
Duplicates. 2022 and 2023 list most scenes twice, under two processing
baselines (17b: 2,197 and 2,633 extra). One per tile and sensing time
is kept, the newest baseline.

The reflectance offset. From baseline 04.00 (2022 on) Level-2A values
carry an offset of 1000, removed before NDVI is computed. The label is
not trusted on its own: one 2021 scene labelled 04.00 turned out to have
no offset in its data, so whether to remove it is decided from the
values (see detect_offset).

Tile overlap. Neighbouring tiles overlap by about 10 km. When one scene
covers at least 97% of a county it is used alone; otherwise every scene
of that date is used and pixels in the overlap count twice. That
reweights a thin strip slightly and is preferable to dropping half a
county.
"""

import os
import time
from pathlib import Path

for k, v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
             "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
             "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
             "GDAL_HTTP_MULTIPLEX": "YES", "GDAL_HTTP_VERSION": "2",
             "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "3"}.items():
    os.environ.setdefault(k, v)

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import shape

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
S2DIR = ROOT / "data" / "s2"
CORN30 = S2DIR / "corn30"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

YEARS = range(2017, 2025)
FIRST_DOY, N_WIN = 81, 14
MAX_CLOUD = 50
MIN_CLEAR = 0.30
TRIES = 3
COVER_ALONE = 0.97
GOOD_SCL = (4, 5)


def with_retries(fn, *args, tries=4, label=""):
    for attempt in range(1, tries + 1):
        try:
            return fn(*args)
        except Exception as e:                       # noqa: BLE001
            if attempt == tries or type(e).__name__ == "NoDataInBounds":
                raise
            print(f"      {label} read failed ({type(e).__name__}), retry "
                  f"{attempt}/{tries - 1}", flush=True)
            time.sleep(5 * attempt)


def dedupe(items):
    best = {}
    for it in items:
        key = (it.properties.get("s2:mgrs_tile"), it.properties["datetime"][:19])
        b = str(it.properties.get("s2:processing_baseline", "0"))
        if key not in best or b > str(best[key].properties.get("s2:processing_baseline", "0")):
            best[key] = it
    return list(best.values())


def scenes_for_year(cat, bounds, year):
    its = dedupe(list(cat.search(collections=["sentinel-2-l2a"], bbox=list(bounds),
                                 datetime=f"{year}-03-21/{year}-10-31").items()))
    rows = []
    for it in its:
        cc = it.properties.get("eo:cloud_cover", 100)
        if cc >= MAX_CLOUD:
            continue
        date = pd.Timestamp(it.properties["datetime"][:10])
        w = (date.dayofyear - FIRST_DOY) // 16 + 1
        if 1 <= w <= N_WIN:
            rows.append({"item": it, "date": date, "window": w, "cloud": cc,
                         "baseline": str(it.properties.get("s2:processing_baseline", "0")),
                         "geometry": shape(it.geometry)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def read_scene(it, geom):
    """Red and NIR from their 20 m overviews and the 20 m scene
    classification, cut to the county's bounding box."""
    out = {}
    for asset, ov, masked in (("B04", 0, True), ("B08", 0, True), ("SCL", None, False)):
        kw = {"overview_level": ov} if ov is not None else {}
        def read(href=it.assets[asset].href):
            a = rioxarray.open_rasterio(pc.sign(href), masked=masked, **kw).squeeze()
            g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(a.rio.crs)
            return a.rio.clip_box(*g.total_bounds).load()
        out[asset] = with_retries(read, label=f"{asset}")
    if out["SCL"].shape != out["B04"].shape:
        out["SCL"] = out["SCL"].rio.reproject_match(out["B04"], resampling=Resampling.nearest)
    return out


def detect_offset(red_dn, baseline):
    """The reflectance offset, decided by the data rather than the label.

    The first version trusted the processing baseline label. A 2021
    scene reprocessed in May 2022 (T15TWF, 25 September 2021) is labelled
    04.00, but its red values have a median of 658 and 79% fall under
    1000, so it never carried the offset. Subtracting 1000 made red
    negative and NDVI 1.04. With a real offset, land values under 1000
    would mean negative reflectance and are rare, so the offset is
    removed only when fewer than 5% of values fall under 1000.
    """
    if baseline < "04.00":
        return 0.0
    v = red_dn[np.isfinite(red_dn)]
    if v.size == 0:
        return 1000.0
    return 1000.0 if np.mean(v < 1000) < 0.05 else 0.0


def county_on_date(scenes, geom, corn30, corn_total):
    """NDVI and clear share for one county on one date."""
    eq = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:5070").iloc[0]
    area = eq.area
    ov = [(s, gpd.GeoSeries([s.geometry], crs="EPSG:4326").to_crs("EPSG:5070")
           .iloc[0].intersection(eq).area / area) for s in scenes.itertuples()]
    ov.sort(key=lambda t: -t[1])
    use = [ov[0][0]] if ov[0][1] >= COVER_ALONE else [s for s, _ in ov]

    num = den = 0.0
    for s in use:
        try:
            b = read_scene(s.item, geom)
        except Exception as e:                       # noqa: BLE001
            if type(e).__name__ == "NoDataInBounds":
                continue
            raise
        red = b["B04"]
        offset = detect_offset(red.values, s.baseline)
        r = (red.values.astype("float64") - offset) * 1e-4
        n = (b["B08"].values.astype("float64") - offset) * 1e-4
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (n - r) / (n + r)
        g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(red.rio.crs)
        inside = ~geometry_mask(g.geometry, out_shape=red.shape,
                                transform=red.rio.transform())
        f = (corn30.rio.reproject_match(red, resampling=Resampling.average)
             .values.astype("float64"))
        ok = inside & np.isin(b["SCL"].values, GOOD_SCL) & np.isfinite(ndvi) & np.isfinite(f)
        # ndvi * f is evaluated everywhere before np.where picks the valid
        # pixels, and infinite NDVI (red + NIR = 0) times zero weight raised
        # a warning on most counties. Those pixels are never selected.
        with np.errstate(invalid="ignore"):
            num += float(np.sum(np.where(ok, ndvi * f, 0.0)))
        den += float(np.sum(np.where(ok, f, 0.0)))
    if den <= 0:
        return np.nan, 0.0, len(use)
    return num / den, min(1.0, den / corn_total), len(use)


def run_year(cat, ia, year):
    final = S2DIR / f"s2_ndvi_{year}.parquet"
    if final.exists():
        print(f"{year}: done, skipping")
        return
    part = S2DIR / f"partial_{year}.csv"
    done = set(pd.read_csv(part, dtype={"fips": str}).fips) if part.exists() else set()
    print(f"\n=== {year} ===  counties already done: {len(done)}", flush=True)
    sc = scenes_for_year(cat, ia.total_bounds, year)
    print(f"  scenes under {MAX_CLOUD}% cloud, after duplicates removed: {len(sc)}")

    t0, n_new = time.time(), 0
    todo = ia[~ia.fips.isin(done)]
    for k, row in enumerate(todo.itertuples(), 1):
        corn = rioxarray.open_rasterio(CORN30 / str(year) / f"{row.fips}.tif").squeeze()
        corn = corn.astype("float32").rio.write_nodata(np.nan)
        # Total corn weight inside the county, in 20 m pixel units, for
        # the clear share. The 30 m count is scaled by (30/20)^2.
        g5070 = gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_crs("EPSG:5070")
        inside30 = ~geometry_mask(g5070.geometry, out_shape=corn.shape,
                                  transform=corn.rio.transform())
        corn_total = float(np.nansum(np.where(inside30, corn.values, 0))) * 2.25
        here = sc[sc.intersects(row.geometry)]

        rows = []
        for w in range(1, N_WIN + 1):
            cand = here[here.window == w]
            if cand.empty:
                rows.append({"fips": row.fips, "year": year, "window": w, "date": None,
                             "ndvi": np.nan, "clear_share": 0.0, "scenes": 0, "tries": 0})
                continue
            dates = cand.groupby("date").cloud.mean().sort_values().index[:TRIES]
            best = None
            for t, d in enumerate(dates, 1):
                v, clear, nsc = county_on_date(cand[cand.date == d], row.geometry,
                                               corn, corn_total)
                if best is None or clear > best["clear_share"]:
                    best = {"fips": row.fips, "year": year, "window": w,
                            "date": d.date().isoformat(), "ndvi": v,
                            "clear_share": clear, "scenes": nsc, "tries": t}
                if clear >= MIN_CLEAR:
                    break
            rows.append(best)

        pd.DataFrame(rows).to_csv(part, mode="a", header=not part.exists(), index=False)
        n_new += 1
        el = time.time() - t0
        left = (len(todo) - k) * el / n_new / 3600
        got = sum(1 for r in rows if r["clear_share"] >= MIN_CLEAR)
        print(f"    {year} {len(done) + k:>3}/{len(ia)} {row.fips} "
              f"{row.county_name[:14]:<14} windows clear {got:>2}/{N_WIN}   "
              f"{el/3600:5.2f} h, ~{left:5.2f} h left this year", flush=True)

    df = pd.read_csv(part, dtype={"fips": str}).drop_duplicates(["fips", "window"], keep="last")
    df.to_parquet(final, index=False)
    ok = df[df.clear_share >= MIN_CLEAR]
    print(f"  {year}: {len(df)} rows, {df.fips.nunique()} counties, "
          f"{100 * len(ok) / len(df):.0f}% of county-windows at least {MIN_CLEAR:.0%} clear")
    story = ok[ok.fips == "19169"]
    if len(story):
        i = story.ndvi.idxmax()
        print(f"  Story County peak: {story.loc[i, 'ndvi']:.3f} on {story.loc[i, 'date']}")


def main() -> None:
    tee_to("s2")
    try:
        pd.DataFrame({"x": [1]}).to_parquet(__import__("io").BytesIO())
    except ImportError as e:
        raise SystemExit(f"cannot write parquet ({e}); use the cropyield environment")
    counties = gpd.read_file(COUNTIES, layer="counties")
    ia = counties[counties.state_alpha == "IA"].sort_values("fips").reset_index(drop=True)
    missing = [y for y in YEARS if not (CORN30 / str(y)).exists()]
    if missing:
        raise SystemExit(f"no corn maps for {missing}; run 17c_export_corn30.py first")
    cat = pystac_client.Client.open(STAC)
    for y in YEARS:
        run_year(cat, ia, y)


if __name__ == "__main__":
    main()
