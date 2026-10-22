"""
Has Terra's orbit drift changed the NDVI it records? Measured on forest,
which should not change.

    python src/16_drift_check.py

Runs in the cropyield environment. About 75 minutes the first time:
45 building forest masks from two years of the corn map, 30 reading
NDVI. Both are cached.

Reads   data/counties.gpkg, data/model_table.parquet,
        data/model_table_recent.parquet
Writes  data/masks/forest_stable_2008_2021.npz (and the 2021-only
        version for comparison), data/forest_ndvi.parquet,
        results/drift_forest_by_year.csv, logs/drift.log

THE IDEA
--------
Terra has been crossing earlier since 2020, and the sun zenith angle at
overpass over Story County rose from 28 to 30 degrees through 2022 to 32
in 2023 and 35 in 2024. A lower sun changes shadowing and the path light
takes through the canopy, which can move NDVI with nothing on the ground
having changed.

Cropland cannot show that cleanly, because the crop changes every year.
Forest mostly does not. So the test uses pixels that are at least 90%
stable forest (Cropland Data Layer classes 63 and 141 to 143, below), in the counties
with the most such pixels, and follows their midsummer NDVI from 2008
to 2024. Forest NDVI still moves with the weather, so the question is
not whether 2023 and 2024 differ from earlier years but whether they
differ by more than years normally do.

WHAT IS COMPARED
----------------
For each county, mean NDVI over its pure forest pixels in composites 8
and 9 (about 12 July to 12 August), using only pixels MODIS rates as
reliable. Each county's 2008 to 2021 mean is subtracted, leaving an
anomaly, and the anomalies are averaged across counties for each year.

  drift test    2022, 2023 and 2024 against the spread of the 2008 to
                2021 annual anomalies
  trend test    the 2008 to 2021 slope in forest NDVI, against the slope
                in corn peak NDVI over the same years. If corn greened
                and forest stayed flat, the greening is in the canopy
                rather than the sensor. Intervals resample years, since
                counties in one season share its weather.

STABLE FOREST
-------------
A pixel counts only if it was forest in both the 2008 and the 2021 map,
decided at 30 m before averaging onto the 250 m MODIS grid. A stand cut
or blown down in between drops out rather than adding a sudden change
of its own. The 2008 map is resampled onto the 2021 grid by nearest
neighbour, which keeps class codes intact. The 2021-only fraction is
kept too, so the share removed by the stability rule is reported.

The forest codes 141 to 143 sit in the part of the Cropland Data Layer's
legend derived from the National Land Cover Database (codes 110 to 195),
so they reflect that mapping more than NASS's crop classifier. Large
blocks of forest are what it is reliable for, which is what a 90%
threshold on a 250 m pixel selects.

THE EDGE ARTEFACT
-----------------
05 read the corn map with no margin around a county. Here the read has
a 500 m margin, and only pixels at least 90% forest count, so pixels
with partial coverage are excluded anyway.
"""

import importlib
import os
import time
from pathlib import Path

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import rioxarray  # noqa: F401
from rasterio.enums import Resampling
from shapely.geometry import box, shape

from runlog import tee_to

m05 = importlib.import_module("05_build_corn_masks")

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
MASKS = ROOT / "data" / "masks"
# Named after the forest definition, so NDVI read for one set of counties
# is never reused for another. The first run used 2021-only forest; its
# outputs are kept with a _2021only suffix for comparison.
TAG = "stable_2008_2021"
OUT = ROOT / "data" / f"forest_ndvi_{TAG}.parquet"
RESULTS = ROOT / "results"

# 141 deciduous, 142 evergreen, 143 mixed, and 63, the non-crop "Forest"
# class. Legends have varied between states and years (Illinois folded
# some forest classes together), so all four count, and the 2008 test of
# stability does not drop real forest coded differently.
FOREST = [63, 141, 142, 143]
REF_YEAR = 2021
EARLY_YEAR = 2008                 # a pixel must be forest in both
PURE = 0.9
MIN_PURE_PX = 100                 # about 5 km2 of pure forest
MAX_COUNTIES = 80
COMPOSITES = (8, 9)
FIRST_DOY = 81
YEARS = range(2008, 2025)
MARGIN = 500
NDVI, REL, SZA = ("250m_16_days_NDVI", "250m_16_days_pixel_reliability",
                  "250m_16_days_sun_zenith_angle")
B = 2000
SEED = 5


def with_retries(fn, *args, tries=4, label=""):
    for attempt in range(1, tries + 1):
        try:
            return fn(*args)
        except Exception as e:                       # noqa: BLE001
            # A tile footprint that touches the county while the raster
            # does not is an answer, not a network fault. Retrying it
            # three times cost the first run 30 seconds per such county.
            if attempt == tries or type(e).__name__ == "NoDataInBounds":
                raise
            print(f"    {label} read failed ({type(e).__name__}), retry "
                  f"{attempt}/{tries - 1}", flush=True)
            time.sleep(5 * attempt)


def load_grids(cat, counties, year):
    items = [i for i in cat.search(
        collections=["modis-13Q1-061"], bbox=list(counties.total_bounds),
        datetime=f"{year}-07-01/{year}-07-20").items() if i.id.startswith("MOD")]
    grids = {}
    for it in items:
        key = (it.properties["modis:horizontal-tile"], it.properties["modis:vertical-tile"])
        if key not in grids:
            print(f"    loading tile h{key[0]:02d}v{key[1]:02d}", flush=True)
            grids[key] = with_retries(lambda h: m05.open_signed(h).load(),
                                      it.assets[NDVI].href, label=str(key))
    return grids


def template_and_tile(grids, geom):
    """05's template_for, also saying which tile it used, so the NDVI
    read later comes from the same grid."""
    g4326 = gpd.GeoSeries([geom], crs="EPSG:4326")
    for key, arr in grids.items():
        g = g4326.to_crs(arr.rio.crs)
        minx, miny, maxx, maxy = g.total_bounds
        ax0, ay0, ax1, ay1 = arr.rio.bounds()
        if maxx < ax0 or minx > ax1 or maxy < ay0 or miny > ay1:
            continue
        try:
            c = arr.rio.clip_box(minx, miny, maxx, maxy).rio.clip(g.geometry, g.crs, drop=True)
        except Exception:
            continue
        if c.size and bool(np.isfinite(c).any()):
            return key, c
    return None, None


def cdl_footprints(cat, counties, year):
    items = [it for it in cat.search(collections=["usda-cdl"],
                                     bbox=list(counties.total_bounds),
                                     datetime=f"{year}-01-01/{year}-12-31").items()
             if "cropland" in it.assets]
    return gpd.GeoDataFrame({"item": items},
                            geometry=[shape(i.geometry) if i.geometry else box(*i.bbox)
                                      for i in items], crs="EPSG:4326")


def read_cdl(fp, geom, label):
    """One county's corn map at 30 m, from however many Planetary
    Computer tiles it touches, with a margin."""
    parts = []
    for it in fp[fp.intersects(geom)].item:
        def read(href):
            arr = m05.open_signed(href)
            g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(arr.rio.crs)
            minx, miny, maxx, maxy = g.total_bounds
            sub = arr.rio.clip_box(minx - MARGIN, miny - MARGIN, maxx + MARGIN, maxy + MARGIN)
            return sub.rio.clip(g.geometry, g.crs, drop=True).load()
        try:
            parts.append(with_retries(read, it.assets["cropland"].href, label=label))
        except Exception:
            continue                           # footprint touches, raster does not
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    from rioxarray.merge import merge_arrays
    return merge_arrays(parts)


def build_forest_masks(cat, counties):
    tag = f"{EARLY_YEAR}_{REF_YEAR}"
    path = MASKS / f"forest_stable_{tag}.npz"
    path_ref = MASKS / f"forest_{REF_YEAR}_only.npz"
    tiles_path = MASKS / f"forest_stable_{tag}_tiles.csv"
    if path.exists() and tiles_path.exists():
        print(f"forest masks cached: {path.name}")
        t = pd.read_csv(tiles_path, dtype={"fips": str})
        return (dict(np.load(path)), dict(np.load(path_ref)),
                {r.fips: (r.h, r.v) for r in t.itertuples()})

    print(f"--- forest masks: forest in both {EARLY_YEAR} and {REF_YEAR} ---")
    fp_ref = cdl_footprints(cat, counties, REF_YEAR)
    fp_early = cdl_footprints(cat, counties, EARLY_YEAR)
    grids = load_grids(cat, counties, REF_YEAR)
    stable, ref_only, tiles, t0 = {}, {}, {}, time.time()
    for n, row in enumerate(counties.itertuples(), 1):
        key, template = template_and_tile(grids, row.geometry)
        if template is None:
            continue
        c_ref = read_cdl(fp_ref, row.geometry, row.fips)
        c_early = read_cdl(fp_early, row.geometry, row.fips)
        if c_ref is None or c_early is None:
            continue
        # The early map onto the reference map's 30 m grid. Nearest
        # neighbour, because averaging class codes would invent classes.
        c_early = c_early.rio.reproject_match(c_ref, resampling=Resampling.nearest)
        f_ref = c_ref.isin(FOREST)
        f_both = f_ref & c_early.isin(FOREST)
        for store, f in ((ref_only, f_ref), (stable, f_both)):
            frac = (f.astype("float32").rio.write_nodata(np.nan)
                    .rio.reproject_match(template, resampling=Resampling.average)
                    .fillna(0))
            store[row.fips] = frac.values.astype("float32")
        tiles[row.fips] = key
        if n % 25 == 0 or n == len(counties):
            el = time.time() - t0
            print(f"    {n:>3}/{len(counties)}   {el/60:4.1f} min, "
                  f"~{(len(counties) - n) * el / n / 60:4.1f} min left", flush=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **stable)
    np.savez_compressed(path_ref, **ref_only)
    pd.DataFrame([(f, h, v) for f, (h, v) in tiles.items()],
                 columns=["fips", "h", "v"]).to_csv(tiles_path, index=False)
    return stable, ref_only, tiles


def tile_items(cat, key, year):
    its = [i for i in cat.search(
        collections=["modis-13Q1-061"],
        query={"modis:horizontal-tile": {"eq": key[0]},
               "modis:vertical-tile": {"eq": key[1]}},
        datetime=f"{year}-07-01/{year}-08-20").items() if i.id.startswith("MOD")]
    by_date = {}
    for it in sorted(its, key=lambda i: i.id):          # one item per date
        by_date[it.properties["start_datetime"]] = it
    out = []
    for sd, it in sorted(by_date.items()):
        doy = pd.Timestamp(sd[:10]).dayofyear
        comp = (doy - FIRST_DOY) // 16 + 1
        if comp in COMPOSITES and (doy - FIRST_DOY) % 16 == 0:
            out.append((comp, sd[:10], it))
    return out


def extract(cat, counties, masks, tiles, chosen):
    rows, mismatched = [], 0
    geo = counties.set_index("fips").geometry
    by_tile = {}
    for f in chosen:
        by_tile.setdefault(tiles[f], []).append(f)
    t0 = time.time()
    for yi, year in enumerate(YEARS, 1):
        for key, fl in sorted(by_tile.items()):
            for comp, date, it in tile_items(cat, key, year):
                probe = m05.open_signed(it.assets[NDVI].href)
                crs = probe.rio.crs
                gs = {f: gpd.GeoSeries([geo[f]], crs="EPSG:4326").to_crs(crs) for f in fl}
                ub = gpd.GeoSeries(pd.concat([g for g in gs.values()]), crs=crs).total_bounds

                def window(href):
                    return m05.open_signed(href).rio.clip_box(*ub).load()
                arr = {a: with_retries(window, it.assets[a].href, label=f"{key} {date}")
                       for a in (NDVI, REL, SZA)}
                for f in fl:
                    g = gs[f]
                    sub = {a: arr[a].rio.clip_box(*g.total_bounds)
                           .rio.clip(g.geometry, g.crs, drop=True).values for a in arr}
                    m = masks[f]
                    if sub[NDVI].shape != m.shape:
                        mismatched += 1
                        continue
                    sel = (m >= PURE) & np.isin(sub[REL], (0, 1)) & np.isfinite(sub[NDVI])
                    if sel.sum() < 10:
                        continue
                    rows.append({"fips": f, "year": year, "date": date, "composite": comp,
                                 "ndvi": float(np.mean(sub[NDVI][sel]) * 1e-4),
                                 "sza": float(np.nanmean(sub[SZA][m >= PURE]) * 0.01),
                                 "pixels": int(sel.sum())})
        el = time.time() - t0
        print(f"  {year}: {len(rows):,} rows so far   {el/60:4.1f} min, "
              f"~{(len(YEARS) - yi) * el / yi / 60:4.1f} min left", flush=True)
    if mismatched:
        print(f"  grid mismatches skipped: {mismatched}")
    return pd.DataFrame(rows)


def analyse(df):
    cy = df.groupby(["fips", "year"])[["ndvi", "sza"]].mean().reset_index()
    base = cy[cy.year <= 2021].groupby("fips").ndvi.mean().rename("base")
    cy = cy.join(base, on="fips").assign(anom=lambda d: d.ndvi - d.base)
    yr = cy.groupby("year").agg(counties=("fips", "nunique"), ndvi=("ndvi", "mean"),
                                anomaly=("anom", "mean"), sza=("sza", "mean"))
    ref = yr.loc[2008:2021, "anomaly"]
    sd = ref.std(ddof=1)
    yr["z_vs_2008_2021"] = (yr.anomaly - ref.mean()) / sd
    print("\n--- forest NDVI by year, composites 8 and 9, pure forest pixels ---")
    print(yr.round(4).to_string())
    print(f"\n  year-to-year spread of the annual anomaly, 2008-2021: sd {sd:.4f}, "
          f"range {ref.min():+.4f} to {ref.max():+.4f}")
    for y in (2022, 2023, 2024):
        if y in yr.index:
            a = yr.loc[y, "anomaly"]
            inside = ref.min() <= a <= ref.max()
            print(f"  {y}: anomaly {a:+.4f}, z {yr.loc[y, 'z_vs_2008_2021']:+.2f}, "
                  f"{'inside' if inside else 'OUTSIDE'} the 2008-2021 range, "
                  f"sun zenith {yr.loc[y, 'sza']:.1f} deg")

    # Trend 2008-2021. The interval resamples years, not counties: every
    # county shares a year's weather, so 14 seasons are the real sample
    # size. Resampling counties gave a falsely narrow interval that
    # excluded zero on synthetic data with no trend at all.
    rng = np.random.default_rng(SEED)
    f_yr = yr.loc[2008:2021, "anomaly"]
    crop = pd.read_parquet(ROOT / "data" / "model_table.parquet")
    c_yr = crop.groupby("year").ndvi_peak.mean().loc[2008:2021]

    def slope_ci(s):
        x, y = s.index.to_numpy(dtype=float), s.to_numpy()
        point = np.polyfit(x, y, 1)[0]
        boots = []
        for _ in range(B):
            i = rng.integers(0, len(x), len(x))
            if np.unique(x[i]).size > 2:
                boots.append(np.polyfit(x[i], y[i], 1)[0])
        return point, *np.percentile(boots, [2.5, 97.5])

    fs, flo, fhi = slope_ci(f_yr)
    cs, clo, chi = slope_ci(c_yr)
    print(f"\n--- trend 2008-2021, intervals from resampling years ---")
    print(f"  forest NDVI     {fs:+.5f} per year  [{flo:+.5f}, {fhi:+.5f}]  "
          f"({cy.fips.nunique()} counties)")
    print(f"  corn peak NDVI  {cs:+.5f} per year  [{clo:+.5f}, {chi:+.5f}]")
    print("  If the forest interval sits near zero and well below the corn slope,")
    print("  the corn greening is in the canopy, not the sensor.")
    return yr


def main() -> None:
    tee_to("drift")
    t0 = time.time()
    counties = gpd.read_file(COUNTIES, layer="counties")
    cat = pystac_client.Client.open(m05.STAC)

    try:
        pd.DataFrame({"x": [1]}).to_parquet(__import__("io").BytesIO())
    except ImportError as e:
        raise SystemExit(f"cannot write parquet ({e}); use the cropyield environment")

    masks, ref_only, tiles = build_forest_masks(cat, counties)
    pure = pd.Series({f: int((m >= PURE).sum()) for f, m in masks.items()})
    pure_ref = sum(int((m >= PURE).sum()) for m in ref_only.values())
    print(f"pure forest pixels: {pure_ref:,} forest in {REF_YEAR}, "
          f"{pure.sum():,} forest in both {EARLY_YEAR} and {REF_YEAR} "
          f"({100 * (1 - pure.sum() / max(pure_ref, 1)):.1f}% removed by the stability rule)")
    chosen = pure[pure >= MIN_PURE_PX].sort_values(ascending=False).head(MAX_COUNTIES)
    st = counties.set_index("fips").state_alpha
    print(f"\ncounties with at least {MIN_PURE_PX} pure forest pixels: "
          f"{(pure >= MIN_PURE_PX).sum()}; using the top {len(chosen)}")
    print("  by state: " + ", ".join(f"{s} {n}" for s, n in
                                       st[chosen.index].value_counts().items()))
    print(f"  pure forest pixels used: {chosen.sum():,} "
          f"(about {chosen.sum() * 0.0537:,.0f} km2)")

    if OUT.exists():
        print(f"\nforest NDVI cached: {OUT.name}")
        df = pd.read_parquet(OUT)
    else:
        print("\n--- reading forest NDVI, 2008 to 2024 ---")
        df = extract(cat, counties, masks, tiles, list(chosen.index))
        df.to_parquet(OUT, index=False)
    print(f"  rows {len(df):,}, counties {df.fips.nunique()}, years "
          f"{df.year.min()}-{df.year.max()}")

    yr = analyse(df)
    RESULTS.mkdir(exist_ok=True)
    yr.to_csv(RESULTS / f"drift_forest_by_year_{TAG}.csv")
    print(f"\nwritten: {RESULTS / f'drift_forest_by_year_{TAG}.csv'}")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
