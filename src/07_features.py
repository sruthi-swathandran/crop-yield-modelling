"""
Turn fourteen NDVI composites per county-year into features, and join
them to the NASS yields.

    python src/07_features.py

Reads   data/ndvi/ndvi_*.parquet, data/masks/corn_*.npz,
        data/nass_corn_yields.csv
Writes  data/model_table.parquet, one row per county-year

WHY COMPOSITE INDEX AND NOT DATE
--------------------------------
MOD13Q1 composites start on fixed days of year, 081, 097, 113 and so
on, in every year including leap years. So the day of year identifies
the composite exactly, and composite 9 (day 209) is late July in 2008
and in 2019 alike. Indexing by composite makes a feature comparable
across years; indexing by date would give different column names in
every year.

THE LEAP YEAR TRAP
------------------
06 searched 1 April to 31 October. In a leap year the composite that
starts on day 305 falls on 31 October and gets caught; in other years
it starts on 1 November and does not. Leap years therefore arrive with
fifteen composites and the rest with fourteen.

Left alone, every leap year's season integral would gain one extra
value, and its end-of-season features would come from a week later.
2012 is a leap year. A drought signal confounded with a counting
artefact is worse than no signal, so every year is cut to the same
fourteen composites before any feature is computed.

TWO KINDS OF FEATURE
--------------------
The raw curve, ndvi_01 to ndvi_15, which lets the model find whatever
it finds and makes no assumption about which part of the season
matters.

And phenology, which encodes what agronomy says matters: how high the
canopy got, when it peaked, how fast it closed, how long it stayed
green, how fast it died back. These are interpretable in a way that
"the value on 11 July" is not, and a model that leans on them is
easier to defend.

Both are included. Which the model actually uses is a result rather
than an assumption.

THE TREND PROBLEM
-----------------
Yields rose 2.1 bushels per acre per year across this period through
breeding and management, independent of weather. That is +21% over
sixteen years, and over twelve years it is worth about as much as the
2012 drought cost.

A model given the year can learn that trend and look like it is
predicting yield while really predicting the calendar. So year is not
a feature here. The detrending decision belongs in the modelling step,
where its effect can be measured, and this script only records what
the trend is so that decision is informed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
NDVI_DIR = ROOT / "data" / "ndvi"
MASK_DIR = ROOT / "data" / "masks"
YIELDS = ROOT / "data" / "nass_corn_yields.csv"
OUT = ROOT / "data" / "model_table.parquet"

FIRST_DOY = 81          # composite 1 starts on day 081, 21 or 22 March
N_COMPOSITES = 14       # days 081 to 289, present in every year


def phenology(g: pd.DataFrame) -> pd.Series:
    """Summarise one county-year's curve."""
    g = g.sort_values("composite")
    v = g.ndvi.to_numpy(dtype="float64")
    idx = g.composite.to_numpy()

    if np.isfinite(v).sum() < 8:
        # Fewer than eight usable composites cannot describe a season.
        return pd.Series(dtype="float64")

    peak_i = int(np.nanargmax(v))
    peak = float(v[peak_i])
    first, last = float(np.nanmin(v[:3])), float(np.nanmean(v[-2:]))

    # Green-up and senescence as slopes per composite, which is per
    # 16 days. Computed from the trough before the peak rather than
    # from composite 1, because a late spring shifts the start.
    trough_i = int(np.nanargmin(v[: peak_i + 1])) if peak_i > 0 else 0
    greenup = (peak - v[trough_i]) / max(peak_i - trough_i, 1)
    senesc = (v[-1] - peak) / max(len(v) - 1 - peak_i, 1)

    return pd.Series({
        "ndvi_peak": peak,
        "ndvi_peak_composite": float(idx[peak_i]),
        "ndvi_mean": float(np.nanmean(v)),
        "ndvi_integral": float(np.nansum(v)),          # season-long greenness
        "ndvi_start": first,
        "ndvi_end": last,
        "ndvi_amplitude": peak - first,
        "greenup_rate": float(greenup),
        "senescence_rate": float(senesc),
        "composites_above_0.7": float(np.nansum(v > 0.7)),
        "composites_usable": float(np.isfinite(v).sum()),
    })


def main() -> None:
    tee_to("features")
    files = sorted(NDVI_DIR.glob("ndvi_*.parquet"))
    if not files:
        raise SystemExit(f"no NDVI files in {NDVI_DIR}; run 06 first")
    print(f"NDVI files: {len(files)}  ({files[0].name} .. {files[-1].name})")

    ndvi = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  rows {len(ndvi):,}  counties {ndvi.fips.nunique()}  "
          f"years {ndvi.year.min()}-{ndvi.year.max()}")

    # Composite number from day of year rather than by counting rows.
    # Counting would shift every later column by one if a single
    # composite were missing for a county, and nothing would error.
    doy = pd.to_datetime(ndvi.date).dt.dayofyear
    off_grid = ((doy - FIRST_DOY) % 16 != 0).sum()
    if off_grid:
        raise SystemExit(f"{off_grid} rows not on the 16-day grid; check 06")
    ndvi["composite"] = (doy - FIRST_DOY) // 16 + 1

    per_year = ndvi.groupby("year").composite.max()
    print(f"  composites per year before trimming: "
          f"{per_year.min()} to {per_year.max()}")
    print(f"  leap years with a 15th: "
          f"{sorted(per_year[per_year > N_COMPOSITES].index.tolist())}")
    ndvi = ndvi[ndvi.composite.between(1, N_COMPOSITES)].copy()
    print(f"  trimmed to composites 1 to {N_COMPOSITES} in every year")

    # --- the curve, as fifteen columns ---------------------------
    wide = (ndvi.pivot_table(index=["fips", "year"],
                             columns="composite", values="ndvi")
            .rename(columns=lambda c: f"ndvi_{int(c):02d}"))

    # --- phenology ------------------------------------------------
    phen = (ndvi.groupby(["fips", "year"], group_keys=True)
            .apply(phenology, include_groups=False)
            .dropna(how="all"))
    print(f"  county-years with a usable curve: {len(phen):,}")

    # --- how much corn, and how much of the county was readable ---
    masks = {}
    for f in sorted(MASK_DIR.glob("corn_*.npz")):
        yr = int(f.stem.split("_")[1])
        z = np.load(f)
        for fips in z.files:
            masks[(fips, yr)] = float(z[fips].sum())     # corn pixel-equivalents
    corn = pd.Series(masks, name="corn_px").rename_axis(["fips", "year"])

    usable = (ndvi.groupby(["fips", "year"])
              .apply(lambda g: g.usable_px.sum() / max(g.county_px.sum(), 1),
                     include_groups=False)
              .rename("usable_fraction"))

    # --- join to yields -------------------------------------------
    y = pd.read_csv(YIELDS, dtype={"fips": str})
    y = y[y.year.between(2008, 2021)]       # CDL coverage, see 03
    print(f"  yield rows in period: {len(y):,}")

    table = (y.set_index(["fips", "year"])
             .join(phen, how="inner")
             .join(wide, how="left")
             .join(corn, how="left")
             .join(usable, how="left")
             .reset_index())

    print(f"\n  model table: {len(table):,} county-years, "
          f"{table.fips.nunique()} counties, "
          f"{table.year.min()}-{table.year.max()}")
    print(f"  yields without NDVI: {len(y) - len(table):,}")

    # --- the trend, recorded rather than removed -----------------
    by_year = table.groupby("year").yield_bu_acre.mean()
    slope = np.polyfit(by_year.index, by_year.values, 1)[0]
    print(f"\n  mean yield by year:")
    print(by_year.round(1).to_string())
    print(f"  linear trend: {slope:+.2f} bu/acre/year "
          f"({slope*(2021-2008):+.0f} across the period)")
    print("  Not removed here. The modelling step decides and measures.")

    # --- checks ---------------------------------------------------
    print("\n--- checks ---")
    print(f"  duplicate county-years : {table.duplicated(['fips','year']).sum()}")
    print(f"  missing ndvi_peak      : {table.ndvi_peak.isna().sum()}")
    print(f"  ndvi_peak range        : {table.ndvi_peak.min():.3f} "
          f"to {table.ndvi_peak.max():.3f}")
    print(f"  peak composite, median : {table.ndvi_peak_composite.median():.0f} "
          f"(composite 9 is late July)")
    print(f"  corn_px zero           : {(table.corn_px == 0).sum()}")

    # Correlation is not the model, but a feature set with no relation
    # to the target at all would mean something upstream is broken.
    for col in ["ndvi_peak", "ndvi_integral", "greenup_rate", "corn_px"]:
        r = table[col].corr(table.yield_bu_acre)
        print(f"  corr({col}, yield) = {r:+.3f}")

    OUT.parent.mkdir(exist_ok=True)
    table.to_parquet(OUT, index=False)
    print(f"\nwritten: {OUT}  ({len(table.columns)} columns)")


if __name__ == "__main__":
    main()
