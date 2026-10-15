"""
Light use efficiency model, scored on exactly the rows and folds the
gradient boosting baseline used.

    python src/10_lue_model.py

Reads   data/model_table.parquet, data/weather_composites.parquet,
        data/predictions_gbm.parquet
Writes  data/lue_inputs.parquet         fAPAR, APAR per county-year
        data/predictions_lue.parquet
        results/comparison.csv          every model, every scheme
        results/comparison_2012.csv
        logs/lue.log

THE MODEL
---------
Monteith's framework. Biomass is absorbed light times the efficiency
with which the crop turns it into dry matter; grain is a fixed share
of biomass.

    APAR    = fAPAR x PAR                     per 16-day composite
    biomass = RUE x sum(APAR)                 over the season
    grain   = HI x biomass

The three physical constants come from the literature. The NDVI end
members for fAPAR, below, come from this dataset's NDVI but never see
a yield.

RUE   3.8 g dry matter per MJ absorbed PAR, above ground. Lindquist et
      al. (2005), Agronomy Journal 97:72-78, measured in irrigated
      maize in Nebraska and recommended for models of yield without
      growth limitations. It is an optimum, so the uncalibrated model
      should overpredict.
HI    0.55. Ruiz et al. (2023), Field Crops Research, measured 54 US
      commercial hybrids released 1983 to 2020 and found harvest index
      rising from about 0.52 to 0.58. 0.55 sits in the range for
      hybrids of this period. An earlier draft used 0.50, below that
      range; the change makes the uncalibrated model overpredict more,
      not less, and leaves the calibrated versions untouched because k
      absorbs any constant multiplier.
PAR   0.48 of shortwave, on an energy basis. Published ratios run from
      about 0.41 to 0.50 depending on site and cloudiness.

FAPAR FROM NDVI
---------------
Sellers et al. (1994), International Journal of Remote Sensing
15:3519-3545, the method behind the ISLSCP FPAR fields. fAPAR is linear
in the simple ratio SR = (1 + NDVI) / (1 - NDVI), scaled between a
bare-soil end member and a full-canopy end member:

    fAPAR = 0.949 x (SR - SR02) / (SR98 - SR02) + 0.001,  clipped to 0.001..0.95

Sellers took SR02 and SR98 from the 2nd and 98th percentiles of the
NDVI distribution. Their published values are for AVHRR, which has a
narrower NDVI range than MODIS, so the same percentiles are taken here
from this dataset's own corn-weighted NDVI. That uses NDVI alone and
never sees a yield, so it cannot leak the target.

THE SEASON
----------
Composites 4 to 12, days 129 to 272, which is about 9 May to 29
September: emergence to physiological maturity for Corn Belt corn.
An agronomic choice made before seeing results, not tuned.

Clouded composites are filled by interpolating along the county's own
curve, so a cloudy county-year does not absorb less light on paper
than it did in the field.

THREE VERSIONS
--------------
lue_0   no fitted parameters at all. Literature values only.
lue_1   one scalar k multiplying the prediction, fitted on the training
        folds. It absorbs everything systematic: the RUE optimum, the
        harvest index guess, mixed pixels diluting NDVI.
lue_2   k plus a linear change in k over time, fitted on the training
        folds. Breeding has raised what a hybrid does with the light it
        absorbs, and that is the one thing the canopy cannot show.

UNITS
-----
Corn is sold at 15.5% moisture and 56 lb per bushel, so one bushel per
acre is 62.77 kg/ha as sold and 53.04 kg/ha of dry grain.
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from runlog import tee_to

base = importlib.import_module("09_baseline_gbm")   # same load, folds, scores

ROOT = Path(__file__).resolve().parent.parent
WEATHER = ROOT / "data" / "weather_composites.parquet"
GBM_PRED = ROOT / "data" / "predictions_gbm.parquet"
INPUTS_OUT = ROOT / "data" / "lue_inputs.parquet"
PRED_OUT = ROOT / "data" / "predictions_lue.parquet"
RESULTS = ROOT / "results"

RUE = 3.8               # g DM / MJ APAR      Lindquist et al. 2005
HI = 0.55               # grain / above-ground biomass   Ruiz et al. 2023
PAR_FRACTION = 0.48     # PAR / shortwave, energy basis
FPAR_MIN, FPAR_MAX = 0.001, 0.95       # Sellers et al. 1994
SEASON = range(4, 13)   # composites 4..12, days 129..272

LB_PER_BU, KG_PER_LB, HA_PER_ACRE, MOISTURE = 56, 0.45359237, 0.40468564, 0.155
KG_HA_DRY_PER_BU_AC = LB_PER_BU * KG_PER_LB * (1 - MOISTURE) / HA_PER_ACRE

CENTRE_YEAR = 2015      # lue_2 measures change in k from here
MODELS = ["gbm_raw", "gbm_detrended", "lue_0", "lue_1", "lue_2", "trend_only"]


def sr(ndvi):
    return (1 + ndvi) / (1 - ndvi)


def fill_curve(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    return np.interp(np.arange(v.size), np.flatnonzero(ok), v[ok])


def end_members(df: pd.DataFrame):
    curve = df[base.CURVE].to_numpy(dtype="float64")
    return tuple(np.percentile(curve[np.isfinite(curve)], [2, 98]))


def lue_inputs(df: pd.DataFrame, weather=WEATHER, ends=None) -> pd.DataFrame:
    """ends: the NDVI end members to scale fAPAR with. Left as None they
    come from df itself. For later seasons they must be passed in from the
    training years, or the fAPAR scale would shift with the test data."""
    curve = df[base.CURVE].to_numpy(dtype="float64")
    n02, n98 = ends if ends is not None else end_members(df)
    print(f"  NDVI end members: 2nd percentile {n02:.3f}, "
          f"98th percentile {n98:.3f}"
          f"{'  (passed in)' if ends is not None else ''}")

    filled = np.apply_along_axis(fill_curve, 1, curve)
    fpar = (FPAR_MAX - FPAR_MIN) * (sr(filled) - sr(n02)) / (sr(n98) - sr(n02)) + FPAR_MIN
    fpar = np.clip(fpar, FPAR_MIN, FPAR_MAX)

    w = pd.read_parquet(weather)
    sw = (w.pivot_table(index=["fips", "year"], columns="composite",
                        values="sw_mj_m2")
          .reindex(pd.MultiIndex.from_frame(df[["fips", "year"]])))
    if sw.isna().any().any():
        raise SystemExit(f"{sw.isna().any(axis=1).sum()} county-years lack "
                         f"radiation; rerun 08")
    sw = sw[list(range(1, 15))].to_numpy()

    cols = [c - 1 for c in SEASON]
    apar = fpar[:, cols] * PAR_FRACTION * sw[:, cols]          # MJ/m^2
    apar_total = apar.sum(axis=1)
    biomass = RUE * apar_total                                  # g/m^2
    grain_kg_ha = HI * biomass * 10
    out = df[["fips", "year", "region", base.TARGET]].copy()
    out["fpar_mean"] = fpar[:, cols].mean(axis=1)
    out["par_total"] = (PAR_FRACTION * sw[:, cols]).sum(axis=1)
    out["apar_total"] = apar_total
    out["biomass_t_ha"] = biomass / 100
    out["lue_0"] = grain_kg_ha / KG_HA_DRY_PER_BU_AC
    # Per-composite APAR, kept so a season can be cut short later to
    # ask what the model knew in July.
    for j, c in enumerate(SEASON):
        out[f"apar_{c:02d}"] = apar[:, j]
    return out


def fit_k(p0, y):
    return float(np.sum(p0 * y) / np.sum(p0 ** 2))


def fit_k_trend(p0, y, yr):
    X = np.column_stack([p0, p0 * (yr - CENTRE_YEAR)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def run_scheme(df, inp, scheme):
    y, p0, yr = inp[base.TARGET].to_numpy(), inp.lue_0.to_numpy(), inp.year.to_numpy()
    preds = {m: np.full(len(df), np.nan) for m in ["lue_1", "lue_2", "trend_only"]}
    ks = []
    for _, tr, te in base.folds(df, scheme):
        k = fit_k(p0[tr], y[tr])
        k0, k1 = fit_k_trend(p0[tr], y[tr], yr[tr])
        preds["lue_1"][te] = k * p0[te]
        preds["lue_2"][te] = p0[te] * (k0 + k1 * (yr[te] - CENTRE_YEAR))
        preds["trend_only"][te] = base.trend(df[tr])(yr[te])
        ks.append((k, k0, k1))
    ks = np.array(ks)
    print(f"  {scheme:<7} folds {len(ks):>4}   k {ks[:,0].mean():.3f} "
          f"(range {ks[:,0].min():.3f} to {ks[:,0].max():.3f})   "
          f"lue_2 k change per year {ks[:,2].mean():+.4f}", flush=True)
    return preds


def score_table(long: pd.DataFrame, subset=None) -> pd.DataFrame:
    d = long if subset is None else long[subset(long)]
    rows = []
    for (scheme, model), g in d.groupby(["scheme", "model"], sort=False,
                                        observed=True):
        rows.append({"scheme": scheme, "model": model, "n": len(g),
                     **base.score(g.y_true.to_numpy(), g.y_pred.to_numpy())})
    t = pd.DataFrame(rows)
    trend = t[t.model == "trend_only"].set_index("scheme").rmse
    t["skill_vs_trend"] = 1 - t.rmse / t.scheme.map(trend)
    return t


def main() -> None:
    tee_to("lue")
    df = base.load()
    print(f"rows {len(df):,}  counties {df.fips.nunique()}  "
          f"years {df.year.min()}-{df.year.max()}")
    print(f"constants: RUE {RUE} g/MJ APAR, HI {HI}, PAR {PAR_FRACTION} of "
          f"shortwave, season composites {SEASON.start}-{SEASON.stop - 1}")
    print(f"  1 bu/acre = {KG_HA_DRY_PER_BU_AC:.2f} kg/ha dry grain")

    inp = lue_inputs(df)

    # --- physical sanity, before any fitting ---------------------
    print("\n--- the physics, uncalibrated ---")
    print(f"  season PAR          {inp.par_total.mean():6.0f} MJ/m^2")
    print(f"  mean fAPAR          {inp.fpar_mean.mean():6.3f}")
    print(f"  season APAR         {inp.apar_total.mean():6.0f} MJ/m^2")
    print(f"  above-ground biomass{inp.biomass_t_ha.mean():6.1f} t/ha dry")
    print(f"  lue_0 yield         {inp.lue_0.mean():6.1f} bu/acre   "
          f"(actual {inp[base.TARGET].mean():.1f})")
    print(f"  implied k, all rows {fit_k(inp.lue_0.values, inp[base.TARGET].values):6.3f}"
          f"   (below 1 means the literature constants overpredict)")
    print(f"  corr(lue_0, yield)  {inp.lue_0.corr(inp[base.TARGET]):+6.3f}")

    by_year = inp.groupby("year")[["par_total", "fpar_mean", "apar_total",
                                   "lue_0", base.TARGET]].mean()
    print("\n  by year:")
    print(by_year.round({"par_total": 0, "fpar_mean": 3, "apar_total": 0,
                         "lue_0": 1, base.TARGET: 1}).to_string())

    # --- folds ----------------------------------------------------
    print("\n--- calibrating on training folds ---")
    long = []
    y = inp[base.TARGET].to_numpy()
    for scheme in base.SCHEMES:
        preds = run_scheme(df, inp, scheme)
        preds["lue_0"] = inp.lue_0.to_numpy()
        for m, p in preds.items():
            long.append(pd.DataFrame({"fips": df.fips, "year": df.year,
                                      "scheme": scheme, "model": m,
                                      "y_true": y, "y_pred": p}))
    lue_long = pd.concat(long, ignore_index=True)

    gbm = pd.read_parquet(GBM_PRED)
    gbm = gbm[gbm.target.isin(["raw", "detrended"])].copy()
    gbm["model"] = "gbm_" + gbm.target
    gbm = gbm[["fips", "year", "scheme", "model", "y_true", "y_pred"]]
    allp = pd.concat([gbm, lue_long], ignore_index=True)

    # Identical rows for every model under every scheme, or the
    # comparison is not a comparison.
    counts = allp.groupby(["scheme", "model"]).size().unstack()
    if len(np.unique(counts.to_numpy())) != 1:
        raise SystemExit(f"row counts differ between models:\n{counts}")
    check = allp.pivot_table(index=["fips", "year", "scheme"],
                             columns="model", values="y_true")
    if not np.allclose(check.max(axis=1), check.min(axis=1)):
        raise SystemExit("models disagree on y_true; the rows are misaligned")

    allp["model"] = pd.Categorical(allp.model, MODELS, ordered=True)
    allp["scheme"] = pd.Categorical(allp.scheme, base.SCHEMES, ordered=True)
    allp = allp.sort_values(["scheme", "model"])

    pd.set_option("display.width", 130)
    comp = score_table(allp)
    comp12 = score_table(allp, lambda d: d.year == 2012)
    print("\n--- all years, every model on identical rows and folds ---")
    print(comp.round(3).to_string(index=False))
    print("\n--- 2012 only ---")
    print(comp12.round(3).to_string(index=False))

    # --- the hypothesis -----------------------------------------
    print("\n--- does the physical model degrade less when held out? ---")
    piv = comp.pivot(index="model", columns="scheme", values="rmse")[base.SCHEMES]
    piv["both_vs_random"] = piv["both"] / piv["random"] - 1
    print(piv.round(2).to_string())
    print("  both_vs_random: how much worse the operational case is than "
          "the random split, as a fraction.")

    RESULTS.mkdir(exist_ok=True)
    inp.to_parquet(INPUTS_OUT, index=False)
    lue_long.to_parquet(PRED_OUT, index=False)
    comp.to_csv(RESULTS / "comparison.csv", index=False)
    comp12.to_csv(RESULTS / "comparison_2012.csv", index=False)
    print(f"\nwritten: {INPUTS_OUT}\nwritten: {PRED_OUT}")
    print(f"written: {RESULTS / 'comparison.csv'}, "
          f"{RESULTS / 'comparison_2012.csv'}")


if __name__ == "__main__":
    main()
