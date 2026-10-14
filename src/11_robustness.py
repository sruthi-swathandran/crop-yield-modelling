"""
Stress tests for the findings of 09 and 10, before any of them go in
the README.

    python src/11_robustness.py

Reads   data/model_table.parquet, data/weather_composites.parquet,
        data/predictions_gbm.parquet, data/counties.gpkg
Writes  data/predictions_robust.parquet
        results/robust_rmse.csv          RMSE with 95% intervals
        results/robust_differences.csv   paired differences with intervals
        results/in_season.csv
        results/importance_grouped.csv
        logs/robust.log

FOUR QUESTIONS
--------------
1. Is the district holdout strong enough?
   A district is about ten counties and its neighbours stay in
   training, so "an unseen season costs more than an unseen place"
   might only mean the spatial test leaks. Two harder ones are added:

     region_buffer   hold out a district and train without it or any
                     district touching it
     state           hold out a whole state, eight folds
     both_state      hold out a year and a state together

2. Is the light use efficiency model's stability physics or just
   having few parameters?
   Three controls that know no physics:

     peak_k     yield = k x peak NDVI, one parameter
     peak_lin   yield = a + b x peak NDVI, two parameters
     ridge      linear regression on all 24 features, detrended the
                same way as the gradient boosting model

   If peak_k transfers as well as lue_1 and is as accurate, the
   finding is that simple models transfer, not that physical ones do.

3. Which differences are real?
   Every RMSE gets a 95% interval from a bootstrap that resamples
   whole seasons. Counties in the same year share weather and so do
   not count as independent; resampling seasons respects that.
   Differences between models are bootstrapped in pairs, using the
   same resampled seasons for both, which is what makes a small
   difference detectable when it is real.

   2012 is one season, so a season bootstrap cannot say anything about
   it. Its intervals resample districts within 2012 instead, and so
   describe spread across places in that one year, nothing more.

4. What does the model know before harvest?
   Every scheme so far used that year's corn map, which USDA publishes
   the following February, and NDVI through October. That makes it an
   end-of-season estimate. Here the season is cut off at composites 6,
   8 and 10 (ending about 25 June, 27 July and 28 August) and every
   feature is computed from what existed by then.

   The current-year corn map is still used. Replacing it with the
   previous year's needs the NDVI extraction rerun and is left for
   later; it is stated as a limit rather than hidden.

INTEGRITY CHECK
---------------
The leave-one-year-out gradient boosting predictions are refitted here
and must match the ones 09 saved. If they don't, something in the
shared code has drifted and nothing below is comparable.
"""

import importlib
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from runlog import tee_to

base = importlib.import_module("09_baseline_gbm")
lue = importlib.import_module("10_lue_model")

ROOT = Path(__file__).resolve().parent.parent
COUNTIES = ROOT / "data" / "counties.gpkg"
GBM_PRED = ROOT / "data" / "predictions_gbm.parquet"
PRED_OUT = ROOT / "data" / "predictions_robust.parquet"
RESULTS = ROOT / "results"

T = base.TARGET
B = 2000                 # bootstrap replicates
SEED = 7
SCHEMES = ["random", "year", "region", "region_buffer", "state",
           "both", "both_state"]
NEW_GBM_SCHEMES = ["region_buffer", "state", "both_state"]
CHEAP = ["lue_1", "peak_k", "peak_lin", "ridge", "trend_only"]
CUTOFFS = [6, 8, 10, 14]
CUTOFF_LABEL = {6: "late June", 8: "late July", 10: "late August",
                14: "full season"}

GROUPS = {
    "curve, early (1-5)": [f"ndvi_{i:02d}" for i in range(1, 6)],
    "curve, mid (6-10)": [f"ndvi_{i:02d}" for i in range(6, 11)],
    "curve, late (11-14)": [f"ndvi_{i:02d}" for i in range(11, 15)],
    "canopy size": ["ndvi_peak", "ndvi_mean", "ndvi_integral",
                    "ndvi_amplitude", "composites_above_07"],
    "timing": ["ndvi_peak_composite", "ndvi_start", "ndvi_end"],
    "rates": ["greenup_rate", "senescence_rate"],
}


# ------------------------------------------------------------------
# folds
# ------------------------------------------------------------------

def district_neighbours(df) -> dict:
    c = gpd.read_file(COUNTIES, layer="counties")[["fips", "geometry"]]
    c = c.merge(df[["fips", "region"]].drop_duplicates(), on="fips")
    # A one kilometre buffer so districts that meet across a sliver
    # gap in the boundary file still count as touching.
    d = c.to_crs("EPSG:5070").dissolve("region")
    d = gpd.GeoDataFrame(geometry=d.buffer(1000), crs=d.crs)
    d.index.name = "region"
    j = gpd.sjoin(d, d, predicate="intersects")
    right = [col for col in j.columns if col in ("index_right", "region_right")][0]
    return {r: sorted(set(g[right]) - {r}) for r, g in j.groupby(level=0)}


def folds(df, scheme, nbrs):
    if scheme in base.SCHEMES:
        yield from base.folds(df, scheme)
        return
    year, region = df.year.to_numpy(), df.region.to_numpy()
    state = df.state_alpha.to_numpy()
    if scheme == "region_buffer":
        for r in np.unique(region):
            test = region == r
            excluded = np.isin(region, [r, *nbrs.get(r, [])])
            yield r, ~excluded, test
    elif scheme == "state":
        for s in np.unique(state):
            test = state == s
            yield s, ~test, test
    elif scheme == "both_state":
        for y in np.unique(year):
            for s in np.unique(state):
                test = (year == y) & (state == s)
                if test.any():
                    yield f"{y} {s}", (year != y) & (state != s), test


# ------------------------------------------------------------------
# models
# ------------------------------------------------------------------

def gbm(train, test, feats):
    line = base.trend(train)
    m = LGBMRegressor(**base.PARAMS)
    m.fit(train[feats], train[T] - line(train.year))
    return m.predict(test[feats]) + line(test.year), m


def ridge(train, test, feats):
    line = base.trend(train)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      Ridge(alpha=1.0))
    m.fit(train[feats], train[T] - line(train.year))
    return m.predict(test[feats]) + line(test.year)


def through_origin(x_tr, y_tr, x_te):
    k = np.sum(x_tr * y_tr) / np.sum(x_tr ** 2)
    return k * x_te


def linear(x_tr, y_tr, x_te):
    b, a = np.polyfit(x_tr, y_tr, 1)
    return a + b * x_te


def cheap_predictions(train, test, model, cols):
    """cols names the columns each simple model uses, so in-season
    versions can swap in truncated ones."""
    y = train[T].to_numpy()
    if model == "lue_1":
        return through_origin(train[cols["lue"]].to_numpy(), y,
                              test[cols["lue"]].to_numpy())
    if model == "peak_k":
        return through_origin(train[cols["peak"]].to_numpy(), y,
                              test[cols["peak"]].to_numpy())
    if model == "peak_lin":
        return linear(train[cols["peak"]].to_numpy(), y,
                      test[cols["peak"]].to_numpy())
    if model == "ridge":
        return ridge(train, test, cols["features"])
    if model == "trend_only":
        return base.trend(train)(test.year)
    raise ValueError(model)


def run(df, scheme, models, nbrs, cols, label=""):
    fl = list(folds(df, scheme, nbrs))
    preds = {m: np.full(len(df), np.nan) for m in models}
    t0, n = time.time(), len(fl)
    every = max(1, n // 10)
    for i, (_, tr, te) in enumerate(fl, 1):
        train, test = df[tr], df[te]
        for m in models:
            if m == "gbm_detrended":
                preds[m][te] = gbm(train, test, cols["features"])[0]
            else:
                preds[m][te] = cheap_predictions(train, test, m, cols)
        if "gbm_detrended" in models and (i % every == 0 or i == n):
            el = time.time() - t0
            print(f"    {label}{scheme:<14} {i:>4}/{n}   {el/60:4.1f} min, "
                  f"~{(n - i) * el / i / 60:4.1f} min left", flush=True)
    for m, p in preds.items():
        if np.isnan(p).any():
            raise SystemExit(f"{scheme}/{m}: {np.isnan(p).sum()} unpredicted")
    return preds


def to_long(df, scheme, preds, extra=None):
    out = []
    for m, p in preds.items():
        d = pd.DataFrame({"fips": df.fips.values, "year": df.year.values,
                          "region": df.region.values, "scheme": scheme,
                          "model": m, "y_true": df[T].values, "y_pred": p})
        if extra:
            d = d.assign(**extra)
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------

def boot_counts(n_groups, rng):
    return np.stack([np.bincount(rng.integers(0, n_groups, n_groups),
                                 minlength=n_groups) for _ in range(B)])


def boot_rmse(long, by, counts, keys):
    """Bootstrap RMSE for every (scheme, model), resampling the groups
    in `by`. The same counts matrix is used for every model, so the
    replicates are paired."""
    long = long.assign(se=(long.y_pred - long.y_true) ** 2)
    agg = long.groupby(["scheme", "model", by]).se.agg(["sum", "count"])
    out = {}
    for (s, m), g in agg.groupby(level=[0, 1]):
        g = g.droplevel([0, 1]).reindex(keys, fill_value=0)
        sse, n = g["sum"].to_numpy(), g["count"].to_numpy()
        out[(s, m)] = np.sqrt((counts @ sse) / np.maximum(counts @ n, 1))
    return out


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return lo, hi


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    tee_to("robust")
    t_start = time.time()
    df = base.load()
    inp = lue.lue_inputs(df)
    df = pd.concat([df, inp[["lue_0"] + [c for c in inp.columns
                                         if c.startswith("apar_")]]], axis=1)
    print(f"rows {len(df):,}  counties {df.fips.nunique()}  "
          f"districts {df.region.nunique()}  states {df.state_alpha.nunique()}")

    nbrs = district_neighbours(df)
    nn = pd.Series({r: len(v) for r, v in nbrs.items()})
    print(f"district neighbours: median {nn.median():.0f}, "
          f"range {nn.min()} to {nn.max()}")
    missing = set(df.region) - set(nbrs)
    if missing:
        print(f"  districts with no geometry match: {sorted(missing)}")

    full_cols = {"lue": "lue_0", "peak": "ndvi_peak", "features": base.FEATURES}

    # --- integrity check -----------------------------------------
    print("\n--- integrity: refit leave-one-year-out, compare with 09 ---")
    saved = pd.read_parquet(GBM_PRED)
    saved = saved[(saved.scheme == "year") & (saved.target == "detrended")]
    redo = run(df, "year", ["gbm_detrended"], nbrs, full_cols)["gbm_detrended"]
    diff = np.abs(saved.y_pred.to_numpy() - redo).max()
    print(f"  largest difference from saved predictions: {diff:.2e} bu/acre")
    if diff > 1e-6:
        raise SystemExit("refit does not reproduce 09; stop and investigate")

    # --- questions 1 and 2: every scheme, every model ------------
    print("\n--- every scheme, every model ---")
    long = []
    g = pd.read_parquet(GBM_PRED)
    g = g[g.target.isin(["raw", "detrended"])].assign(model=lambda d: "gbm_" + d.target)
    long.append(g[["fips", "year", "region", "scheme", "model", "y_true", "y_pred"]])
    for scheme in SCHEMES:
        models = CHEAP + (["gbm_detrended"] if scheme in NEW_GBM_SCHEMES else [])
        print(f"  {scheme}", flush=True)
        long.append(to_long(df, scheme, run(df, scheme, models, nbrs, full_cols)))
    long = pd.concat(long, ignore_index=True)

    # --- question 3: intervals ------------------------------------
    rng = np.random.default_rng(SEED)
    years = sorted(df.year.unique())
    yc = boot_counts(len(years), rng)
    boots = boot_rmse(long, "year", yc, years)

    rows = []
    for (s, m), b in boots.items():
        g = long[(long.scheme == s) & (long.model == m)]
        point = float(np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)))
        lo, hi = ci(b)
        rows.append({"scheme": s, "model": m, "rmse": point,
                     "lo95": lo, "hi95": hi})
    rmse = pd.DataFrame(rows)
    order_m = ["gbm_detrended", "gbm_raw", "ridge", "lue_1", "peak_k",
               "peak_lin", "trend_only"]
    rmse["scheme"] = pd.Categorical(rmse.scheme, SCHEMES, ordered=True)
    rmse["model"] = pd.Categorical(rmse.model, order_m, ordered=True)
    rmse = rmse.sort_values(["model", "scheme"])

    pd.set_option("display.width", 160)
    cell = rmse.assign(txt=lambda d: d.apply(
        lambda r: f"{r.rmse:5.1f} [{r.lo95:4.1f},{r.hi95:5.1f}]", axis=1))
    print("\n--- RMSE, bu/acre, with 95% season-bootstrap interval ---")
    print(cell.pivot(index="model", columns="scheme", values="txt")
          .reindex(columns=SCHEMES).fillna("").to_string())

    # Paired differences. Negative means the first is better.
    def paired(s1, m1, s2, m2, what):
        a, b = boots[(s1, m1)], boots[(s2, m2)]
        d = a - b
        lo, hi = ci(d)
        p1 = rmse[(rmse.scheme == s1) & (rmse.model == m1)].rmse.iloc[0]
        p2 = rmse[(rmse.scheme == s2) & (rmse.model == m2)].rmse.iloc[0]
        return {"comparison": what, "first": f"{m1} / {s1}",
                "second": f"{m2} / {s2}", "difference": p1 - p2,
                "lo95": lo, "hi95": hi,
                "clear": "yes" if (lo > 0 or hi < 0) else "no"}

    comps = []
    for s in ["random", "year", "region", "both"]:
        comps.append(paired(s, "gbm_detrended", s, "gbm_raw",
                            "detrended minus raw"))
    for s in SCHEMES:
        comps.append(paired(s, "gbm_detrended", s, "lue_1",
                            "boosting minus LUE"))
        comps.append(paired(s, "lue_1", s, "peak_k",
                            "LUE minus k x peak NDVI"))
        comps.append(paired(s, "gbm_detrended", s, "ridge",
                            "boosting minus ridge"))
    for m in ["gbm_detrended", "lue_1", "peak_k", "ridge"]:
        comps.append(paired("both_state", m, "random", m,
                            "cost of the operational holdout"))
    comps.append(paired("region_buffer", "gbm_detrended", "region",
                        "gbm_detrended", "buffered minus plain district"))
    comps.append(paired("state", "gbm_detrended", "region", "gbm_detrended",
                        "state minus district"))
    comps.append(paired("year", "gbm_detrended", "state", "gbm_detrended",
                        "unseen year minus unseen state"))
    comps.append(paired("year", "gbm_detrended", "region", "gbm_detrended",
                        "unseen year minus unseen district"))
    diffs = pd.DataFrame(comps)
    print("\n--- paired differences in RMSE, with 95% interval ---")
    print("  negative means the first is better; clear means the interval "
          "excludes zero")
    print(diffs.round(2).to_string(index=False))

    # How much each model degrades, as a ratio, with interval.
    print("\n--- degradation: RMSE(held out) / RMSE(random) ---")
    for m in ["gbm_detrended", "ridge", "lue_1", "peak_k", "peak_lin", "trend_only"]:
        parts = []
        for s in ["both", "both_state"]:
            r = boots[(s, m)] / boots[("random", m)]
            pt = (rmse[(rmse.scheme == s) & (rmse.model == m)].rmse.iloc[0]
                  / rmse[(rmse.scheme == "random") & (rmse.model == m)].rmse.iloc[0])
            lo, hi = ci(r)
            parts.append(f"{s} {pt:.2f} [{lo:.2f},{hi:.2f}]")
        print(f"  {m:<14} " + "   ".join(parts))

    # 2012, district bootstrap within the season.
    print("\n--- 2012, intervals from resampling districts within 2012 ---")
    l12 = long[long.year == 2012]
    regions = sorted(l12.region.unique())
    rc = boot_counts(len(regions), np.random.default_rng(SEED + 1))
    b12 = boot_rmse(l12, "region", rc, regions)
    rows12 = []
    for s in ["year", "both", "both_state"]:
        for m in ["gbm_detrended", "gbm_raw", "lue_1", "peak_k", "ridge"]:
            if (s, m) not in b12:
                continue
            g = l12[(l12.scheme == s) & (l12.model == m)]
            e = g.y_pred - g.y_true
            lo, hi = ci(b12[(s, m)])
            rows12.append({"scheme": s, "model": m,
                           "rmse": float(np.sqrt(np.mean(e ** 2))),
                           "lo95": lo, "hi95": hi, "bias": float(e.mean())})
    r12 = pd.DataFrame(rows12)
    # Bias relative to each model's own average bias across all years,
    # so a model that runs low everywhere is not credited for it in 2012.
    own = (long.assign(e=long.y_pred - long.y_true)
           .groupby(["scheme", "model"]).e.mean())
    r12["bias_shift"] = r12.bias - [own[(s, m)] for s, m in zip(r12.scheme, r12.model)]
    print(r12.round(1).to_string(index=False))

    # --- question 4: in season -----------------------------------
    print("\n--- in season: what was known by each cutoff ---")
    ins = []
    for c in CUTOFFS:
        curve = [f"ndvi_{i:02d}" for i in range(1, c + 1)]
        v = df[curve].to_numpy(dtype="float64")
        d = df.copy()
        d["s_peak"] = np.nanmax(v, axis=1)
        d["s_mean"] = np.nanmean(v, axis=1)
        d["s_peak_at"] = np.nanargmax(np.where(np.isfinite(v), v, -1), axis=1) + 1
        d["s_last"] = v[:, -1]
        d["lue_cut"] = d[[f"apar_{k:02d}" for k in range(4, min(c, 12) + 1)]
                         ].sum(axis=1) * lue.HI * lue.RUE * 10 / lue.KG_HA_DRY_PER_BU_AC
        feats = curve + ["s_peak", "s_mean", "s_peak_at", "s_last"]
        cols = {"lue": "lue_cut", "peak": "s_peak", "features": feats}
        for s in ["year", "both_state"]:
            preds = run(d, s, ["gbm_detrended", "lue_1", "peak_k", "trend_only"],
                        nbrs, cols, label=f"cutoff {c:>2} ")
            for m, p in preds.items():
                ins.append({"cutoff": c, "known_by": CUTOFF_LABEL[c],
                            "scheme": s, "model": m,
                            "rmse": float(np.sqrt(np.mean((p - d[T]) ** 2)))})
    ins = pd.DataFrame(ins)
    print(ins.pivot_table(index=["scheme", "model"], columns="cutoff",
                          values="rmse").round(1).to_string())
    print("  cutoff 14 here uses only the curve and four simple summaries,")
    print("  so it differs slightly from the full-feature model above.")
    print("  lue_1 cut short measures light absorbed so far, not light the")
    print("  crop will absorb, so early cutoffs are not a fair test of it.")
    print("  A proper in-season LUE would add expected radiation for the")
    print("  rest of the season; that is not attempted here.")

    # --- grouped permutation importance --------------------------
    print("\n--- grouped permutation importance, leave one year out ---")
    print("  rise in RMSE when a group of features is shuffled on the "
          "held-out year")
    rng_p = np.random.default_rng(SEED + 2)
    rise = {k: [] for k in GROUPS}
    for _, tr, te in base.folds(df, "year"):
        train, test = df[tr], df[te]
        p, m = gbm(train, test, base.FEATURES)
        line = base.trend(train)
        base_rmse = np.sqrt(np.mean((p - test[T]) ** 2))
        for gname, cols_g in GROUPS.items():
            X = test[base.FEATURES].copy()
            perm = rng_p.permutation(len(X))
            X[cols_g] = X[cols_g].to_numpy()[perm]     # shuffle the group together
            pp = m.predict(X) + line(test.year)
            rise[gname].append(np.sqrt(np.mean((pp - test[T]) ** 2)) - base_rmse)
    imp = (pd.DataFrame({k: np.mean(v) for k, v in rise.items()}, index=["rmse_rise"])
           .T.sort_values("rmse_rise", ascending=False))
    print(imp.round(2).to_string())

    RESULTS.mkdir(exist_ok=True)
    long.to_parquet(PRED_OUT, index=False)
    rmse.to_csv(RESULTS / "robust_rmse.csv", index=False)
    diffs.to_csv(RESULTS / "robust_differences.csv", index=False)
    r12.to_csv(RESULTS / "robust_2012.csv", index=False)
    ins.to_csv(RESULTS / "in_season.csv", index=False)
    imp.to_csv(RESULTS / "importance_grouped.csv")
    print(f"\nwritten: {PRED_OUT} and six tables in {RESULTS}")
    print(f"total {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
