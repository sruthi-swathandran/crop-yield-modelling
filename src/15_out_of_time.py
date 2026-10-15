"""
Train on 2008 to 2021, predict 2022 to 2024.

    python src/07_features.py --years 2022-2024 \
        --yields data/nass_corn_yields_recent.csv --out data/model_table_recent.parquet
    python src/15_out_of_time.py

Reads   data/model_table.parquet, data/model_table_recent.parquet,
        data/weather_composites.parquet, data/weather_composites_recent.parquet,
        results/robust_rmse.csv
Writes  data/predictions_future.parquet
        results/future_scores.csv, results/future_by_year.csv,
        results/future_differences.csv, logs/future.log

WHAT THIS TESTS
---------------
Every earlier score held out seasons from inside 2008 to 2021, so the
trend line was always interpolated between years on either side. Here
the model sees nothing after 2021 and has to extrapolate. That is the
situation of anyone using a yield model.

Two schemes:

future         train on every county 2008 to 2021, predict every
               county 2022 to 2024
future_state   for each state, train on the other seven states
               2008 to 2021, predict that state 2022 to 2024

THE QUESTION THE EARLIER RESULTS SET UP
---------------------------------------
Leave one year out put gradient boosting at 19.1 bushels, and a year
and a state held out at 22.0. If those held-out-year scores were honest
estimates, the future scores should land near them. If they come out
clearly worse, something about the future is harder than any held-out
year inside the record was, and the candidates are named in advance:
the trend extrapolated past its data, Terra's orbit drift (the sun
zenith angle at overpass rose from about 29 degrees to 32 in 2023 and
35 in 2024), and the shrinking set of counties NASS still reports.

UNCERTAINTY WITH THREE SEASONS
------------------------------
Three test seasons are too few to resample. Intervals here resample
districts within each season, so they describe spread across places,
not uncertainty about future seasons in general. Each year is also
reported separately, and that is where the real uncertainty shows.

CONSISTENCY WITH TRAINING
-------------------------
fAPAR for the test years is scaled with the NDVI end members of the
training years. Recomputing them on 2022 to 2024 would let the test data
reset the scale, and drift in NDVI would vanish into it.
"""

import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from runlog import tee_to

base = importlib.import_module("09_baseline_gbm")
lue = importlib.import_module("10_lue_model")
rob = importlib.import_module("11_robustness")

ROOT = Path(__file__).resolve().parent.parent
RECENT = ROOT / "data" / "model_table_recent.parquet"
W_RECENT = ROOT / "data" / "weather_composites_recent.parquet"
PRED_OUT = ROOT / "data" / "predictions_future.parquet"
RESULTS = ROOT / "results"
T, F = base.TARGET, base.FEATURES
SEED = 23
B = 2000

MODELS = ["gbm_detrended", "gbm_raw", "ridge", "lue_1", "peak_k",
          "peak_lin", "trend_only"]


def load_recent() -> pd.DataFrame:
    df = pd.read_parquet(RECENT)
    df = df.rename(columns={"composites_above_0.7": "composites_above_07"})
    df["region"] = df.state_alpha + "-" + df.asd.astype(str)
    return df.dropna(subset=[T]).reset_index(drop=True)


def gbm_raw(train, test):
    m = LGBMRegressor(**base.PARAMS)
    m.fit(train[F], train[T])
    return m.predict(test[F])


def predict_all(train, test):
    cols = {"lue": "lue_0", "peak": "ndvi_peak", "features": F}
    out = {}
    out["gbm_detrended"] = rob.gbm(train, test, F)[0]
    out["gbm_raw"] = gbm_raw(train, test)
    for m in ["ridge", "lue_1", "peak_k", "peak_lin", "trend_only"]:
        out[m] = np.asarray(rob.cheap_predictions(train, test, m, cols))
    return out


def district_boot(long, rng):
    """RMSE intervals per (scheme, model), resampling districts within
    each year so every year keeps its weight."""
    long = long.assign(se=(long.y_pred - long.y_true) ** 2)
    agg = long.groupby(["scheme", "model", "year", "region"]).se.agg(["sum", "count"])
    keys = agg.droplevel([0, 1]).index.unique()
    by_year = {}
    for y, r in keys:
        by_year.setdefault(y, []).append(r)
    # one set of district weights per replicate, shared by every model
    weights = {}
    for y, regs in by_year.items():
        k = len(regs)
        c = np.stack([np.bincount(rng.integers(0, k, k), minlength=k)
                      for _ in range(B)])
        for j, r in enumerate(regs):
            weights[(y, r)] = c[:, j]
    out = {}
    for (s, m), g in agg.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        sse = np.zeros(B)
        n = np.zeros(B)
        for (y, r), row in g.iterrows():
            w = weights[(y, r)]
            sse += w * row["sum"]
            n += w * row["count"]
        out[(s, m)] = np.sqrt(sse / np.maximum(n, 1))
    return out


def main() -> None:
    tee_to("future")
    t0 = time.time()
    train = base.load()
    test = load_recent()
    print(f"train: {len(train):,} county-years, {train.fips.nunique()} counties, "
          f"{train.year.min()}-{train.year.max()}")
    print(f"test : {len(test):,} county-years, {test.fips.nunique()} counties, "
          f"{test.year.min()}-{test.year.max()}")
    print("  test county-years per year: "
          + ", ".join(f"{y} {n}" for y, n in test.groupby("year").size().items()))
    unseen = sorted(set(test.fips) - set(train.fips))
    print(f"  test counties never in training: {len(unseen)}")

    # --- light use efficiency inputs, scaled on the training years ----
    ends = lue.end_members(train)
    li_train = lue.lue_inputs(train)
    li_test = lue.lue_inputs(test, weather=W_RECENT, ends=ends)
    train = train.assign(lue_0=li_train.lue_0.to_numpy())
    test = test.assign(lue_0=li_test.lue_0.to_numpy(),
                       apar_total=li_test.apar_total.to_numpy(),
                       fpar_mean=li_test.fpar_mean.to_numpy())

    # --- what changed between the periods -------------------------
    print("\n--- the test years against the training years ---")
    tr_by = train.groupby("year")[[T, "ndvi_peak"]].mean()
    te_by = test.groupby("year")[[T, "ndvi_peak", "fpar_mean", "apar_total"]].mean()
    line = base.trend(train)
    te_by["trend_says"] = line(te_by.index.to_numpy())
    print(f"  training years: mean yield {train[T].mean():.1f}, "
          f"mean peak NDVI {train.ndvi_peak.mean():.3f}, "
          f"2019-2021 peak NDVI {tr_by.loc[2019:2021, 'ndvi_peak'].mean():.3f}")
    print(te_by.round(3).to_string())

    # --- predictions ---------------------------------------------
    long = []
    print("\n--- fitting ---")
    preds = predict_all(train, test)
    for m, p in preds.items():
        long.append(test[["fips", "year", "region", "state_alpha"]]
                    .assign(scheme="future", model=m, y_true=test[T].to_numpy(), y_pred=p))
    print("  future: done", flush=True)
    for s in sorted(test.state_alpha.unique()):
        tr = train[train.state_alpha != s]
        te = test[test.state_alpha == s]
        ps = predict_all(tr, te)
        for m, p in ps.items():
            long.append(te[["fips", "year", "region", "state_alpha"]]
                        .assign(scheme="future_state", model=m,
                                y_true=te[T].to_numpy(), y_pred=p))
        print(f"  future_state: {s} done ({len(te)} county-years)", flush=True)
    long = pd.concat(long, ignore_index=True)

    # --- scores --------------------------------------------------
    boots = district_boot(long, np.random.default_rng(SEED))
    rows = []
    for (s, m), b in boots.items():
        g = long[(long.scheme == s) & (long.model == m)]
        e = g.y_pred - g.y_true
        lo, hi = rob.ci(b)
        rows.append({"scheme": s, "model": m, "n": len(g),
                     "rmse": float(np.sqrt(np.mean(e ** 2))), "lo95": lo, "hi95": hi,
                     "bias": float(e.mean())})
    scores = pd.DataFrame(rows)
    scores["model"] = pd.Categorical(scores.model, MODELS, ordered=True)
    scores = scores.sort_values(["scheme", "model"])
    pd.set_option("display.width", 150)
    print("\n--- RMSE on 2022-2024, intervals resample districts within each year ---")
    print(scores.round(2).to_string(index=False))

    byy = (long.assign(e=long.y_pred - long.y_true)
           .groupby(["scheme", "model", "year"])
           .e.agg(rmse=lambda e: float(np.sqrt(np.mean(e ** 2))), bias="mean")
           .reset_index())
    print("\n--- by year ---")
    print(byy.pivot_table(index=["scheme", "model"], columns="year",
                          values=["rmse", "bias"]).round(1).to_string())

    # --- against what the held-out-year scores promised -------------
    ref = pd.read_csv(RESULTS / "robust_rmse.csv")
    ref = ref[ref.model == "gbm_detrended"].set_index("scheme")
    fut = scores.set_index(["scheme", "model"]).rmse
    print("\n--- did the held-out-year estimates hold? gradient boosting ---")
    for s_ref, s_fut, label in [("year", "future", "year held out"),
                                ("both_state", "future_state", "year and state held out")]:
        r = ref.loc[s_ref]
        f = fut[(s_fut, "gbm_detrended")]
        print(f"  {label:<26} estimate {r.rmse:5.1f} [{r.lo95:.1f}, {r.hi95:.1f}]   "
              f"2022-2024 actual {f:5.1f}   ratio {f / r.rmse:.2f}")

    # --- paired differences --------------------------------------
    diffs = []
    for s in ["future", "future_state"]:
        for a, b in [("gbm_detrended", "ridge"), ("gbm_detrended", "lue_1"),
                     ("gbm_detrended", "gbm_raw"), ("lue_1", "peak_k")]:
            d = boots[(s, a)] - boots[(s, b)]
            lo, hi = rob.ci(d)
            pa, pb = fut[(s, a)], fut[(s, b)]
            diffs.append({"scheme": s, "comparison": f"{a} minus {b}",
                          "difference": pa - pb, "lo95": lo, "hi95": hi,
                          "clear": "yes" if (lo > 0 or hi < 0) else "no"})
    diffs = pd.DataFrame(diffs)
    print("\n--- paired differences, negative means the first is better ---")
    print(diffs.round(2).to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    long.to_parquet(PRED_OUT, index=False)
    scores.to_csv(RESULTS / "future_scores.csv", index=False)
    byy.to_csv(RESULTS / "future_by_year.csv", index=False)
    diffs.to_csv(RESULTS / "future_differences.csv", index=False)
    print(f"\nwritten: {PRED_OUT} and three tables in {RESULTS}")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
