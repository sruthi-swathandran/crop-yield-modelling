"""
Gradient boosting baseline, evaluated four ways.

    python src/09_baseline_gbm.py

Reads   data/model_table.parquet
Writes  data/predictions_gbm.parquet   every out-of-fold prediction
        results/gbm_scores.csv         one row per scheme and target
        results/gbm_2012.csv           the drought year on its own
        logs/gbm.log

THE FOUR SCHEMES
----------------
random   five folds over rows, shuffled. What most papers report.
year     leave one year out, fourteen folds. An unseen season.
region   leave one agricultural statistics district out. An unseen place.
both     for every year and district, train on rows from neither and
         predict the rows from both. An unseen season in an unseen
         place, which is the situation an operational user is in.

Every row is predicted exactly once under every scheme, so the four
sets of scores cover identical rows and differ only in what the model
was allowed to see.

Districts are numbered within each state, so district 10 in Iowa and
district 10 in Illinois are different places. The region key is state
plus district.

THE TREND
---------
Yields rise about three bushels per acre per year through breeding and
management. Two targets are compared:

raw         the model predicts yield directly from the NDVI features.
detrended   a straight line in year is fitted to the training rows only,
            the model predicts the departure from it, and the line is
            added back. Year is known when a season is being predicted,
            so this uses nothing the forecaster would not have.

The trend line is refitted inside every fold. Fitting it once on all
rows would let held-out years shape the line used to predict them.

THE BAR TO CLEAR
----------------
A model that cannot beat the trend line alone has learned nothing from
the satellite. So the trend line's own error is computed on the same
folds, and skill is reported against it:

    skill = 1 - RMSE(model) / RMSE(trend only)

Zero means the satellite added nothing. Negative means it made things
worse.

NO TUNING
---------
Hyperparameters are fixed and identical across schemes. Tuning them on
random folds would favour the random scheme, and tuning them properly
for each scheme needs a second layer of cross-validation inside every
fold. Moderate fixed settings keep the comparison about the evaluation
rather than about how much search each scheme received.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "model_table.parquet"
PRED_OUT = ROOT / "data" / "predictions_gbm.parquet"
RESULTS = ROOT / "results"

SEED = 42
TARGET = "yield_bu_acre"
SCHEMES = ["random", "year", "region", "both"]
MODES = ["raw", "detrended"]

CURVE = [f"ndvi_{i:02d}" for i in range(1, 15)]
PHENOLOGY = [
    "ndvi_peak", "ndvi_peak_composite", "ndvi_mean", "ndvi_integral",
    "ndvi_start", "ndvi_end", "ndvi_amplitude",
    "greenup_rate", "senescence_rate", "composites_above_07",
]
FEATURES = CURVE + PHENOLOGY

PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, verbose=-1,
)


def load() -> pd.DataFrame:
    df = pd.read_parquet(TABLE)
    df = df.rename(columns={"composites_above_0.7": "composites_above_07"})
    df["region"] = df.state_alpha + "-" + df.asd.astype(str)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"model table lacks {missing}; rerun 07")
    return df


def folds(df: pd.DataFrame, scheme: str):
    """Yield (label, train mask, test mask) as boolean arrays."""
    year, region = df.year.to_numpy(), df.region.to_numpy()
    if scheme == "random":
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (_, te) in enumerate(kf.split(df), 1):
            test = np.zeros(len(df), bool)
            test[te] = True
            yield f"fold {k}", ~test, test
    elif scheme == "year":
        for y in np.unique(year):
            test = year == y
            yield str(y), ~test, test
    elif scheme == "region":
        for r in np.unique(region):
            test = region == r
            yield r, ~test, test
    elif scheme == "both":
        for y in np.unique(year):
            for r in np.unique(region):
                test = (year == y) & (region == r)
                if test.any():
                    yield f"{y} {r}", (year != y) & (region != r), test


def trend(train: pd.DataFrame):
    slope, intercept = np.polyfit(train.year, train[TARGET], 1)
    return lambda yrs: intercept + slope * np.asarray(yrs, dtype=float)


def fit_predict(train, test, mode, line):
    if mode == "detrended":
        y_train = train[TARGET] - line(train.year)
        offset = line(test.year)
    else:
        y_train = train[TARGET]
        offset = 0.0
    m = LGBMRegressor(**PARAMS)
    m.fit(train[FEATURES], y_train)
    return m.predict(test[FEATURES]) + offset


def score(y, p) -> dict:
    e = p - y
    sse = float(np.sum(e ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "r2": 1 - sse / float(np.sum((y - y.mean()) ** 2)),
    }


def run_scheme(df, scheme):
    fold_list = list(folds(df, scheme))
    n = len(fold_list)
    preds = {m: np.full(len(df), np.nan) for m in MODES + ["trend_only"]}
    every = 1 if n <= 80 else 25
    print(f"\n=== {scheme}: {n} folds ===", flush=True)
    t0 = time.time()

    for i, (label, tr, te) in enumerate(fold_list, 1):
        train, test = df[tr], df[te]
        line = trend(train)
        preds["trend_only"][te] = line(test.year)
        for mode in MODES:
            preds[mode][te] = fit_predict(train, test, mode, line)
        if i % every == 0 or i == n:
            el = time.time() - t0
            left = (n - i) * el / i / 60
            print(f"  {i:>4}/{n}  {label:<12} train {tr.sum():>5}  "
                  f"test {te.sum():>4}   {el/60:5.1f} min elapsed, "
                  f"~{left:5.1f} min left", flush=True)

    for m, p in preds.items():
        if np.isnan(p).any():
            raise SystemExit(f"{scheme}/{m}: {np.isnan(p).sum()} rows unpredicted")
    return preds


def main() -> None:
    tee_to("gbm")
    df = load()
    print(f"rows {len(df):,}  counties {df.fips.nunique()}  "
          f"regions {df.region.nunique()}  years {df.year.min()}-{df.year.max()}")
    print(f"features: {len(FEATURES)}  ({len(CURVE)} curve, "
          f"{len(PHENOLOGY)} phenology)")

    y = df[TARGET].to_numpy()
    is2012 = (df.year == 2012).to_numpy()
    long, rows, rows12 = [], [], []

    for scheme in SCHEMES:
        preds = run_scheme(df, scheme)
        base = score(y, preds["trend_only"])["rmse"]
        base12 = score(y[is2012], preds["trend_only"][is2012])["rmse"]
        for m, p in preds.items():
            s = score(y, p)
            s12 = score(y[is2012], p[is2012])
            rows.append({"scheme": scheme, "target": m, **s,
                         "skill_vs_trend": 1 - s["rmse"] / base})
            rows12.append({"scheme": scheme, "target": m,
                           "rmse_2012": s12["rmse"], "bias_2012": s12["bias"],
                           "skill_vs_trend_2012": 1 - s12["rmse"] / base12})
            long.append(pd.DataFrame({
                "fips": df.fips, "year": df.year, "region": df.region,
                "scheme": scheme, "target": m, "y_true": y, "y_pred": p,
            }))

    scores = pd.DataFrame(rows)
    s2012 = pd.DataFrame(rows12)

    pd.set_option("display.width", 120)
    print("\n--- all years, out of fold ---")
    print(scores.round(3).to_string(index=False))
    print("\n--- 2012 only ---")
    print(s2012.round(3).to_string(index=False))

    # The headline: how much does the random split flatter the model?
    for mode in MODES:
        r = scores[(scores.scheme == "random") & (scores.target == mode)].rmse.iloc[0]
        b = scores[(scores.scheme == "both") & (scores.target == mode)].rmse.iloc[0]
        print(f"\n  {mode}: RMSE random {r:.1f}, both {b:.1f} bu/acre. "
              f"The random split reports {100*(1 - r/b):.0f}% less error "
              f"than the operational case.")

    # Per-year error under leave-one-year-out, to see which seasons
    # are hard rather than averaging them away.
    yr = (pd.concat(long).query("scheme == 'year' and target == 'detrended'")
          .assign(se=lambda d: (d.y_pred - d.y_true) ** 2,
                  err=lambda d: d.y_pred - d.y_true)
          .groupby("year").agg(rmse=("se", lambda s: np.sqrt(s.mean())),
                               bias=("err", "mean")))
    print("\n--- leave one year out, detrended, by year ---")
    print(yr.round(1).to_string())

    # What the model leans on. Fitted on all rows, so this describes the
    # model rather than evaluating it.
    full = LGBMRegressor(**{**PARAMS, "importance_type": "gain"})
    line = trend(df)
    full.fit(df[FEATURES], df[TARGET] - line(df.year))
    imp = (pd.Series(full.feature_importances_, index=FEATURES)
           .sort_values(ascending=False))
    imp = 100 * imp / imp.sum()
    print("\n--- share of gain, detrended model fitted on all rows ---")
    print(imp.head(12).round(1).to_string())

    RESULTS.mkdir(exist_ok=True)
    pd.concat(long, ignore_index=True).to_parquet(PRED_OUT, index=False)
    scores.to_csv(RESULTS / "gbm_scores.csv", index=False)
    s2012.to_csv(RESULTS / "gbm_2012.csv", index=False)
    print(f"\nwritten: {PRED_OUT}")
    print(f"written: {RESULTS / 'gbm_scores.csv'}, {RESULTS / 'gbm_2012.csv'}")


if __name__ == "__main__":
    main()
