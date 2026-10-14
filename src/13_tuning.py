"""
Does tuning help, and does the usual way of tuning mislead?

    python src/13_tuning.py

Reads   data/model_table.parquet, data/counties.gpkg (through 11)
Writes  results/tuning_scores.csv, results/tuning_differences.csv,
        results/tuning_choices.csv, logs/tuning.log

THREE VERSIONS OF GRADIENT BOOSTING
-----------------------------------
fixed           the settings used everywhere else, never tuned
tuned_random    tuned by random search, scored on inner folds drawn at
                random from the training rows. The usual way.
tuned_blocked   the same search, scored on inner folds that hold out a
                block of years and a block of states together, copying
                the outer test.

Both tuned versions try the same sixteen settings, drawn once with a
fixed seed, so they differ only in how a setting is judged.

NESTED, SO THE TEST NEVER CHOOSES
---------------------------------
Inside every outer fold the search sees only that fold's training rows.
The winner is refitted on all of them and then scored once on the
outer test rows, which played no part in choosing it.

TWO OUTER TESTS
---------------
random        five random folds, as most papers report
year_state    32 folds. Years are dealt into four interleaved blocks
              (2008, 2012, 2016, 2020 in one) and each fold holds out
              one state within one block of years, training on the
              other states in the other years. Every row is tested
              once. A coarser version of 11's year-and-state scheme,
              which at 112 folds would need about 25,000 fits here.

THE PREDICTION
--------------
Random inner folds reward settings that recall neighbouring counties,
which means many small leaves. So tuned_random should beat fixed on the
random outer test and do worse, or no better, on year_state, while
tuned_blocked should choose simpler trees. If that happens, the usual
way of tuning makes a model look better and work worse. If it does not,
that is reported too.

Expect about an hour: 37 outer folds, each with one fixed fit, 96
inner fits and two refits.
"""

import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from runlog import tee_to

base = importlib.import_module("09_baseline_gbm")
rob = importlib.import_module("11_robustness")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
T, F = base.TARGET, base.FEATURES
SEED = 11
N_CONFIGS = 16
N_INNER = 3
N_YEAR_BLOCKS = 4

FIXED = {k: base.PARAMS[k] for k in
         ["n_estimators", "learning_rate", "num_leaves", "min_child_samples",
          "subsample", "colsample_bytree"]}
FIXED["reg_lambda"] = 0.0
ALWAYS = dict(subsample_freq=1, random_state=base.SEED, verbose=-1,
              deterministic=True, force_row_wise=True, num_threads=4)

SPACE = {
    "num_leaves": [7, 15, 31, 63, 127],
    "min_child_samples": [5, 10, 20, 50, 100],
    "learning_rate": [0.01, 0.03, 0.1],
    "n_estimators": [200, 500, 1000],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0],
}


def draw_configs():
    rng = np.random.default_rng(SEED)
    return [{k: v[rng.integers(len(v))] for k, v in SPACE.items()}
            for _ in range(N_CONFIGS)]


def fit_predict(params, train, test):
    line = base.trend(train)
    m = LGBMRegressor(**{**ALWAYS, **params})
    m.fit(train[F], train[T] - line(train.year))
    return m.predict(test[F]) + line(test.year)


# ------------------------------------------------------------------
# folds
# ------------------------------------------------------------------

def year_blocks(years):
    ys = np.sort(np.unique(years))
    return {y: i % N_YEAR_BLOCKS for i, y in enumerate(ys)}


def outer_folds(df, scheme):
    if scheme == "random":
        kf = KFold(5, shuffle=True, random_state=SEED)
        for k, (tr, te) in enumerate(kf.split(df), 1):
            test = np.zeros(len(df), bool)
            test[te] = True
            yield f"fold {k}", ~test, test
    elif scheme == "year_state":
        blk = df.year.map(year_blocks(df.year)).to_numpy()
        st = df.state_alpha.to_numpy()
        for s in np.unique(st):
            for b in range(N_YEAR_BLOCKS):
                test = (st == s) & (blk == b)
                yield f"{s} block {b}", (st != s) & (blk != b), test


def inner_folds(train, kind, rng):
    n = len(train)
    if kind == "random":
        kf = KFold(N_INNER, shuffle=True, random_state=int(rng.integers(1e9)))
        for tr, te in kf.split(train):
            a, b = np.zeros(n, bool), np.zeros(n, bool)
            a[tr], b[te] = True, True
            yield a, b
    else:
        years = rng.permutation(np.unique(train.year))
        states = rng.permutation(np.unique(train.state_alpha))
        yg = {y: i % N_INNER for i, y in enumerate(years)}
        sg = {s: i % N_INNER for i, s in enumerate(states)}
        ty = train.year.map(yg).to_numpy()
        ts = train.state_alpha.map(sg).to_numpy()
        for k in range(N_INNER):
            test = (ty == k) & (ts == k)
            if test.any():
                yield (ty != k) & (ts != k), test


def choose(train, configs, kind, rng):
    folds = list(inner_folds(train, kind, rng))
    scores = []
    for cfg in configs:
        se, cnt = 0.0, 0
        for tr, te in folds:
            p = fit_predict(cfg, train[tr], train[te])
            se += float(np.sum((p - train[te][T].to_numpy()) ** 2))
            cnt += int(te.sum())
        scores.append(np.sqrt(se / cnt))
    i = int(np.argmin(scores))
    return configs[i], scores[i]


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    tee_to("tuning")
    t_start = time.time()
    df = base.load()
    configs = draw_configs()
    print(f"rows {len(df):,}   configurations {N_CONFIGS}   inner folds {N_INNER}")
    print(f"fixed: {FIXED}")

    long, choices = [], []
    for scheme in ["random", "year_state"]:
        fl = list(outer_folds(df, scheme))
        preds = {v: np.full(len(df), np.nan)
                 for v in ["fixed", "tuned_random", "tuned_blocked"]}
        print(f"\n=== outer: {scheme}, {len(fl)} folds ===", flush=True)
        t0 = time.time()
        for i, (label, tr, te) in enumerate(fl, 1):
            train, test = df[tr], df[te]
            rng = np.random.default_rng([SEED, i, len(scheme)])
            preds["fixed"][te] = fit_predict(FIXED, train, test)
            for kind in ["random", "blocked"]:
                cfg, inner = choose(train, configs, kind, rng)
                preds[f"tuned_{kind}"][te] = fit_predict(cfg, train, test)
                choices.append({"scheme": scheme, "fold": label,
                                "inner": kind, "inner_rmse": inner, **cfg})
            el = time.time() - t0
            print(f"  {i:>3}/{len(fl)}  {label:<14} test {te.sum():>5}   "
                  f"{el/60:5.1f} min, ~{(len(fl) - i) * el / i / 60:5.1f} min left",
                  flush=True)
        for v, p in preds.items():
            if np.isnan(p).any():
                raise SystemExit(f"{scheme}/{v}: unpredicted rows")
            long.append(pd.DataFrame({"fips": df.fips, "year": df.year,
                                      "region": df.region, "scheme": scheme,
                                      "model": v, "y_true": df[T], "y_pred": p}))
    long = pd.concat(long, ignore_index=True)
    choices = pd.DataFrame(choices)

    # --- scores with intervals, seasons resampled ------------------
    years = sorted(df.year.unique())
    counts = rob.boot_counts(len(years), np.random.default_rng(SEED))
    boots = rob.boot_rmse(long, "year", counts, years)

    rows = []
    for (s, m), b in boots.items():
        g = long[(long.scheme == s) & (long.model == m)]
        lo, hi = rob.ci(b)
        rows.append({"scheme": s, "model": m,
                     "rmse": float(np.sqrt(np.mean((g.y_pred - g.y_true) ** 2))),
                     "lo95": lo, "hi95": hi})
    scores = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\n--- RMSE, 95% interval from resampling seasons ---")
    print(scores.round(2).to_string(index=False))

    diffs = []
    for s in ["random", "year_state"]:
        for a, b in [("tuned_random", "fixed"), ("tuned_blocked", "fixed"),
                     ("tuned_random", "tuned_blocked")]:
            d = boots[(s, a)] - boots[(s, b)]
            lo, hi = rob.ci(d)
            pa = scores[(scores.scheme == s) & (scores.model == a)].rmse.iloc[0]
            pb = scores[(scores.scheme == s) & (scores.model == b)].rmse.iloc[0]
            diffs.append({"scheme": s, "comparison": f"{a} minus {b}",
                          "difference": pa - pb, "lo95": lo, "hi95": hi,
                          "clear": "yes" if (lo > 0 or hi < 0) else "no"})
    diffs = pd.DataFrame(diffs)
    print("\n--- paired differences; negative means the first is better ---")
    print(diffs.round(2).to_string(index=False))

    # --- what each kind of tuning chose ---------------------------
    print("\n--- settings chosen, median across outer folds ---")
    summ = (choices.groupby(["scheme", "inner"])
            [["num_leaves", "min_child_samples", "learning_rate",
              "n_estimators", "reg_lambda", "inner_rmse"]].median())
    print(summ.to_string())
    print(f"  fixed for comparison: num_leaves {FIXED['num_leaves']}, "
          f"min_child_samples {FIXED['min_child_samples']}")

    # Inner score against outer score: how much did each kind of inner
    # validation flatter the setting it picked?
    out = (long.assign(se=(long.y_pred - long.y_true) ** 2)
           .groupby(["scheme", "model"]).se.mean() ** 0.5)
    print("\n--- inner estimate against outer result ---")
    for s in ["random", "year_state"]:
        for kind in ["random", "blocked"]:
            inner = choices[(choices.scheme == s) & (choices.inner == kind)].inner_rmse.mean()
            print(f"  outer {s:<10} inner {kind:<8} inner says {inner:5.1f}, "
                  f"outer gives {out[(s, 'tuned_' + kind)]:5.1f}")

    RESULTS.mkdir(exist_ok=True)
    scores.to_csv(RESULTS / "tuning_scores.csv", index=False)
    diffs.to_csv(RESULTS / "tuning_differences.csv", index=False)
    choices.to_csv(RESULTS / "tuning_choices.csv", index=False)
    print(f"\nwritten: three tables in {RESULTS}")
    print(f"total {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
