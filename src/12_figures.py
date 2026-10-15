"""
The figures in the README, drawn from the saved results.

    python src/12_figures.py

Reads   results/*.csv, data/predictions_gbm.parquet,
        data/predictions_robust.parquet, data/counties.gpkg
Writes  figures/*.png

Every title states a finding, and every number in a title is computed
from the data it labels rather than typed in, so a title cannot drift
out of step with its chart when the results change.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runlog import tee_to

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATA = ROOT / "data"

# Okabe and Ito, readable with the common forms of colour blindness.
C = {"gbm_detrended": "#0072B2", "gbm_raw": "#D55E00", "ridge": "#E69F00", "lue_1": "#009E73",
     "peak_k": "#CC79A7", "peak_lin": "#56B4E9", "trend_only": "#999999"}
LABEL = {"gbm_detrended": "Gradient boosting", "ridge": "Ridge regression",
         "lue_1": "Light use efficiency", "peak_k": "k x peak NDVI",
         "peak_lin": "Linear on peak NDVI", "trend_only": "Trend line"}
SCHEME = {"random": "Random", "region": "District", "year": "Year",
          "region_buffer": "District and neighbours", "both": "Year and district",
          "state": "State", "both_state": "Year and state"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def fig_schemes(out):
    r = pd.read_csv(RES / "robust_rmse.csv")
    models = ["gbm_detrended", "ridge", "lue_1"]
    r = r[r.model.isin(models)]
    order = (r[r.model == "gbm_detrended"].sort_values("rmse").scheme.tolist())
    g = r.set_index(["model", "scheme"])
    lo_g, hi_g = g.loc[("gbm_detrended", "random"), "rmse"], g.loc[("gbm_detrended", "both_state"), "rmse"]
    lead_rand = g.loc[("ridge", "random"), "rmse"] - lo_g
    lead_strict = g.loc[("ridge", "both_state"), "rmse"] - hi_g

    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(order))
    for j, m in enumerate(models):
        d = g.loc[m].reindex(order)
        off = (j - 1) * 0.22
        ax.errorbar(d.rmse, y + off, xerr=[d.rmse - d.lo95, d.hi95 - d.rmse],
                    fmt="o", color=C[m], ms=5, capsize=2, lw=1.2, label=LABEL[m])
    ax.set_yticks(y, [SCHEME[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, bushels per acre (95% interval, seasons resampled)")
    ax.set_title(f"Boosting's error rises {100*(hi_g/lo_g - 1):.0f}% from random folds "
                 f"to a held-out year and state,\nand its lead over ridge regression "
                 f"falls from {lead_rand:.1f} bushels to {lead_strict:.1f}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3)
    ax.grid(axis="x", color="#eeeeee")
    ax.text(0, -0.3, "Holdout schemes ordered by gradient boosting error. "
            "Negative lead: ridge ahead.", transform=ax.transAxes, fontsize=8,
            color="#666666")
    save(fig, out, "schemes.png")


def fig_degradation(out):
    r = pd.read_csv(RES / "robust_rmse.csv").set_index(["model", "scheme"]).rmse
    params = {"gbm_detrended": "hundreds", "ridge": "25", "peak_lin": "2",
              "trend_only": "2", "peak_k": "1", "lue_1": "1"}
    rows = [(m, r[(m, "both_state")] / r[(m, "random")]) for m in params]
    d = pd.DataFrame(rows, columns=["model", "ratio"]).sort_values("ratio")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh([f"{LABEL[m]}  ({params[m]})" for m in d.model], 100 * (d.ratio - 1),
            color=[C[m] for m in d.model])
    for i, v in enumerate(100 * (d.ratio - 1)):
        ax.text(v + 0.6, i, f"+{v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("Rise in RMSE from random folds to year and state held out (%)")
    lue = 100 * (d.set_index("model").ratio["lue_1"] - 1)
    pk = 100 * (d.set_index("model").ratio["peak_k"] - 1)
    ax.set_title(f"The physical model degrades least (+{lue:.0f}%), but a one-parameter\n"
                 f"model with no physics is close behind (+{pk:.0f}%)")
    ax.text(0, -0.2, "Fitted parameters in brackets.", transform=ax.transAxes,
            fontsize=8, color="#666666")
    save(fig, out, "degradation.png")


def fig_bias_by_year(out):
    p = pd.read_parquet(DATA / "predictions_gbm.parquet")
    p = p[(p.scheme == "year") & (p.target == "detrended")]
    b = (p.y_pred - p.y_true).groupby(p.year).mean()
    under = b[b < -10].index.tolist()
    over = b[b > 10].index.tolist()

    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = ["#0072B2" if v < 0 else "#E69F00" for v in b.values]
    bars = ax.bar(b.index, b.values, color=cols)
    i12 = list(b.index).index(2012)
    bars[i12].set_edgecolor("black")
    bars[i12].set_linewidth(1.5)
    ax.annotate("2012\ndrought", (2012, b[2012]), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(b.index, [str(y) for y in b.index], rotation=0, fontsize=8)
    ax.set_ylabel("Mean error, bu/acre\n(prediction minus actual)")
    ax.set_title(f"Each held-out season is missed as a whole: under by more than 10 in "
                 f"{', '.join(map(str, under))};\nover by more than 10 in "
                 f"{', '.join(map(str, over))}")
    ax.set_ylim(top=b.max() + 6)
    ax.text(0, -0.2, "Gradient boosting, detrended, one year held out at a time. "
            "Blue: underestimated. Orange: overestimated.", transform=ax.transAxes,
            fontsize=8, color="#666666")
    save(fig, out, "bias_by_year.png")


def fig_2012_map(out):
    p = pd.read_parquet(DATA / "predictions_robust.parquet")
    p = p[(p.scheme == "both_state") & (p.model == "gbm_detrended") & (p.year == 2012)]
    p = p.assign(err=p.y_pred - p.y_true)
    c = gpd.read_file(DATA / "counties.gpkg", layer="counties").to_crs("EPSG:5070")
    m = c.merge(p[["fips", "err"]], on="fips", how="left")
    states = c.dissolve("state_alpha")
    share_over = 100 * (p.err > 0).mean()
    worst = (m.dropna(subset=["err"]).groupby("state_alpha").err.mean()
             .sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    m[m.err.isna()].plot(ax=ax, color="#eeeeee", lw=0)
    lim = 60
    m.dropna(subset=["err"]).plot(ax=ax, column="err", cmap="RdBu_r", vmin=-lim,
                                  vmax=lim, lw=0, legend=True,
                                  legend_kwds={"shrink": 0.6,
                                               "label": "Prediction minus actual, bu/acre"})
    states.boundary.plot(ax=ax, color="black", lw=0.6)
    ax.set_axis_off()
    ax.set_title(f"2012, year and state held out: overestimated in {share_over:.0f}% of "
                 f"counties,\nmost in {worst.index[0]} (mean {worst.iloc[0]:+.0f}) and "
                 f"{worst.index[1]} ({worst.iloc[1]:+.0f})")
    ax.text(0, -0.02, f"Gradient boosting. Colour scale capped at +/-{lim}. "
            "Grey: no published 2012 yield.", transform=ax.transAxes, fontsize=8,
            color="#666666")
    save(fig, out, "error_map_2012.png")


def fig_in_season(out):
    d = pd.read_csv(RES / "in_season.csv")
    d = d[d.model.isin(["gbm_detrended", "trend_only"])]
    labels = {6: "late June", 8: "late July", 10: "late August", 14: "full season"}

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for s, ls in [("year", "--"), ("both_state", "-")]:
        g = d[(d.scheme == s) & (d.model == "gbm_detrended")].sort_values("cutoff")
        ax.plot(g.cutoff, g.rmse, ls, marker="o", color=C["gbm_detrended"],
                label=f"Gradient boosting, {SCHEME[s].lower()} held out")
    t = d[(d.scheme == "both_state") & (d.model == "trend_only")].rmse.iloc[0]
    ax.axhline(t, color=C["trend_only"], lw=1, ls=":")
    ax.text(14, t + 0.4, "trend line", ha="right", fontsize=8, color="#666666")
    ax.set_xticks(list(labels), list(labels.values()))
    ax.set_ylabel("RMSE, bu/acre")
    g = d[(d.scheme == "both_state") & (d.model == "gbm_detrended")].set_index("cutoff").rmse
    skill = lambda c: 100 * (1 - g[c] / t)
    share_jul = skill(8) / skill(14) * 100
    ax.set_title(f"Almost no skill by late June ({skill(6):.0f}% better than the trend); "
                 f"by late July,\n{share_jul:.0f}% of the full-season skill "
                 f"(year and state held out)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#eeeeee")
    save(fig, out, "in_season.png")


def fig_tuning(out):
    ch = pd.read_csv(RES / "tuning_choices.csv")
    sc = pd.read_csv(RES / "tuning_scores.csv").set_index(["scheme", "model"]).rmse
    d = ch[ch.scheme == "year_state"].groupby("inner").inner_rmse.mean()
    rows = [("Tuned on random\ninner folds", d["random"], sc[("year_state", "tuned_random")]),
            ("Tuned on held-out\ninner folds", d["blocked"], sc[("year_state", "tuned_blocked")])]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color="#bbbbbb",
           label="What tuning expected (inner folds)")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color=C["gbm_detrended"],
           label="What it got (year and state held out)")
    for i, r in enumerate(rows):
        ax.text(i - 0.18, r[1] + 0.3, f"{r[1]:.1f}", ha="center", fontsize=9)
        ax.text(i + 0.18, r[2] + 0.3, f"{r[2]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("RMSE, bu/acre")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    r0 = rows[0]
    ax.set_title(f"Tuned the usual way, the model expected {r0[1]:.1f} bushels of error\n"
                 f"and got {r0[2]:.1f} on a held-out year and state")
    save(fig, out, "tuning.png")


def fig_future(out):
    d = pd.read_csv(RES / "future_by_year.csv")
    d = d[d.scheme == "future"]
    models = ["gbm_detrended", "gbm_raw", "trend_only"]
    labels = {"gbm_detrended": "Gradient boosting, detrended",
              "gbm_raw": "Gradient boosting, raw yield",
              "trend_only": "Trend line alone"}
    years = sorted(d.year.unique())
    b = d.set_index(["model", "year"]).bias

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(years))
    for j, m in enumerate(models):
        vals = [b[(m, y)] for y in years]
        ax.bar(x + (j - 1) * 0.26, vals, 0.26, color=C.get(m, "#D55E00"),
               label=labels[m])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [str(y) for y in years])
    ax.set_ylabel("Mean error, bu/acre\n(prediction minus actual)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8)
    last = years[-1]
    ax.set_title(f"Trained on 2008 to 2021, the raw model underestimates {last} by "
                 f"{-b[('gbm_raw', last)]:.0f} bushels;\nthe detrended model, by "
                 f"{-b[('gbm_detrended', last)]:.0f}")
    save(fig, out, "future_bias.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    tee_to("figures")
    print(f"writing to {args.out}")
    fig_schemes(args.out)
    fig_degradation(args.out)
    fig_bias_by_year(args.out)
    fig_2012_map(args.out)
    fig_in_season(args.out)
    if (RES / "tuning_choices.csv").exists():
        fig_tuning(args.out)
    if (RES / "future_by_year.csv").exists():
        fig_future(args.out)


if __name__ == "__main__":
    main()
