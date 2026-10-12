"""
Check the cached masks against the figures the probe produced.

    python src/05b_verify_masks.py

04_probe_extraction.py built Story County's 2020 corn fraction by a
different route: no tile caching, no clip_box windowing, no footprint
matching, one county at a time. It reported

    corn fraction, in-county      0.465
    corn fraction, whole array    0.223
    pixels with any corn       24,338
    array shape               166 x 347

05_build_corn_masks.py then restructured all of that for speed. Every
change was meant to remove redundant work rather than approximate
anything, so the mask it produces for Story County should be identical
to the one the probe produced.

If it is, the optimisations are demonstrably lossless rather than
assumed to be. If it isn't, something in the restructuring changed an
answer, and finding that out now is worth more than eight hours of
runtime.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASKS = ROOT / "data" / "masks" / "corn_2020.npz"
YIELDS = ROOT / "data" / "nass_corn_yields.csv"

STORY = "19169"
EXPECTED = {
    "shape": (166, 347),
    "array_mean": 0.223,
    "in_county_mean": 0.465,
    "pixels_with_corn": 24338,
}


def main() -> None:
    z = np.load(MASKS)
    print(f"counties in file: {len(z.files)}")

    arr = z[STORY]
    print(f"\n--- Story County, Iowa ({STORY}) ---")
    print(f"  shape            : {arr.shape}      expected {EXPECTED['shape']}")
    print(f"  mean over array  : {arr.mean():.3f}        expected {EXPECTED['array_mean']}")
    print(f"  pixels with corn : {int((arr > 0).sum()):,}     expected {EXPECTED['pixels_with_corn']:,}")
    print(f"  max              : {arr.max():.3f}")

    # The in-county mean cannot be recovered from the saved array
    # alone, because outside-polygon cells were filled with zero and
    # are now indistinguishable from genuine no-corn cells. That is a
    # limitation of the cache format worth knowing: the weighting in
    # 06 uses the NDVI mask to exclude them, so it is unaffected, but
    # any summary computed from these files alone is diluted.
    nonzero = arr[arr > 0]
    print(f"  mean where corn>0: {nonzero.mean():.3f}  "
          f"(not the same as in-county mean)")

    ok = (
        arr.shape == EXPECTED["shape"]
        and abs(arr.mean() - EXPECTED["array_mean"]) < 0.002
        and abs(int((arr > 0).sum()) - EXPECTED["pixels_with_corn"]) < 50
    )
    print(f"\n  matches the probe: {'yes' if ok else 'NO'}")

    # --- the distribution across all counties --------------------
    stats = pd.DataFrame(
        [(f, float(z[f].mean()), float(z[f].max()), int((z[f] > 0).sum()))
         for f in z.files],
        columns=["fips", "array_mean", "max", "corn_pixels"],
    )

    print("\n--- across 742 counties ---")
    print(stats.array_mean.describe().round(3).to_string())

    zero = stats[stats.array_mean == 0]
    print(f"\n  counties with no corn at all: {len(zero)}")
    if len(zero):
        y = pd.read_csv(YIELDS, dtype={"fips": str})
        names = y[["fips", "county_name", "state_alpha"]].drop_duplicates()
        print(zero.merge(names, on="fips", how="left")
              [["fips", "county_name", "state_alpha"]].head(15).to_string(index=False))
        print("  A county with no corn should also have no NASS yield.")
        print(f"  of those, how many have a 2020 yield: "
              f"{y[(y.fips.isin(zero.fips)) & (y.year == 2020)].shape[0]}")

    print("\n  highest corn fraction:")
    top = stats.nlargest(8, "array_mean")
    y = pd.read_csv(YIELDS, dtype={"fips": str})
    names = y[["fips", "county_name", "state_alpha"]].drop_duplicates()
    print(top.merge(names, on="fips", how="left")
          [["fips", "county_name", "state_alpha", "array_mean"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
