"""
catch22 window-survival diagnostic.

For each catch22 base feature, the per-window series is built from a SINGLE
daily column, so the present-day count is feature-specific (different columns go
missing at different rates, both structurally and at the value level). There is
therefore no single "window survives" number: this prints one survival curve per
base feature.

For each base feature and each candidate floor X, it reports how many (and what
fraction of) PHQ-8 windows have >= X non-NaN days for that column, at the chosen
window size. Run from the scripts/ directory's data layout:

    python scratch/catch22_window_survival.py --window-size 14

Only counts/fractions (metadata) are printed -- never participant rows.
"""
import argparse
import math
import os
import pandas as pd
import yaml

# Resolve data_cleaned/ and the spec relative to the repo root (this file lives
# in scratch/), so the script runs from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO_ROOT, "data_cleaned")
PHQ8 = f"{DATA}/phq8_data.csv"
SPEC = os.path.join(REPO_ROOT, "feature_spec.yaml")

# Map a stream name to its cleaned CSV.
STREAM_FILES = {
    "sleep":     f"{DATA}/sleep_data.csv",
    "activity":  f"{DATA}/activity_data.csv",
    "steps":     f"{DATA}/steps_data.csv",
    "heartrate": f"{DATA}/heartrate_data.csv",
}


def load_base_features():
    """Read the catch22 base-feature list from feature_spec.yaml so the spec
    stays the single source of truth. Each entry is "stream.column"; returns
    {entry: (cleaned_file, column)}."""
    with open(SPEC) as fh:
        spec = yaml.safe_load(fh)
    out = {}
    for entry in spec["catch22"]["base_features"]:
        stream, col = entry.split(".", 1)
        out[entry] = (STREAM_FILES[stream], col)
    return out


# Each catch22 base feature -> (cleaned file, daily column name).
BASE_FEATURES = load_base_features()

GROUP = ["participant_name", "timeInterval_ID"]


def window_present_days(phq8, daily, column, window_size):
    """Per PHQ-8 window, count days where `column` is observed (non-NaN).

    Replicates merge_datasets.create_window: each assessment expands to the
    `window_size` prior day-IDs (timeInterval_ID - 1 .. - window_size), then we
    left-join the daily column and count non-NaN matches.
    """
    w = phq8[GROUP].copy()
    w = w.loc[w.index.repeat(window_size)].copy()
    w["lag"] = list(range(1, window_size + 1)) * len(phq8)
    w["day"] = w["timeInterval_ID"] - w["lag"]

    src = daily[["participant_name", "timeInterval_ID", column]].rename(
        columns={"timeInterval_ID": "day"}
    )
    w = w.merge(src, on=["participant_name", "day"], how="left")
    w["present"] = w[column].notna().astype(int)
    return w.groupby(GROUP)["present"].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-size", type=int, default=14, choices=[3, 5, 7, 14, 21, 28])
    ap.add_argument("--floors", type=int, nargs="+", default=None,
                    help="Present-day floors to report. Default: ~50-100%% of the "
                         "window size, so the sweep scales with --window-size.")
    args = ap.parse_args()
    N = args.window_size
    if args.floors is None:
        # Scale floors with N (fabrication-bound framing: floor = fraction of N).
        floors = sorted({math.ceil(frac * N) for frac in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)})
    else:
        floors = [f for f in args.floors if f <= N]

    phq8 = pd.read_csv(PHQ8)
    n_windows = len(phq8)
    print(f"window_size={N}  |  total PHQ-8 windows={n_windows}\n")

    header = f"{'base feature':<26}" + "".join(f">={f:>3} " for f in floors)
    print(header)
    print("-" * len(header))

    survival = {}
    for name, (path, col) in BASE_FEATURES.items():
        daily = pd.read_csv(path)
        if col not in daily.columns:
            print(f"{name:<26}  COLUMN '{col}' NOT FOUND in {path}")
            continue
        counts = window_present_days(phq8, daily, col, N)
        # reindex so windows with no matching days at all count as 0 present
        counts = counts.reindex(
            phq8.set_index(GROUP).index, fill_value=0
        )
        survival[name] = counts
        row = f"{name:<26}"
        for f in floors:
            pct = (counts >= f).mean() * 100
            row += f"{pct:>4.0f}%"
        print(row)

    # Intersection (all base columns >= X simultaneously) -- shown only to
    # illustrate why a single global floor is the wrong, over-restrictive unit.
    if len(survival) == len(BASE_FEATURES):
        print("\n[for contrast] windows with ALL columns >= X (intersection):")
        aligned = pd.DataFrame(survival)
        row = f"{'ALL  simultaneously':<26}"
        for f in floors:
            pct = (aligned >= f).all(axis=1).mean() * 100
            row += f"{pct:>4.0f}%"
        print(row)


if __name__ == "__main__":
    main()
