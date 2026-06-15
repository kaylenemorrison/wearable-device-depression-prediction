"""
Basic-aggregation sampling-sensitivity check.

Question this answers:
    For a window of N days, how well do the basic aggregations (median, std)
    computed on only k of the N days recover the value you'd get from the full
    window? I.e. is the looser basic missing-threshold (allowing ~50% present)
    statistically defensible, or does tightening toward the catch22 0.75 floor
    materially clean up the estimates?

Method (isolates SAMPLING sufficiency, not MNAR selection bias):
    Take only FULLY-OBSERVED windows (all N days present) so the full-window
    aggregate is a clean reference. For each k, draw random k-day subsets of
    each such window, recompute the stat, and measure how far the subsample
    stat lands from the full-window stat. Because we subsample WITHIN a fixed
    full window, the device-non-wear selection effect is held out -- this is the
    best case for the loose threshold. (The MNAR/representativeness question is
    separate; compare across windows of differing completeness for that.)

Metric:
    norm_MAE(k) = median(|stat_k - stat_full|) / between-window spread of stat_full
    A value << 1 means subsampling noise is small relative to the real signal
    spread you are trying to capture (k is enough); ~1 or more means the stat at
    k is mostly sampling noise. For std we also report the signed bias, since the
    sample std is biased low at small k.

Reads only cleaned daily columns; prints only aggregate error metrics (metadata),
never participant rows. Run e.g.:

    python scratch/basic_agg_sensitivity.py --window-size 14
"""
import argparse
import math
import os
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO_ROOT, "data_cleaned")
PHQ8 = f"{DATA}/phq8_data.csv"
SPEC = os.path.join(REPO_ROOT, "feature_spec.yaml")

STREAM_FILES = {
    "sleep":     f"{DATA}/sleep_data.csv",
    "activity":  f"{DATA}/activity_data.csv",
    "steps":     f"{DATA}/steps_data.csv",
    "heartrate": f"{DATA}/heartrate_data.csv",
}

GROUP = ["participant_name", "timeInterval_ID"]

# Continuous profiles: median/std are the relevant aggregations. timeofday is
# circular (different stats) and binary_count* features are 0/1 day-flags
# aggregated by sum -- both are excluded from a median/std sensitivity check.
CONTINUOUS_PROFILES = {"level", "level_low", "level_high", "total", "total_low"}


def load_continuous_features():
    """{ 'stream.col': (cleaned_file, column) } for every continuous-profile
    feature in the spec. Skips circular and binary day-flag features."""
    with open(SPEC) as fh:
        spec = yaml.safe_load(fh)
    out = {}
    for stream in STREAM_FILES:
        for col, meta in spec.get(stream, {}).items():
            if isinstance(meta, dict) and meta.get("profile") in CONTINUOUS_PROFILES:
                out[f"{stream}.{col}"] = (STREAM_FILES[stream], col)
    return out


def window_value_matrix(phq8, daily, column, N):
    """(n_windows, N) array of each window's N prior-day values for `column`,
    NaN where that day's row is absent. Row order = lag 1..N per window."""
    w = phq8[GROUP].copy()
    w = w.loc[w.index.repeat(N)].reset_index(drop=True)
    w["lag"] = list(range(1, N + 1)) * len(phq8)
    w["day"] = w["timeInterval_ID"] - w["lag"]
    w["_ord"] = np.arange(len(w))
    src = daily[["participant_name", "timeInterval_ID", column]].rename(
        columns={"timeInterval_ID": "day"}
    )
    w = w.merge(src, on=["participant_name", "day"], how="left").sort_values("_ord")
    return w[column].to_numpy(dtype=float).reshape(len(phq8), N)


def stat_along(arr, name, axis):
    if name == "median":
        return np.median(arr, axis=axis)
    if name == "std":
        return np.std(arr, axis=axis, ddof=1)
    raise ValueError(name)


def sensitivity_for_feature(M, ks, stats, draws, rng):
    """M: (W, N) fully-observed reference windows (no NaN). Returns
    {stat: {'mae': {k: v}, 'bias': {k: v}, 'spread': s}}."""
    W, N = M.shape
    res = {}
    for stat in stats:
        full = stat_along(M, stat, axis=1)            # (W,)
        spread = np.std(full, ddof=1)
        res[stat] = {"mae": {}, "bias": {}, "spread": spread}
        if spread < 1e-12:                             # constant across windows
            for k in ks:
                res[stat]["mae"][k] = np.nan
                res[stat]["bias"][k] = np.nan
            continue
        for k in ks:
            # distinct random k-subsets per (window, draw)
            order = np.argsort(rng.random((W, draws, N)), axis=2)[:, :, :k]
            gathered = np.take_along_axis(M[:, None, :], order, axis=2)  # (W,draws,k)
            sub = stat_along(gathered, stat, axis=2)   # (W, draws)
            err = sub - full[:, None]
            res[stat]["mae"][k] = np.median(np.abs(err)) / spread
            res[stat]["bias"][k] = np.mean(err) / spread
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-size", type=int, default=14, choices=[3, 5, 7, 14, 21, 28])
    ap.add_argument("--draws", type=int, default=25, help="random subsets per window")
    ap.add_argument("--max-ref", type=int, default=1500,
                    help="cap on fully-observed reference windows per feature")
    ap.add_argument("--min-ref", type=int, default=50,
                    help="skip features with fewer fully-observed windows")
    ap.add_argument("--stats", nargs="+", default=["median", "std"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N = args.window_size
    rng = np.random.default_rng(args.seed)

    # k sweep: fractions of N, always distinct and < N.
    ks = sorted({math.ceil(f * N) for f in (0.25, 0.4, 0.5, 0.6, 0.75, 0.9)} - {N})
    k_basic = N - (N + 1) // 2        # min present under the basic threshold (~50%)
    k_c22 = math.ceil(0.75 * N)        # catch22 floor
    ks = sorted(set(ks) | {k_basic, k_c22})
    ks = [k for k in ks if 1 < k < N]

    phq8 = pd.read_csv(PHQ8)
    feats = load_continuous_features()
    print(f"window_size={N}  | reference = fully-observed windows (all {N} days)")
    print(f"k sweep: {ks}   (basic min-present={k_basic}, catch22 floor={k_c22})")
    print(f"draws/window={args.draws}, max_ref={args.max_ref}\n")
    print("norm_MAE = median(|stat_k - stat_full|) / between-window spread(stat_full)")
    print("  <~0.3 sampling noise small vs signal | ~1+ stat at k is mostly noise\n")

    kcols = "".join(f"{k:>6}" for k in ks)
    print(f"{'feature / stat':<28}{'Wref':>6}{kcols}")
    print("-" * (28 + 6 + len(kcols)))

    for name, (path, col) in feats.items():
        daily = pd.read_csv(path)
        if col not in daily.columns:
            print(f"{name:<28}  COLUMN '{col}' NOT FOUND")
            continue
        M = window_value_matrix(phq8, daily, col, N)
        present = np.isfinite(M).sum(axis=1)
        ref = M[present == N]
        W = len(ref)
        if W < args.min_ref:
            print(f"{name:<28}{W:>6}   (too few fully-observed windows; skipped)")
            continue
        if W > args.max_ref:
            ref = ref[rng.choice(W, args.max_ref, replace=False)]
        res = sensitivity_for_feature(ref, ks, args.stats, args.draws, rng)
        for i, stat in enumerate(args.stats):
            label = f"{name} :: {stat}" if i == 0 else f"{'':>4}:: {stat}"
            wcol = f"{W:>6}" if i == 0 else f"{'':>6}"
            cells = "".join(f"{res[stat]['mae'][k]:>6.2f}" for k in ks)
            print(f"{label:<28}{wcol}{cells}")
            if stat == "std":  # surface the small-sample downward bias
                bias = "".join(f"{res[stat]['bias'][k]:>6.2f}" for k in ks)
                print(f"{'    (std bias)':<28}{'':>6}{bias}")


if __name__ == "__main__":
    main()
