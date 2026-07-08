import argparse
import json
import os
import time
from functools import reduce
from typing import Callable

import pandas as pd
import numpy as np
import yaml
import pycatch22

PATH_TO_PHQ8 = "../data_cleaned/phq8_data.csv"
PATH_TO_SLEEP = "../data_cleaned/sleep_data.csv"
PATH_TO_ACTIVITY = "../data_cleaned/activity_data.csv"
PATH_TO_STEPS = "../data_cleaned/steps_data.csv"
PATH_TO_HEARTRATE = "../data_cleaned/heartrate_data.csv"
PATH_TO_DEMOGRAPHICS = "../data_cleaned/totaldataset.parquet"
PATH_TO_MERGED = "../data_merged"
PATH_TO_YAML = "../feature_spec.yaml"

SIX_HOURS_SECONDS = 6 * 60 * 60

GROUP_COLS = ['participant_name', 'timeInterval_ID']

# catch22 panels are nulled for a window whose longest run of consecutive
# missing days exceeds this, even if the window clears min_present_fraction: a
# long contiguous gap can only be filled by interpolating a multi-day ramp, i.e.
# fabricated dynamics, which corrupts the (spacing/adjacency-sensitive) catch22
# axes. This gate is on CONSECUTIVE missingness only; scattered gaps up to the
# present-fraction budget are still interpolated.
# Deliberately a FLAT 2 at every window size (not scaled to the window): the
# threshold is set by the SMALLEST catch22 window (14), where a run longer than
# 2 interpolated days is already an unreasonable share of the series to
# fabricate. A larger window does not justify a longer contiguous fabricated
# stretch, so the same cap holds at 21/28.
MAX_CATCH22_CONSECUTIVE_MISSING = 2


def dataset_stem(window_size: int, min_present_fraction: float,
                 catch22_enabled: bool) -> str:
    """Filename stem encoding every choice that determines the dataset's content,
    so it is fully identifiable from the name and datasets built at different
    settings live side by side instead of overwriting:
      - `w{N}`   window size
      - `mp{x}`  the single present-fraction floor, shared by the simple
                 aggregations and the catch22 panel
      - `c22` / `noc22`  whether the catch22 panel is present (gated on the
                 window via enabled_window_sizes). This is derivable from the
                 window, but is spelled out so the catch22 regime -- which also
                 sets the min allowed `mp` floor -- is readable without consulting
                 feature_spec.yaml.

    modeling.py reconstructs the same name (it derives catch22_enabled from
    feature_spec.yaml) -- keep the two conventions in sync.
    """
    c22 = "c22" if catch22_enabled else "noc22"
    return f"final_dataset_w{window_size}_mp{min_present_fraction:g}_{c22}"


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def create_window(phq8: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """One row per (assessment, lag): the `window_size` prior day IDs of each
    PHQ-8 assessment, used to join the daily streams onto each assessment."""
    window = phq8[GROUP_COLS].copy()

    window = window.loc[window.index.repeat(window_size)].copy()
    window['lag'] = list(range(1, window_size + 1)) * len(phq8)

    # Compute the prior day IDs (lag=1 is the most recent prior day).
    window['timeInterval_ID_window'] = window['timeInterval_ID'] - window['lag']
    return window


# ---------------------------------------------------------------------------
# Circular statistics for time-of-day columns (hours in [0, 24))
# ---------------------------------------------------------------------------
def circular_std(series: pd.Series) -> float:
    """Approximate circular std for time-of-day in hours safely."""
    clean_series = series.dropna()
    if clean_series.empty:
        return np.nan
    radians = (clean_series % 24) / 24 * 2 * np.pi
    sin_mean = np.sin(radians).mean()
    cos_mean = np.cos(radians).mean()
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    # Clip R to avoid floating point errors breaking log domains
    R = np.clip(R, 1e-9, 1.0)
    val = -2 * np.log(R)
    # Ensure value isn't slightly negative due to precision issues
    val = max(0.0, val)
    return np.sqrt(val) * 24 / (2 * np.pi)


def circular_median(series: pd.Series) -> float:
    """Circular median for time-of-day in hours.
    Returns the observed value that minimises total arc distance to all others.
    Robust to outlier nights; brute-force is fine for window sizes 3-28."""
    clean = series.dropna() % 24
    if clean.empty:
        return np.nan
    if len(clean) == 1:
        return clean.iloc[0]
    angles = clean.values / 24 * 2 * np.pi

    def arc_dist(a: float, b: float) -> float:
        d = abs(a - b) % (2 * np.pi)
        return min(d, 2 * np.pi - d)

    total_dists = [sum(arc_dist(a, b) for b in angles) for a in angles]
    return clean.iloc[int(np.argmin(total_dists))]


def calculate_sleep_duration(onset: pd.Series, offset: pd.Series, max_val: int = 24) -> pd.Series:
    return (offset - onset + max_val) % max_val


# Maps a profile operation name (from feature_spec.yaml) to something pandas'
# named aggregation accepts: a built-in reducer string, or a circular-stat
# callable for time-of-day columns.
OP_DISPATCH = {
    'median': 'median',
    'mean': 'mean',
    'std': 'std',
    'min': 'min',
    'max': 'max',
    'sum': 'sum',
    'circ_median': circular_median,
    'circ_std': circular_std,
}

# ---------------------------------------------------------------------------
# Generic, profile-driven simple aggregation
# ---------------------------------------------------------------------------
def _compute_adherence_indicators(
    phq8: pd.DataFrame,
    stream: pd.DataFrame,
    stream_name: str,
    window_size: int,
) -> pd.DataFrame:
    """Compute adherence indicators for a stream:
    - Days with no data at all (structural count): '<stream_name>_missing_last<window_size>'
    - Maximum consecutive missing days: '<stream_name>_missing_consecutive_max'
    """
    window = create_window(phq8, window_size)
    stream = stream.rename(columns={'timeInterval_ID': 'timeInterval_ID_stream'})

    window = window.merge(
        stream,
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_stream'],
        how='left',
    )

    # Compute structural count: days with no row at all
    window['_row_absent'] = window['timeInterval_ID_stream'].isna().astype(int)
    structural = (
        window.groupby(GROUP_COLS)['_row_absent']
        .sum()
        .reset_index(name=f'{stream_name}_missing_last{window_size}')
    )

    # Compute max consecutive missing days
    def max_consecutive_missing(group):
        """Find the longest streak of consecutive missing days in a window."""
        is_missing = group['_row_absent'].values
        if len(is_missing) == 0:
            return 0
        max_streak = 0
        current_streak = 0
        for missing in is_missing:
            if missing:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    consecutive = (
        window.groupby(GROUP_COLS)
        .apply(max_consecutive_missing, include_groups=False)
        .reset_index(name=f'{stream_name}_missing_consecutive_max')
    )

    # Merge both indicators
    result = structural.merge(consecutive, on=GROUP_COLS, how='left')
    return result


def aggregate_stream(
    phq8: pd.DataFrame,
    stream: pd.DataFrame,
    pipeline: dict[str, list[str]],
    stream_name: str,
    window_size: int,
    min_present_fraction: float,
    derive: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Window a daily stream onto each PHQ-8 assessment and apply, per base
    feature, only the aggregation operations its YAML profile assigns.

    `pipeline` is {feature_name: [op, ...]} produced by parse_yaml. Output
    columns follow "<stream>_<feature>_<op>". Per-feature missing counts used
    for nulling are dropped before returning.

    `min_present_fraction` is the floor on the fraction of window-days a feature
    must have present (same convention as the catch22 panel): a feature whose
    present fraction falls below it has its aggregates nulled for that window.

    `derive(window)` optionally adds derived daily columns (e.g. sleep_midpoint)
    that the pipeline references but the raw table does not contain.
    """
    window = create_window(phq8, window_size)
    stream = stream.rename(columns={'timeInterval_ID': 'timeInterval_ID_stream'})

    window = window.merge(
        stream,
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_stream'],
        how='left',
    )

    if derive is not None:
        window = derive(window)

    feats = list(pipeline.keys())

    # --- profile-driven aggregations ---------------------------------------
    # Each feature contributes only the ops its profile lists, named
    # "<stream>_<feature>_<op>" via pandas named aggregation.
    named_aggs = {}
    for feat, ops in pipeline.items():
        for op in ops:
            named_aggs[f'{stream_name}_{feat}_{op}'] = (feat, OP_DISPATCH[op])
    agg = window.groupby(GROUP_COLS).agg(**named_aggs).reset_index()

    # --- per-feature missing-day counts (used internally for nulling) ------
    # A feature is missing on a window-day if its value is NaN, folding both a
    # wholly-absent row (structural) and a value-level NaN into one count.
    miss = window[GROUP_COLS + feats].copy()
    miss[feats] = miss[feats].isna().astype(int)
    miss = miss.groupby(GROUP_COLS)[feats].sum().reset_index()
    agg = agg.merge(miss, on=GROUP_COLS, how='left')

    # --- per-feature nulling -----------------------------------------------
    # Null only the aggregates of features below the present-fraction floor, so
    # a window with good data for most features keeps them even when one sparse
    # column is under budget. agg[feat] holds the per-feature missing-day count;
    # convert to a present fraction so the test mirrors the catch22 panel.
    for feat in feats:
        present_fraction = (window_size - agg[feat]) / window_size
        invalid = (present_fraction < min_present_fraction).to_numpy()
        derived_cols = [f'{stream_name}_{feat}_{op}' for op in pipeline[feat]]
        agg.loc[invalid, derived_cols] = np.nan

    # Drop the per-feature counts.
    agg = agg.drop(columns=feats)

    return agg


def _derive_sleep(window: pd.DataFrame) -> pd.DataFrame:
    """Add the derived circadian-phase column the sleep pipeline references."""
    duration = calculate_sleep_duration(window['sleep_onset'], window['sleep_offset'])
    window['sleep_midpoint'] = (window['sleep_onset'] + duration / 2) % 24
    return window


def create_simple_aggs_sleep(
    phq8: pd.DataFrame,
    sleep: pd.DataFrame,
    sleep_pipeline: dict[str, list[str]],
    window_size: int,
    min_present_fraction: float,
) -> pd.DataFrame:
    return aggregate_stream(phq8, sleep, sleep_pipeline, 'sleep',
                            window_size, min_present_fraction, derive=_derive_sleep)


def create_simple_aggs_activity(
    phq8: pd.DataFrame,
    activity: pd.DataFrame,
    activity_pipeline: dict[str, list[str]],
    window_size: int,
    min_present_fraction: float,
) -> pd.DataFrame:
    return aggregate_stream(phq8, activity, activity_pipeline, 'activity',
                            window_size, min_present_fraction)


def create_simple_aggs_steps(
    phq8: pd.DataFrame,
    steps: pd.DataFrame,
    steps_pipeline: dict[str, list[str]],
    window_size: int,
    min_present_fraction: float,
) -> pd.DataFrame:
    return aggregate_stream(phq8, steps, steps_pipeline, 'steps',
                            window_size, min_present_fraction)


def create_simple_aggs_heartrate(
    phq8: pd.DataFrame,
    heartrate: pd.DataFrame,
    heartrate_pipeline: dict[str, list[str]],
    window_size: int,
    min_present_fraction: float,
) -> pd.DataFrame:
    return aggregate_stream(phq8, heartrate, heartrate_pipeline, 'heartrate',
                            window_size, min_present_fraction)


# ---------------------------------------------------------------------------
# Adherence indicators (optional feature set)
# ---------------------------------------------------------------------------
def create_adherence_sleep(phq8: pd.DataFrame, sleep: pd.DataFrame, window_size: int) -> pd.DataFrame:
    return _compute_adherence_indicators(phq8, sleep, 'sleep', window_size)


def create_adherence_activity(phq8: pd.DataFrame, activity: pd.DataFrame, window_size: int) -> pd.DataFrame:
    return _compute_adherence_indicators(phq8, activity, 'activity', window_size)


def create_adherence_steps(phq8: pd.DataFrame, steps: pd.DataFrame, window_size: int) -> pd.DataFrame:
    return _compute_adherence_indicators(phq8, steps, 'steps', window_size)


def create_adherence_heartrate(phq8: pd.DataFrame, heartrate: pd.DataFrame, window_size: int) -> pd.DataFrame:
    return _compute_adherence_indicators(phq8, heartrate, 'heartrate', window_size)


# ---------------------------------------------------------------------------
# catch22 panel
# ---------------------------------------------------------------------------
def _resolve_panel(panel: list[dict], symptom_tail: str) -> list[str]:
    """Resolve the fixed catch22 panel to concrete feature names for one base
    signal. Direction-sensitive axes pick their variant from `symptom_tail`
    (candidates are ordered [high_variant, low_variant] in the YAML); the
    remaining axes carry a single concrete `feature`."""
    features = []
    for entry in panel:
        if entry.get('select_by') == 'symptom_tail':
            candidates = entry['candidates']
            features.append(candidates[0] if symptom_tail == 'high' else candidates[1])
        else:
            features.append(entry['feature'])
    return features


def _zscore(arr: np.ndarray) -> np.ndarray | None:
    """z-normalize; returns None for a degenerate (zero-variance) series."""
    arr = np.asarray(arr, dtype=float)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return None
    return (arr - np.nanmean(arr)) / sd


def _longest_missing_run(series: np.ndarray) -> int:
    """Longest run of consecutive missing (NaN) days in a chronological series."""
    is_missing = ~np.isfinite(np.asarray(series, dtype=float))
    max_streak = current = 0
    for missing in is_missing:
        current = current + 1 if missing else 0
        if current > max_streak:
            max_streak = current
    return max_streak


def _catch22_panel_values(
    series: np.ndarray,
    feature_names: list[str],
    min_present_fraction: float,
    window_size: int,
) -> list[float]:
    """Compute one base signal's panel over a single window.

    `series` is the window's daily values in chronological order (length
    window_size, NaN for missing days). An all-NaN block is returned (the panel
    is nulled for this window) when EITHER the present fraction is below the
    floor OR the longest run of consecutive missing days exceeds
    MAX_CATCH22_CONSECUTIVE_MISSING -- both signal a series too fabricated to
    read as dynamics. Otherwise the remaining (scattered) missing days are
    interpolated to fix the length, the series is z-normalized, and each panel
    feature is evaluated."""
    series = np.asarray(series, dtype=float)
    present = int(np.isfinite(series).sum())
    if present / window_size < min_present_fraction:
        return [np.nan] * len(feature_names)

    if _longest_missing_run(series) > MAX_CATCH22_CONSECUTIVE_MISSING:
        return [np.nan] * len(feature_names)

    filled = (
        pd.Series(series)
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
        .to_numpy()
    )
    z = _zscore(filled)
    if z is None:
        return [np.nan] * len(feature_names)

    z = z.tolist()
    values = []
    for name in feature_names:
        values.append(float(getattr(pycatch22, name)(z)))
    return values


def create_catch22_aggs(
    phq8: pd.DataFrame,
    sleep: pd.DataFrame,
    activity: pd.DataFrame,
    steps: pd.DataFrame,
    heartrate: pd.DataFrame,
    catch22_cfg: dict,
    window_size: int,
    enabled: bool,
    min_present_fraction: float | None = None,
) -> pd.DataFrame:
    """Apply the fixed, axis-diverse catch22 panel to the configured base
    signals. `enabled` is computed once by the caller from the YAML window
    gating (enabled_window_sizes); when False this returns just the keys so the
    downstream merge is a no-op -- this is the single place that decision is
    made, so it can't drift from the dataset_stem/min_present_fraction-floor
    logic that also depends on it.

    `min_present_fraction` overrides the YAML floor when provided (validated by
    the caller to be no looser than the config floor)."""
    keys = phq8[GROUP_COLS].drop_duplicates().reset_index(drop=True)

    if not enabled:
        return keys.copy()

    if min_present_fraction is None:
        min_present_fraction = catch22_cfg.get('min_present_fraction', 0.75)
    panel = catch22_cfg.get('panel', [])
    base_features = catch22_cfg.get('base_features', [])
    streams = {'sleep': sleep, 'activity': activity, 'steps': steps, 'heartrate': heartrate}

    window = create_window(phq8, window_size)
    result = keys.copy()

    for base in base_features:
        stream_name, col = base['signal'].split('.')
        feature_names = _resolve_panel(panel, base['symptom_tail'])

        stream = streams[stream_name]
        daily = (
            stream[['participant_name', 'timeInterval_ID', col]]
            .rename(columns={'timeInterval_ID': 'timeInterval_ID_window'})
        )
        merged = window.merge(
            daily,
            on=['participant_name', 'timeInterval_ID_window'],
            how='left',
        )
        # Chronological order within each window: largest lag (oldest day) first.
        merged = merged.sort_values(
            GROUP_COLS + ['lag'], ascending=[True, True, False]
        )

        records = []
        for keys_tuple, group in merged.groupby(GROUP_COLS, sort=False):
            values = _catch22_panel_values(
                group[col].to_numpy(), feature_names, min_present_fraction, window_size
            )
            records.append((*keys_tuple, *values))

        out_cols = [f'{stream_name}_{col}_{name}' for name in feature_names]
        block = pd.DataFrame.from_records(records, columns=GROUP_COLS + out_cols)
        result = result.merge(block, on=GROUP_COLS, how='left')

    return result


# ---------------------------------------------------------------------------
# CLI / config parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wearable device data feature aggregation."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        choices=[3, 5, 7, 14, 21, 28],
        default=14,
        help="Window size for the aggregated features. Default: 14.",
    )
    parser.add_argument(
        "--min-present-fraction",
        type=float,
        default=None,
        help=(
            "Single present-fraction floor shared by the simple aggregations and "
            "the catch22 panel: the min fraction of window-days a feature must "
            "have present before its aggregates are nulled (and, for catch22, the "
            "min fraction of real days before the panel is all-NaN). Its allowed "
            "minimum depends on whether catch22 is produced for this window: when "
            "catch22 is present (window in enabled_window_sizes) it must be at "
            "least the catch22 floor in feature_spec.yaml; otherwise it may go as "
            "low as 0.5. Upper bound 1.0 either way. Defaults to 0.75."
        ),
    )
    # Range is validated in main(), where the YAML catch22 floor and this
    # window's catch22 gating are known (the allowed minimum depends on both).
    return parser.parse_args()


def parse_yaml(
    path_to_yaml: str,
    window_size: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict]:
    with open(path_to_yaml, 'r') as file:
        # Use safe_load to prevent executing arbitrary code inside the YAML
        feature_spec = yaml.safe_load(file)
    profiles = feature_spec.get('profiles', {})
    sleep_config = feature_spec.get('sleep', {})
    heartrate_config = feature_spec.get('heartrate', {})
    activity_config = feature_spec.get('activity', {})
    steps_config = feature_spec.get('steps', {})
    catch22_config = feature_spec.get('catch22', {})

    def build_pipeline(label: str, config: dict, with_desc: bool = False) -> dict[str, list[str]]:
        print(f"--- {label} FEATURE PIPELINE MAP ---")
        pipeline = {}
        for feature, cfg in config.items():
            operations = profiles.get(cfg.get('profile'), [])
            pipeline[feature] = operations
            if with_desc:
                print(f"Feature: {feature:<18} ({cfg.get('desc')})  -> Ops: {operations}")
            else:
                print(f"Feature: {feature:<18}  -> Ops: {operations}")
        return pipeline

    sleep_pipeline = build_pipeline("SLEEP", sleep_config)
    heartrate_pipeline = build_pipeline("HEARTRATE", heartrate_config, with_desc=True)
    activity_pipeline = build_pipeline("ACTIVITY", activity_config, with_desc=True)
    steps_pipeline = build_pipeline("STEPS", steps_config, with_desc=True)

    # Only produced (and printed) when this window enables catch22.
    if window_size in catch22_config.get('enabled_window_sizes', []):
        print("--- CATCH22 FEATURE PIPELINE MAP ---")
        panel = catch22_config.get('panel', [])
        for base in catch22_config.get('base_features', []):
            signal = base['signal']
            symptom_tail = base['symptom_tail']
            feature_names = _resolve_panel(panel, symptom_tail)
            print(f"Feature: {signal:<18} (symptom_tail={symptom_tail})  -> Ops: {feature_names}")

    return sleep_pipeline, heartrate_pipeline, activity_pipeline, steps_pipeline, catch22_config


def main() -> None:
    args = parse_args()
    window_size = args.window_size

    start = time.time()

    def log(msg: str) -> None:
        # Progress only: step names + shapes (metadata), never participant rows.
        print(f"[{time.time() - start:6.1f}s] {msg}", flush=True)

    log("Parsing feature spec...")
    (sleep_pipeline, heartrate_pipeline, activity_pipeline,
     steps_pipeline, catch22_config) = parse_yaml(PATH_TO_YAML, window_size)

    # A single present-fraction floor drives both the simple aggregations and the
    # catch22 panel. Its allowed minimum depends on whether catch22 is produced
    # for this window: when the catch22 panel is present, the floor must be at
    # least the (stricter) catch22 floor in feature_spec.yaml, since the same
    # value also caps how much of each series may be interpolated; when catch22 is
    # gated off, the panel imposes no constraint and the floor may go as low as
    # 0.5 (the simple-aggregation minimum). Upper bound is 1.0 either way.
    catch22_enabled = window_size in catch22_config.get('enabled_window_sizes', [])
    catch22_floor = catch22_config.get('min_present_fraction', 0.75)
    lower_bound = catch22_floor if catch22_enabled else 0.5

    min_present_fraction = args.min_present_fraction
    if min_present_fraction is None:
        min_present_fraction = 0.75
    elif not (lower_bound <= min_present_fraction <= 1.0):
        raise SystemExit(
            f"--min-present-fraction must be between {lower_bound} and 1.0 "
            f"(got {min_present_fraction}); the minimum is "
            + (f"the catch22 floor {catch22_floor} because the catch22 panel is "
               f"produced at window {window_size}"
               if catch22_enabled else
               f"0.5 because catch22 is not produced at window {window_size}")
        )

    log(f"Starting merge | window_size={window_size}, "
        f"min_present_fraction={min_present_fraction}, catch22_enabled={catch22_enabled} "
        f"(allowed floor >= {lower_bound})")

    # read cleaned datasets; PHQ8 and the wearable streams remain CSVs, demographics is parquet to preserve categorical metadata
    log("Reading cleaned datasets...")
    phq8 = pd.read_csv(PATH_TO_PHQ8)
    sleep = pd.read_csv(PATH_TO_SLEEP)
    activity = pd.read_csv(PATH_TO_ACTIVITY)
    steps = pd.read_csv(PATH_TO_STEPS)
    heartrate = pd.read_csv(PATH_TO_HEARTRATE)
    demographics = pd.read_parquet(PATH_TO_DEMOGRAPHICS)
    log(f"  done. phq8={phq8.shape}, sleep={sleep.shape}, activity={activity.shape}, "
        f"steps={steps.shape}, heartrate={heartrate.shape}")

    log("Aggregating sleep...")
    simple_sleep_agg = create_simple_aggs_sleep(phq8, sleep, sleep_pipeline, window_size, min_present_fraction)
    log(f"  sleep done -> {simple_sleep_agg.shape}")

    log("Aggregating activity...")
    simple_activity_agg = create_simple_aggs_activity(phq8, activity, activity_pipeline, window_size, min_present_fraction)
    log(f"  activity done -> {simple_activity_agg.shape}")

    log("Aggregating steps...")
    simple_steps_agg = create_simple_aggs_steps(phq8, steps, steps_pipeline, window_size, min_present_fraction)
    log(f"  steps done -> {simple_steps_agg.shape}")

    log("Aggregating heartrate...")
    simple_heartrate_agg = create_simple_aggs_heartrate(phq8, heartrate, heartrate_pipeline, window_size, min_present_fraction)
    log(f"  heartrate done -> {simple_heartrate_agg.shape}")

    log("Aggregating catch22...")
    catch22_agg = create_catch22_aggs(phq8, sleep, activity, steps, heartrate, catch22_config,
                                      window_size, catch22_enabled, min_present_fraction=min_present_fraction)
    log(f"  catch22 done -> {catch22_agg.shape}")

    log("Computing structural adherence counts...")
    adherence_sleep = create_adherence_sleep(phq8, sleep, window_size)
    adherence_activity = create_adherence_activity(phq8, activity, window_size)
    adherence_steps = create_adherence_steps(phq8, steps, window_size)
    adherence_heartrate = create_adherence_heartrate(phq8, heartrate, window_size)
    log(f"  adherence done")

    agg_list = [
        simple_sleep_agg,
        simple_activity_agg,
        simple_steps_agg,
        simple_heartrate_agg,
        catch22_agg,
        adherence_sleep,
        adherence_activity,
        adherence_steps,
        adherence_heartrate,
    ]

    log("Merging stream aggregations...")
    agg = reduce(
        lambda left, right: left.merge(right, on=GROUP_COLS, how="outer"),
        agg_list
    )
    log(f"  merged streams -> {agg.shape}")

    log("Joining PHQ-8 and demographics...")
    final_dataset = phq8.merge(
        agg,
        on=GROUP_COLS,
        how='left'
    ).merge(
        demographics,
        on=['participant_name'],
        how='left'
    )

    final_dataset = final_dataset.drop(columns=["timeInterval_ID"])

    # Check for aggregate columns with no observed values at all.
    all_nan_cols = final_dataset.columns[final_dataset.isna().all()].tolist()
    if all_nan_cols:
        raise SystemExit(
            f"Found {len(all_nan_cols)} all-NaN aggregate columns: {all_nan_cols}\n"
            f"These columns carry no information and should not exist. Check the data pipeline."
        )

    log(f"  all columns have at least one observed value -> {final_dataset.shape}")

    # Resolve family -> columns now (post validation) so both the row filter
    # below and the manifest use the same, final column lists.
    present = set(final_dataset.columns)

    def family_cols(frames: list[pd.DataFrame]) -> list[str]:
        cols = []
        for frame in frames:
            cols += [c for c in frame.columns if c not in GROUP_COLS]
        return [c for c in cols if c in present]

    simple_cols = family_cols([simple_sleep_agg, simple_activity_agg,
                                simple_steps_agg, simple_heartrate_agg])
    catch22_cols = family_cols([catch22_agg])

    # Drop windows (PHQ-8 assessment rows) with zero wearable signal: every
    # simple aggregate AND every catch22 feature is NaN. These rows carry no
    # simple/catch22 information regardless of imputation strategy, so keeping
    # them only pads the row count with pure noise for those two families.
    # (catch22_cols is empty when the panel isn't produced for this window, so
    # the check correctly reduces to "all simple features NaN" there.)
    check_cols = simple_cols + catch22_cols
    if check_cols:
        fully_missing = final_dataset[check_cols].isna().all(axis=1)
        n_dropped = int(fully_missing.sum())
        if n_dropped:
            final_dataset = final_dataset.loc[~fully_missing].reset_index(drop=True)
            log(f"  dropped {n_dropped} rows with all simple+catch22 features NaN")

    log(f"  final dataset -> {final_dataset.shape}")

    # --- column manifest ----------------------------------------------------
    # Emit a sidecar mapping each feature family to the columns it contributed,
    # so modeling.py can select any subset of {demographics, simple, catch22, adherence}
    # without re-deriving the naming convention. This is the producer's ground
    # truth: it reflects all-NaN column validation and catch22 window-gating (an
    # empty list at window sizes where the panel is not produced).
    manifest = {
        "window_size": window_size,
        "min_present_fraction": min_present_fraction,
        "catch22_enabled": catch22_enabled,
        "target": [c for c in ["phq8_score"] if c in present],
        "group": [c for c in ["participant_name"] if c in present],
        "demographics": [c for c in demographics.columns
                         if c != "participant_name" and c in present],
        "simple": simple_cols,
        "catch22": catch22_cols,
        "adherence": family_cols([adherence_sleep, adherence_activity,
                                   adherence_steps, adherence_heartrate]),
    }

    os.makedirs(PATH_TO_MERGED, exist_ok=True)
    stem = dataset_stem(window_size, min_present_fraction, catch22_enabled)
    output_path = f"{PATH_TO_MERGED}/{stem}.parquet"
    final_dataset.to_parquet(output_path, engine='pyarrow', index=False)
    log(f"Wrote {output_path}")

    manifest_path = f"{PATH_TO_MERGED}/{stem}_columns.json"
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    log(f"Wrote {manifest_path} "
        f"(demographics={len(manifest['demographics'])}, "
        f"simple={len(manifest['simple'])}, catch22={len(manifest['catch22'])})")


if __name__ == "__main__":
    main()
