import argparse
import os
import time
import pandas as pd
import numpy as np
from functools import reduce

PATH_TO_PHQ8 = "../data_cleaned/phq8_data.csv"
PATH_TO_SLEEP = "../data_cleaned/sleep_data.csv"
PATH_TO_ACTIVITY = "../data_cleaned/activity_data.csv"
PATH_TO_STEPS = "../data_cleaned/steps_data.csv"
PATH_TO_HEARTRATE = "../data_cleaned/heartrate_data.csv"
PATH_TO_DEMOGRAPHICS = "../data_cleaned/totaldataset.parquet"
PATH_TO_MERGED = "../data_merged"

SIX_HOURS_SECONDS = 6 * 60 * 60

GROUP_COLS = ['participant_name', 'timeInterval_ID']


def create_window(phq8, window_size):
    window = phq8[GROUP_COLS].copy()

    window = window.loc[window.index.repeat(window_size)].copy()
    window['lag'] = list(range(1, window_size+1)) * len(phq8)

    # Compute the prior day IDs
    window['timeInterval_ID_window'] = window['timeInterval_ID'] - window['lag']
    return window


def circular_mean(series):
    """Approximate circular mean for time-of-day in hours."""
    clean_series = series.dropna()
    if clean_series.empty:
        return np.nan
    
    radians = (clean_series % 24) / 24 * 2 * np.pi
    sin_sum = np.sin(radians).mean()
    cos_sum = np.cos(radians).mean()
    
    vector_length = np.sqrt(sin_sum**2 + cos_sum**2)
    
    # Check if the vector length is 0 (or practically 0 due to computer precision)
    if np.isclose(vector_length, 0, atol=1e-5):
        return np.nan
    else:
        mean_angle = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
        
    return (mean_angle / (2 * np.pi)) * 24


def circular_std(series):
    """Approximate circular std for time-of-day in hours safely."""
    clean_series = series.dropna()
    if clean_series.empty:
        return np.nan
    radians = (clean_series % 24)  / 24 * 2 * np.pi
    sin_mean = np.sin(radians).mean()
    cos_mean = np.cos(radians).mean()
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    # Clip R to avoid floating point errors breaking log domains
    R = np.clip(R, 1e-9, 1.0)
    val = -2 * np.log(R)
    # Ensure value isn't slightly negative due to precision issues
    val = max(0.0, val)
    return np.sqrt(val) * 24 / (2 * np.pi)


def circular_median(series):
    """Circular median for time-of-day in hours.
    Returns the observed value that minimises total arc distance to all others.
    Robust to outlier nights; brute-force is fine for window sizes 3-14."""
    clean = series.dropna() % 24
    if clean.empty:
        return np.nan
    if len(clean) == 1:
        return clean.iloc[0]
    angles = clean.values / 24 * 2 * np.pi
    def arc_dist(a, b):
        d = abs(a - b) % (2 * np.pi)
        return min(d, 2 * np.pi - d)
    total_dists = [sum(arc_dist(a, b) for b in angles) for a in angles]
    return clean.iloc[int(np.argmin(total_dists))]

def calculate_sleep_duration(onset, offset, max_val=24):
    return (offset - onset + max_val) % max_val

def streak_length(series):
    """Max consecutive run of 1s.
 
    A missing day (NaN) breaks the streak: it is treated as a reset rather than
    being dropped, so e.g. [1, NaN, 1] yields a max streak of 1, not 2.
    Returns NaN only if the window contains no observed values at all.
    """
    arr = series.values
    if series.dropna().empty:
        return np.nan
    max_streak = 0
    current = 0
    for v in arr:
        if v == 1:
            current += 1
            max_streak = max(max_streak, current)
        else:
            # NaN (missing day) or any non-1 value resets the streak
            current = 0
    return max_streak

def create_simple_aggs_heartrate(phq8, heartrate, window_size, missing_threshold):
    window = create_window(phq8, window_size)
    heartrate = heartrate.rename(columns={'timeInterval_ID': 'timeInterval_ID_heartrate'})

    # Every non-key column of the heartrate table (FEATURE_*, QUALITY_METRIC_*,
    # MISSING_RATE). Heartrate has no time-of-day columns (nothing circular) and
    # no additive counts/durations (nothing to sum): every feature is a HR level,
    # HRV metric, moment, difference, or complexity measure. So every column gets
    # the same simple aggregations.
    value_cols = [c for c in heartrate.columns
                  if c not in ('participant_name', 'timeInterval_ID_heartrate')]

    window = window.merge(
        heartrate,
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_heartrate'],
        how='left'
    )

    # --- per-column missing-day counts (used internally for nulling) -------
    missing_flags = window[GROUP_COLS + value_cols].copy()
    missing_flags[value_cols] = missing_flags[value_cols].isna().astype(int)
    missing_counts = missing_flags.groupby(GROUP_COLS)[value_cols].sum()
    missing_counts.columns = [f'{c}_missing_last{window_size}' for c in value_cols]
    missing_counts = missing_counts.reset_index()

    # --- structural (day-level) missing count, kept as a feature -----------
    # Days with no heartrate row at all = device non-wear / absence.
    window['_row_absent'] = window['timeInterval_ID_heartrate'].isna().astype(int)
    structural_counts = (
        window.groupby(GROUP_COLS)['_row_absent']
        .sum()
        .reset_index(name=f'heartrate_missing_last{window_size}')
    )

    # --- simple aggregations per column ------------------------------------
    funcs = ['min', 'max', 'mean', 'median', 'std']
    agg = window.groupby(GROUP_COLS)[value_cols].agg(funcs)
    agg.columns = [f'{col}_{stat}' for col, stat in agg.columns]
    agg = agg.reset_index()

    agg = (
        agg
        .merge(missing_counts, on=GROUP_COLS, how='left')
        .merge(structural_counts, on=GROUP_COLS, how='left')
    )

    # --- per-column nulling ------------------------------------------------
    for col in value_cols:
        invalid = agg[f'{col}_missing_last{window_size}'] > missing_threshold
        derived = [c for c in agg.columns
                   if c.startswith(f'{col}_')
                   and not c.endswith(f'_missing_last{window_size}')]
        agg.loc[invalid.values, derived] = np.nan

    # Drop the per-column counts; keep only the single structural count.
    per_col_counts = [f'{c}_missing_last{window_size}' for c in value_cols]
    agg = agg.drop(columns=per_col_counts)

    # Prefix every feature column with the stream name to avoid collisions with
    # the other streams' identically-named FEATURE_*/QUALITY_*/MISSING_RATE
    # columns when merged in main(). The structural count is already stream-named.
    agg = agg.rename(columns={c: f'heartrate_{c}' for c in agg.columns
                              if c not in GROUP_COLS and not c.startswith('heartrate_')})

    return agg

def create_simple_aggs_steps(phq8, steps, window_size, missing_threshold):
    window = create_window(phq8, window_size)
    steps = steps.rename(columns = {'timeInterval_ID': 'timeInterval_ID_steps'})

    # FEATURE_027-038 are the BOUT DEFINITION 1 features; bout definition 2
    # (FEATURE_039-050) is kept instead, so no aggregations are computed on
    # bout def 1 (excluded from value_cols below, and from sum_cols further down).
    bout_def1_cols = [f"FEATURE_{i:03d}" for i in range(27, 39)]

    # Every non-key column of the steps table (FEATURE_*, QUALITY_METRIC_*,
    # MISSING_RATE) except bout definition 1. Counts are persisted for all of
    # these; simple aggregations are computed for all of them too.
    value_cols = [c for c in steps.columns
                  if c not in ('participant_name', 'timeInterval_ID_steps')
                  and c not in bout_def1_cols]

    window = window.merge(
        steps,
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_steps'],
        how='left'
    )

    # --- per-column missing-day counts (used internally for nulling) --------
    missing_flags = window[GROUP_COLS + value_cols].copy()
    missing_flags[value_cols] = missing_flags[value_cols].isna().astype(int)
    missing_counts = missing_flags.groupby(GROUP_COLS)[value_cols].sum()
    missing_counts.columns = [f'{c}_missing_last{window_size}' for c in value_cols]
    missing_counts = missing_counts.reset_index()

    # --- structural (day-level) missing count, kept as a feature -------------
    window['_row_absent'] = window['timeInterval_ID_steps'].isna().astype(int)
    structural_counts = (
        window.groupby(GROUP_COLS)['_row_absent']
        .sum()
        .reset_index(name=f'steps_missing_last{window_size}')
    )   

    # --- simple aggregations per column ------------------------------------
    # FEATURE_022, 024, 025, 026 are times-of-day (hours), handled separately
    # below with circular stats; every other column gets linear stats.
    time_of_day_cols = ["FEATURE_022", "FEATURE_024", "FEATURE_025", "FEATURE_026"]
    funcs = ['min', 'max', 'mean', 'median', 'std']
    cols = [col for col in value_cols if col not in time_of_day_cols]
    agg = window.groupby(GROUP_COLS)[cols].agg(funcs)
    agg.columns = [f'{col}_{stat}' for col, stat in agg.columns]
    agg = agg.reset_index()

    ## for some columns, sum is an appropriate aggregation as well
    ## steps totals: FEATURE_001 (daily total), FEATURE_019 (nighttime),
    ## FEATURE_039, 045 (bout def 2 counts). Bout def 1 counts (FEATURE_027, 033)
    ## are excluded.
    sum_cols = ["FEATURE_001", "FEATURE_019", "FEATURE_039", "FEATURE_045"]
    agg_sum = window.groupby(GROUP_COLS)[sum_cols].agg("sum")
    agg_sum.columns = [f'{col}_sum' for col in sum_cols]
    agg_sum = agg_sum.reset_index()

    ## Time-of-day columns: circular mean/median/std, linear min/max
    ## FEATURE_022: most active hour; FEATURE_024/025/026: percentile hours
    feature_timecols_agg = (
        window.groupby(GROUP_COLS)[time_of_day_cols]
        .agg({
            'FEATURE_022': ['min', 'max', circular_mean, circular_median, circular_std],
            'FEATURE_024': ['min', 'max', circular_mean, circular_median, circular_std],
            'FEATURE_025': ['min', 'max', circular_mean, circular_median, circular_std],
            'FEATURE_026': ['min', 'max', circular_mean, circular_median, circular_std],
        })
    )
    # Strip the "circular_" prefix so the columns match the _mean/_median/_std
    # convention used by the sleep timing block and activity's FEATURE_018.
    feature_timecols_agg.columns = [
        f'{col}_{(stat.__name__ if callable(stat) else stat).replace("circular_", "")}'
        for col, stat in feature_timecols_agg.columns
    ]
    feature_timecols_agg = feature_timecols_agg.reset_index()

    agg = (
        agg
        .merge(agg_sum, on=GROUP_COLS, how='left')
        .merge(feature_timecols_agg, on=GROUP_COLS, how='left')
        .merge(missing_counts, on=GROUP_COLS, how='left')
        .merge(structural_counts, on=GROUP_COLS, how='left')
    )

    # --- per-column nulling ------------------------------------------------
    for col in value_cols:
        invalid = agg[f'{col}_missing_last{window_size}'] > missing_threshold
        derived = [c for c in agg.columns
                   if c.startswith(f'{col}_')
                   and not c.endswith(f'_missing_last{window_size}')]
        agg.loc[invalid.values, derived] = np.nan

    # Drop the per-column counts; keep only the single structural count.
    per_col_counts = [f'{c}_missing_last{window_size}' for c in value_cols]
    agg = agg.drop(columns=per_col_counts)

    # Prefix every feature column with the stream name so the FEATURE_*/
    # QUALITY_*/MISSING_RATE columns don't collide with the activity stream's
    # identically-named columns when merged in main(). The structural count
    # (steps_missing_last{N}) is already stream-named, so it is left as-is.
    agg = agg.rename(columns={c: f'steps_{c}' for c in agg.columns
                              if c not in GROUP_COLS and not c.startswith('steps_')})

    return agg

def create_simple_aggs_activity(phq8, activity, window_size, missing_threshold):
    window = create_window(phq8, window_size)
    activity = activity.rename(columns = {'timeInterval_ID': 'timeInterval_ID_activity'})

    # Every non-key column of the activity table (FEATURE_*, QUALITY_METRIC_*,
    # MISSING_RATE). Counts are persisted for all of these; simple aggregations
    # are computed for all of them too.
    value_cols = [c for c in activity.columns
                  if c not in ('participant_name', 'timeInterval_ID_activity')]

    window = window.merge(
        activity,
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_activity'],
        how='left'
    )

    # --- per-column missing-day counts (used internally for nulling) -------
    # Unlike sleep (row-level missingness, one count suffices), activity columns
    # go missing independently: a window-day's row may be absent (structural ->
    # every column NaN) or a single feature may be NaN within a present row
    # (value-level). A per-column NaN count over the window folds both into one
    # number per column. These drive per-column nulling but are dropped before
    # returning (they are heavily collinear and partly structural, so they are
    # not exposed as model features).
    missing_flags = window[GROUP_COLS + value_cols].copy()
    missing_flags[value_cols] = missing_flags[value_cols].isna().astype(int)
    missing_counts = missing_flags.groupby(GROUP_COLS)[value_cols].sum()
    missing_counts.columns = [f'{c}_missing_last{window_size}' for c in value_cols]
    missing_counts = missing_counts.reset_index()

    # --- structural (day-level) missing count, kept as a feature -----------
    # Days in the window with no activity row at all (every column NaN), i.e.
    # device non-wear / absence. One interpretable adherence signal, in contrast
    # to the 23 collinear per-column counts.
    window['_row_absent'] = window['timeInterval_ID_activity'].isna().astype(int)
    structural_counts = (
        window.groupby(GROUP_COLS)['_row_absent']
        .sum()
        .reset_index(name=f'activity_missing_last{window_size}')
    )

    # --- simple aggregations per column ------------------------------------
    # FEATURE_018 ("most active hour") is an hour-of-day, handled separately
    # below with circular stats; every other column gets linear stats.
    funcs = ['min', 'max', 'mean', 'median', 'std']
    cols = [col for col in value_cols if col != 'FEATURE_018']
    agg = window.groupby(GROUP_COLS)[cols].agg(funcs)
    agg.columns = [f'{col}_{stat}' for col, stat in agg.columns]
    agg = agg.reset_index()

    ## for some columns, sum is an appropriate aggregation as well
    sum_cols = ["FEATURE_001", "FEATURE_002", "FEATURE_003", "FEATURE_004", "FEATURE_005",
                "FEATURE_013", "FEATURE_014", "FEATURE_016",]
    agg_sum = window.groupby(GROUP_COLS)[sum_cols].agg("sum")
    agg_sum.columns = [f'{col}_sum' for col in sum_cols]
    agg_sum = agg_sum.reset_index()

    ## FEATURE_018 ("most active hour") is a time-of-day in hours (0-23), which
    ## is circular (hour 23 is adjacent to hour 0): use circular mean/median/std.
    ## min/max stay linear (same midnight-wrap caveat as the sleep timing block).
    feature_018_agg = (
        window.groupby(GROUP_COLS)['FEATURE_018']
        .agg(
            FEATURE_018_min='min',
            FEATURE_018_max='max',
            FEATURE_018_mean=circular_mean,
            FEATURE_018_median=circular_median,
            FEATURE_018_std=circular_std,
        )
        .reset_index()
    )

    agg = (
        agg
        .merge(agg_sum, on=GROUP_COLS, how='left')
        .merge(feature_018_agg, on=GROUP_COLS, how='left')
        .merge(missing_counts, on=GROUP_COLS, how='left')
        .merge(structural_counts, on=GROUP_COLS, how='left')
    )

    # --- per-column nulling ------------------------------------------------
    # Null only the aggregates of the feature that exceeded the threshold,
    # rather than the whole window (as sleep does), so a window with good data
    # for most features keeps them even when one sparse column is over budget.
    for col in value_cols:
        invalid = agg[f'{col}_missing_last{window_size}'] > missing_threshold
        derived = [c for c in agg.columns
                   if c.startswith(f'{col}_')
                   and not c.endswith(f'_missing_last{window_size}')]
        agg.loc[invalid.values, derived] = np.nan

    # Drop the per-column counts now that nulling is done; keep only the single
    # structural adherence count as a feature.
    per_col_counts = [f'{c}_missing_last{window_size}' for c in value_cols]
    agg = agg.drop(columns=per_col_counts)

    # Prefix every feature column with the stream name so the FEATURE_*/
    # QUALITY_*/MISSING_RATE columns don't collide with the steps stream's
    # identically-named columns when merged in main(). The structural count
    # (activity_missing_last{N}) is already stream-named, so it is left as-is.
    agg = agg.rename(columns={c: f'activity_{c}' for c in agg.columns
                              if c not in GROUP_COLS and not c.startswith('activity_')})

    return agg


def create_simple_aggs_sleep(phq8, sleep, window_size, missing_threshold):
    window = create_window(phq8, window_size)

    sleep = sleep.rename(columns = {'timeInterval_ID': 'timeInterval_ID_sleep'})
    sleep['observed'] = 1

    window = window.merge(
        sleep,
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_sleep'],
        how='left'
    )

    window['missing'] = window['observed'].isna().astype(int)

    missing_counts = (
        window.groupby(GROUP_COLS)['missing']
        .sum()
        .reset_index(name=f'sleep_missing_last{window_size}')
    )

    
    """ total_sleep_time """
    funcs_total_sleep_time = ['min', 'max', 'mean', 'median', 'std', 'sum']
    total_sleep_time_agg = window.groupby(GROUP_COLS)[["total_sleep_time"]].agg(funcs_total_sleep_time)
    total_sleep_time_agg.columns = [f"{col}_{stat}" for col, stat in total_sleep_time_agg.columns]

    custom_features = (
        window.groupby(GROUP_COLS)["total_sleep_time"]
        .agg(
            total_sleep_time_pct_short_sleep=lambda x: (x.dropna() < SIX_HOURS_SECONDS).mean(),
        )
        .reset_index()
    )
    total_sleep_time_agg = total_sleep_time_agg.reset_index().merge(custom_features, on=GROUP_COLS, how="left")
    

    """ time_in_bed """
    funcs_time_in_bed_agg = ['min', 'max', 'mean', 'median', 'std', 'sum']
    time_in_bed_agg = window.groupby(GROUP_COLS)[["time_in_bed"]].agg(funcs_time_in_bed_agg)
    time_in_bed_agg.columns = [f"{col}_{stat}" for col, stat in time_in_bed_agg.columns]
    time_in_bed_agg = time_in_bed_agg.reset_index()


    """ sleep stage composition features """
    sleep_stage_cols = ["light_pct", "deep_pct", "REM_pct", "awake_pct", "NREM_pct"]
    sleep_stage_funcs = ["min", "max", "mean", "median", "std"]
    sleep_stage_agg = window.groupby(GROUP_COLS)[sleep_stage_cols].agg(sleep_stage_funcs)
    sleep_stage_agg.columns = [f"{col}_{stat}" for col, stat in sleep_stage_agg.columns]

    window['rem_deep_ratio'] = window["REM_pct"] / window["deep_pct"].replace(0, np.nan)
    window['light_restorative_ratio'] = window["light_pct"] / (window["deep_pct"] + window["REM_pct"]).replace(0, np.nan)

    sleep_stage_custom = (
        window.groupby(GROUP_COLS)
        .agg(
            rem_deep_ratio_min=('rem_deep_ratio', 'min'),
            rem_deep_ratio_max=('rem_deep_ratio', 'max'),
            rem_deep_ratio_mean=('rem_deep_ratio', 'mean'),
            rem_deep_ratio_median=('rem_deep_ratio', 'median'),
            rem_deep_ratio_std=('rem_deep_ratio', 'std'),
            light_restorative_ratio_min=('light_restorative_ratio', 'min'),
            light_restorative_ratio_max=('light_restorative_ratio', 'max'),
            light_restorative_ratio_mean=('light_restorative_ratio', 'mean'),
            light_restorative_ratio_median=('light_restorative_ratio', 'median'),
            light_restorative_ratio_std=('light_restorative_ratio', 'std'),
        )
        .reset_index()
    )
    sleep_stage_agg = sleep_stage_agg.reset_index().merge(sleep_stage_custom, on=GROUP_COLS, how="left")

    
    """ sleep_onset / sleep_offset """
    sleep_timing_cols = ["sleep_onset", "sleep_offset"]
    sleep_timing_agg = window.groupby(GROUP_COLS)[sleep_timing_cols].agg(["min", "max"])
    sleep_timing_agg.columns = [f"{col}_{stat}" for col, stat in sleep_timing_agg.columns]
    sleep_timing_agg = sleep_timing_agg.reset_index()

    # Pre-calculate midpoints and timing differences row-wise
    window['time_in_bed_diff'] = calculate_sleep_duration(window["sleep_onset"], window["sleep_offset"])
    window['sleep_midpoint'] = (window["sleep_onset"] + ( window['time_in_bed_diff'] / 2)) % 24

    sleep_timing_custom = (
        window.groupby(GROUP_COLS)
        .agg(
            sleep_onset_mean=('sleep_onset', circular_mean),
            sleep_onset_median=('sleep_onset', circular_median),
            sleep_onset_std=('sleep_onset', circular_std),
            sleep_offset_mean=('sleep_offset', circular_mean),
            sleep_offset_median=('sleep_offset', circular_median),
            sleep_offset_std=('sleep_offset', circular_std),
            sleep_midpoint_min=('sleep_midpoint', 'min'),
            sleep_midpoint_max=('sleep_midpoint', 'max'),
            sleep_midpoint_mean=('sleep_midpoint', circular_mean),
            sleep_midpoint_median=('sleep_midpoint', circular_median),
            sleep_midpoint_std=('sleep_midpoint', circular_std),
        )
        .reset_index()
    )
    sleep_timing_custom['sleep_phase_instability'] = sleep_timing_custom['sleep_onset_std'] + sleep_timing_custom['sleep_offset_std']
    sleep_timing_agg = sleep_timing_agg.merge(sleep_timing_custom, on=GROUP_COLS, how="left")

    
    """ rem_latency """
    funcs_rem_latency = ["min", "max", "mean", "median", "std", "sum"]
    rem_latency_agg = window.groupby(GROUP_COLS)[["rem_latency"]].agg(funcs_rem_latency)
    rem_latency_agg.columns = [f"{col}_{stat}" for col, stat in rem_latency_agg.columns]
    rem_latency_agg = rem_latency_agg.reset_index()

    rem_latency_custom = (
        window.groupby(GROUP_COLS)["rem_latency"]
        .agg(
            rem_latency_pct_high=lambda x: (x.dropna() > 2).mean(),
            rem_latency_pct_very_high=lambda x: (x.dropna() > 3).mean(),
            rem_latency_range=lambda x: x.max() - x.min(),
        )
        .reset_index()
    )
    rem_latency_agg = rem_latency_agg.merge(rem_latency_custom, on=GROUP_COLS, how="left")

    """ sleep_efficiency """
    funcs_sleep_efficiency = ['min', 'max', 'mean', 'median', 'std']
    sleep_efficiency_agg = window.groupby(GROUP_COLS)[["sleep_efficiency"]].agg(funcs_sleep_efficiency)
    sleep_efficiency_agg.columns = [f"{col}_{stat}" for col, stat in sleep_efficiency_agg.columns]
    sleep_efficiency_agg = sleep_efficiency_agg.reset_index()
    
    """ awakenings """
    funcs_awakenings = ["min", "max", "mean", "median", "std", "sum"]
    awakenings_agg = window.groupby(GROUP_COLS)[["awakenings"]].agg(funcs_awakenings)
    awakenings_agg.columns = [f"{col}_{stat}" for col, stat in awakenings_agg.columns]
    awakenings_agg = awakenings_agg.reset_index()

    awakenings_custom = (
        window.groupby(GROUP_COLS)["awakenings"]
        .agg(
            awakenings_pct_any=lambda x: (x.dropna() > 0).mean(),
            awakenings_pct_mod=lambda x: (x.dropna() >= 2).mean(),
            awakenings_pct_high=lambda x: (x.dropna() >= 3).mean(),
            awakenings_range=lambda x: x.max() - x.min(),
        )
        .reset_index()
    )
    awakenings_agg = awakenings_agg.merge(awakenings_custom, on=GROUP_COLS, how="left")

    
    """ insomnia / hypersomnia """
    binary_cols = ["insomnia", "hypersomnia"]
    funcs_binary = ["sum", "mean", "max"]
    disorder_agg = window.groupby(GROUP_COLS)[binary_cols].agg(funcs_binary)
    disorder_agg.columns = [f"{col}_{stat}" for col, stat in disorder_agg.columns]
    disorder_agg = disorder_agg.reset_index()

    # Pre-calculate row-wise co-occurrence 
    window['any_disorder_flag'] = ((window['insomnia'] == 1) | (window['hypersomnia'] == 1)).astype(float)
    # Mask NaNs so missing days don't falsely register as "0 disorders"
    window.loc[window['insomnia'].isna() & window['hypersomnia'].isna(), 'any_disorder_flag'] = np.nan

    disorder_custom = (
        window.groupby(GROUP_COLS)
        .agg(
            insomnia_max_streak=('insomnia', streak_length),
            hypersomnia_max_streak=('hypersomnia', streak_length),
            any_disorder=('any_disorder_flag', 'max')
        )
        .reset_index()
    )
    disorder_agg = disorder_agg.merge(disorder_custom, on=GROUP_COLS, how="left")

    # Master Merge
    agg_list = [
        total_sleep_time_agg,
        time_in_bed_agg,
        sleep_efficiency_agg,
        sleep_stage_agg,
        sleep_timing_agg,
        rem_latency_agg,
        awakenings_agg,
        disorder_agg
    ]

    agg = reduce(
        lambda left, right: left.merge(right, on=GROUP_COLS, how="outer"),
        agg_list
    )

    sleep_stats_agg = agg.merge(
        missing_counts,
        on=GROUP_COLS,
        how='left'
    )

    invalid_mask = sleep_stats_agg[f'sleep_missing_last{window_size}'] > missing_threshold
    stat_cols = [col for col in agg.columns if col not in GROUP_COLS]
    sleep_stats_agg.loc[invalid_mask, stat_cols] = np.nan

    return sleep_stats_agg

def parse_args():
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
        "--missing-threshold",
        type=int,
        default=None,
        help=(
            "Max missing days allowed before nulling a window. "
            "Must be between ceil(window_size/2) and window_size (inclusive). "
            "Defaults to ceil(window_size/2)."
        ),
    )
    args = parser.parse_args()

    lower = (args.window_size + 1) // 2
    upper = args.window_size
    if args.missing_threshold is None:
        args.missing_threshold = lower
    elif not (lower <= args.missing_threshold <= upper):
        parser.error(
            f"--missing-threshold must be between {lower} and {upper} "
            f"for --window-size {args.window_size} (got {args.missing_threshold})"
        )

    return args

def main():
    args = parse_args()
    window_size = args.window_size
    missing_threshold = args.missing_threshold

    start = time.time()
    def log(msg):
        # Progress only: step names + shapes (metadata), never participant rows.
        print(f"[{time.time() - start:6.1f}s] {msg}", flush=True)

    log(f"Starting merge | window_size={window_size}, missing_threshold={missing_threshold}")

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
    simple_sleep_agg = create_simple_aggs_sleep(phq8, sleep, window_size, missing_threshold)
    log(f"  sleep done -> {simple_sleep_agg.shape}")

    log("Aggregating activity...")
    simple_activity_agg = create_simple_aggs_activity(phq8, activity, window_size, missing_threshold)
    log(f"  activity done -> {simple_activity_agg.shape}")

    log("Aggregating steps...")
    simple_steps_agg = create_simple_aggs_steps(phq8, steps, window_size, missing_threshold)
    log(f"  steps done -> {simple_steps_agg.shape}")

    log("Aggregating heartrate...")
    simple_heartrate_agg = create_simple_aggs_heartrate(phq8, heartrate, window_size, missing_threshold)
    log(f"  heartrate done -> {simple_heartrate_agg.shape}")

    agg_list = [
        simple_sleep_agg,
        simple_activity_agg,
        simple_steps_agg,
        simple_heartrate_agg,
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
         on = ['participant_name'],
         how = 'left'
    )

    final_dataset = final_dataset.drop(columns=["timeInterval_ID"])

    # Drop aggregate columns with no observed values at all: they carry no
    # information and trip per-fold imputers downstream.
    all_nan_cols = final_dataset.columns[final_dataset.isna().all()].tolist()
    if all_nan_cols:
        final_dataset = final_dataset.drop(columns=all_nan_cols)
        log(f"  dropped {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")

    log(f"  final dataset -> {final_dataset.shape}")

    os.makedirs(PATH_TO_MERGED, exist_ok=True)
    output_path = f"{PATH_TO_MERGED}/final_dataset_w{window_size}.parquet"
    final_dataset.to_parquet(output_path, engine='pyarrow', index = False)
    log(f"Wrote {output_path}")
    
if __name__ == "__main__":
    main()

