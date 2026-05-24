import pandas as pd
import numpy as np
from functools import reduce

PATH_TO_PHQ8 = "../data_cleaned/phq8_data.csv"
PATH_TO_SLEEP = "../data_cleaned/sleep_data.csv"
PATH_TO_DEMOGRAPHICS = "../data_cleaned/totaldataset.parquet"
PATH_TO_FINAL_DATASET = "../data_cleaned/final_dataset.parquet"

SIX_HOURS_SECONDS = 6 * 60 * 60


def create_window(phq8, window_size):
    window = phq8[['participant_name', 'timeInterval_ID']].copy()

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


def merge_with_sleep(phq8, sleep, window_size, missing_threshold):
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
        window.groupby(['participant_name', 'timeInterval_ID'])['missing']
        .sum()
        .reset_index(name=f'sleep_missing_last{window_size}')
    )

    group_cols = ['participant_name', 'timeInterval_ID']

    
    """ total_sleep_time """
    funcs_total_sleep_time = ['min', 'max', 'mean', 'median', 'std', 'sum']
    total_sleep_time_agg = window.groupby(group_cols)[["total_sleep_time"]].agg(funcs_total_sleep_time)
    total_sleep_time_agg.columns = [f"{col}_{stat}" for col, stat in total_sleep_time_agg.columns]

    custom_features = (
        window.groupby(group_cols)["total_sleep_time"]
        .agg(
            total_sleep_time_pct_short_sleep=lambda x: (x < SIX_HOURS_SECONDS).mean(),
            total_sleep_time_cv=lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan,
        )
        .reset_index()
    )
    total_sleep_time_agg = total_sleep_time_agg.reset_index().merge(custom_features, on=group_cols, how="left")
    

    """ time_in_bed """
    funcs_time_in_bed_agg = ['min', 'max', 'mean', 'median', 'std', 'sum']
    time_in_bed_agg = window.groupby(group_cols)[["time_in_bed"]].agg(funcs_time_in_bed_agg)
    time_in_bed_agg.columns = [f"{col}_{stat}" for col, stat in time_in_bed_agg.columns]

    time_in_bed_custom = (
        window.groupby(group_cols)["time_in_bed"]
        .agg(time_in_bed_cv=lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
        .reset_index()
    )
    time_in_bed_agg = time_in_bed_agg.reset_index().merge(time_in_bed_custom, on=group_cols, how="left")

    
    """ sleep stage composition features """
    sleep_stage_cols = ["light_pct", "deep_pct", "REM_pct", "awake_pct", "NREM_pct"]
    sleep_stage_funcs = ["min", "max", "mean", "median", "std"]
    sleep_stage_agg = window.groupby(group_cols)[sleep_stage_cols].agg(sleep_stage_funcs)
    sleep_stage_agg.columns = [f"{col}_{stat}" for col, stat in sleep_stage_agg.columns]

    window['rem_deep_ratio'] = window["REM_pct"] / window["deep_pct"].replace(0, np.nan)
    window['light_restorative_ratio'] = window["light_pct"] / (window["deep_pct"] + window["REM_pct"]).replace(0, np.nan)

    sleep_stage_custom = (
        window.groupby(group_cols)
        .agg(
            rem_deep_ratio_mean=('rem_deep_ratio', 'mean'),
            light_restorative_ratio_mean=('light_restorative_ratio', 'mean')
        )
        .reset_index()
    )
    sleep_stage_agg = sleep_stage_agg.reset_index().merge(sleep_stage_custom, on=group_cols, how="left")

    
    """ sleep_onset / sleep_offset """
    sleep_timing_cols = ["sleep_onset", "sleep_offset"]
    sleep_timing_agg = window.groupby(group_cols)[sleep_timing_cols].agg(["min", "max"])
    sleep_timing_agg.columns = [f"{col}_{stat}" for col, stat in sleep_timing_agg.columns]
    sleep_timing_agg = sleep_timing_agg.reset_index()

    # Pre-calculate midpoints and timing differences row-wise
    window['time_in_bed_diff'] = calculate_sleep_duration(window["sleep_onset"], window["sleep_offset"])
    window['sleep_midpoint'] = (window["sleep_onset"] + ( window['time_in_bed_diff'] / 2)) % 24

    sleep_timing_custom = (
        window.groupby(group_cols)
        .agg(
            sleep_onset_mean=('sleep_onset', circular_mean),
            sleep_offset_mean=('sleep_offset', circular_mean),
            sleep_onset_std=('sleep_onset', circular_std),
            sleep_offset_std=('sleep_offset', circular_std),
            avg_time_in_bed_window=('time_in_bed_diff', 'mean'),
            sleep_midpoint_mean=('sleep_midpoint', circular_mean),
            sleep_midpoint_std=('sleep_midpoint', circular_std),
        )
        .reset_index()
    )
    sleep_timing_custom['sleep_phase_instability'] = sleep_timing_custom['sleep_onset_std'] + sleep_timing_custom['sleep_offset_std']
    sleep_timing_agg = sleep_timing_agg.merge(sleep_timing_custom, on=group_cols, how="left")

    
    """ rem_latency """
    funcs_rem_latency = ["min", "max", "mean", "median", "std", "sum"]
    rem_latency_agg = window.groupby(group_cols)[["rem_latency"]].agg(funcs_rem_latency)
    rem_latency_agg.columns = [f"{col}_{stat}" for col, stat in rem_latency_agg.columns]
    rem_latency_agg = rem_latency_agg.reset_index()

    rem_latency_custom = (
        window.groupby(group_cols)["rem_latency"]
        .agg(
            rem_latency_cv=lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan,
            rem_latency_pct_high=lambda x: (x.dropna() > 2).mean(),
            rem_latency_pct_very_high=lambda x: (x.dropna() > 3).mean(),
            rem_latency_range=lambda x: x.max() - x.min(),
        )
        .reset_index()
    )
    rem_latency_agg = rem_latency_agg.merge(rem_latency_custom, on=group_cols, how="left")

    
    """ awakenings """
    funcs_awakenings = ["min", "max", "mean", "median", "std", "sum"]
    awakenings_agg = window.groupby(group_cols)[["awakenings"]].agg(funcs_awakenings)
    awakenings_agg.columns = [f"{col}_{stat}" for col, stat in awakenings_agg.columns]
    awakenings_agg = awakenings_agg.reset_index()

    awakenings_custom = (
        window.groupby(group_cols)["awakenings"]
        .agg(
            awakenings_cv=lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan,
            awakenings_pct_any=lambda x: (x.dropna() > 0).mean(),
            awakenings_pct_mod=lambda x: (x.dropna() >= 2).mean(),
            awakenings_pct_high=lambda x: (x.dropna() >= 3).mean(),
            awakenings_worst_night=lambda x: x.max(),
            awakenings_range=lambda x: x.max() - x.min(),
        )
        .reset_index()
    )
    awakenings_agg = awakenings_agg.merge(awakenings_custom, on=group_cols, how="left")

    
    """ insomnia / hypersomnia """
    binary_cols = ["insomnia", "hypersomnia"]
    funcs_binary = ["sum", "mean", "max"]
    disorder_agg = window.groupby(group_cols)[binary_cols].agg(funcs_binary)
    disorder_agg.columns = [f"{col}_{stat}" for col, stat in disorder_agg.columns]
    disorder_agg = disorder_agg.reset_index()

    # Pre-calculate row-wise co-occurrence 
    window['any_disorder_flag'] = ((window['insomnia'] == 1) | (window['hypersomnia'] == 1)).astype(float)
    # Mask NaNs so missing days don't falsely register as "0 disorders"
    window.loc[window['insomnia'].isna() & window['hypersomnia'].isna(), 'any_disorder_flag'] = np.nan

    disorder_custom = (
        window.groupby(group_cols)
        .agg(
            insomnia_max_streak=('insomnia', streak_length),
            hypersomnia_max_streak=('hypersomnia', streak_length),
            insomnia_cv=('insomnia', lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan),
            hypersomnia_cv=('hypersomnia', lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan),
            any_disorder=('any_disorder_flag', 'max')
        )
        .reset_index()
    )
    disorder_agg = disorder_agg.merge(disorder_custom, on=group_cols, how="left")

    # Master Merge
    agg_list = [
        total_sleep_time_agg,
        time_in_bed_agg,
        sleep_stage_agg,
        sleep_timing_agg,
        rem_latency_agg,
        awakenings_agg,
        disorder_agg
    ]

    agg = reduce(
        lambda left, right: left.merge(right, on=group_cols, how="outer"),
        agg_list
    )

    phq8_with_sleep_stats = phq8.merge(
        agg,
        on=['participant_name', 'timeInterval_ID'],
        how='left'
    ).merge(
        missing_counts,
        on=['participant_name', 'timeInterval_ID'],
        how='left'
    )

    invalid_mask = phq8_with_sleep_stats[f'sleep_missing_last{window_size}'] > missing_threshold
    stat_cols = [col for col in agg.columns if col not in ['participant_name', 'timeInterval_ID']]
    phq8_with_sleep_stats.loc[invalid_mask, stat_cols] = np.nan

    return phq8_with_sleep_stats

def main():
    # read cleaned datasets; PHQ8 and sleep remain CSVs, demographics is parquet to preserve categorical metadata
    phq8 = pd.read_csv(PATH_TO_PHQ8)
    sleep = pd.read_csv(PATH_TO_SLEEP)
    demographics = pd.read_parquet(PATH_TO_DEMOGRAPHICS)

    window_size = 14
    missing_threshold = 7

    phq8_merged_with_sleep = merge_with_sleep(phq8, sleep, window_size, missing_threshold)

    # Calculate missingness rate for each column
    missingness_rate = phq8_merged_with_sleep.isnull().mean()

    # Keep only columns with 40% or less missingness
    phq8_merged_with_sleep = phq8_merged_with_sleep.loc[:, missingness_rate <= 0.40]

    # merge with demographics
    final_dataset = phq8_merged_with_sleep.merge(
         demographics, 
         on = ['participant_name'], 
         how = 'left'
         )
        
    final_dataset.to_parquet(PATH_TO_FINAL_DATASET, engine='pyarrow', index = False)
    
if __name__ == "__main__":
    main()

