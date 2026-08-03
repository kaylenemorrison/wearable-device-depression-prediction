import pandas as pd
import yaml


PATH_TO_TIME_INTERVALS = "../data/radarV1_MDD/csv_files/timeIntervals.csv"
PATH_TO_SLEEP = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_sleep.csv"
PATH_TO_CLEAN_DATA = '../data_cleaned/sleep_data.csv'
PATH_TO_YAML = "../feature_spec.yaml"

# Non-feature key columns always kept in the cleaned output.
ID_COLUMNS = ["participant_name", "timeInterval_ID"]

# Sleep features referenced in feature_spec.yaml that are NOT present in the raw
# sleep data because they are DERIVED later in merge_datasets.py (_derive_sleep):
#   - sleep_midpoint  from sleep_onset / sleep_offset
#   - rem_deep_ratio  from REM_pct / deep_pct
# They are excluded from selection here; listing them explicitly means any OTHER
# missing YAML column still surfaces as an error rather than being silently dropped.
DERIVED_SLEEP_COLUMNS = {"sleep_midpoint", "rem_deep_ratio"}

# Raw sleep columns that are no longer features in their own right (dropped from
# the `sleep:` section of feature_spec.yaml) but must still be retained in the
# cleaned output because a DERIVED feature is computed from them downstream
# (merge_datasets._derive_sleep): rem_deep_ratio = REM_pct / deep_pct. They are
# kept and winsorized here but are not aggregated into the final feature set.
DERIVATION_INPUT_COLUMNS = ["deep_pct", "REM_pct"]


def load_sleep_feature_columns():
    """
    Read the sleep columns to keep in the cleaned dataset from feature_spec.yaml.

    feature_spec.yaml is the single source of truth. Two sections consume raw
    sleep columns downstream, so both are unioned here:
      - the `sleep:` section (basic aggregation features), and
      - any `sleep.*` base signal in the catch22 `base_features` panel.

    Returns the column names de-duplicated, `sleep:` section order first.
    """
    with open(PATH_TO_YAML, "r") as file:
        # safe_load to avoid executing arbitrary code inside the YAML.
        feature_spec = yaml.safe_load(file)

    columns = list(feature_spec.get("sleep", {}).keys())

    catch22_config = feature_spec.get("catch22", {})
    for entry in catch22_config.get("base_features", []):
        stream, _, column = entry.get("signal", "").partition(".")
        if stream == "sleep" and column and column not in columns:
            columns.append(column)

    return columns

def load_in_time_intervals():
    """
    Load time interval data from CSV and preprocess it.

    - Reads the time intervals file.
    - Converts 'datetimeStart' to a normalized date (removes time component).
    - Drops the original datetime columns.

    Returns:
        pd.DataFrame: Cleaned time intervals with a 'date' column.
    """
    time_intervals = pd.read_csv(PATH_TO_TIME_INTERVALS)
    time_intervals["date"] = pd.to_datetime(time_intervals["datetimeStart"]).dt.normalize()
    time_intervals.drop(columns = {"datetimeStart", "datetimeEnd"}, inplace = True)
    return time_intervals

# Only sleep columns that are consumed downstream (i.e. have an entry in the
# `sleep:` section of feature_spec.yaml, or feed a derived feature) are winsorized
# here — winsorizing a column that never becomes a feature adds nothing. Circular
# time-of-day columns (sleep_onset/offset/midpoint) and binary day-flags
# (insomnia/hypersomnia) are intentionally excluded: quantile clipping is
# meaningless for those types. deep_pct/REM_pct are no longer features but are
# winsorized as the raw inputs to the derived rem_deep_ratio (built in
# merge_datasets._derive_sleep), so that ratio inherits bounded tails.
#
# IMPORTANT: if feature_spec.yaml is changed to feed a NEW sleep column into the
# feature set, you must revisit WINSORIZE_LIMITS below and add it (or make a
# deliberate choice to leave it uncapped). This mapping does not track
# feature_spec.yaml automatically.
#
# Winsorization limits per column: (lower_quantile, upper_quantile).
# A None bound leaves that tail uncapped (asymmetrical winsorization).
WINSORIZE_LIMITS = {
    "total_sleep_time": (0.01, 0.99),
    "sleep_efficiency": (0.01, 0.99),
    "deep_pct": (0.01, 0.99),
    "REM_pct": (0.01, 0.99),
    "awake_pct": (0.01, 0.99),
    "rem_latency": (None, 0.95),
    "awakenings": (None, 0.99),
}


def winsorize(sleep_data):
    """
    Clip outliers in selected sleep columns to their quantile bounds.

    For each column in WINSORIZE_LIMITS, values are clipped to the given
    lower/upper quantiles. A None bound leaves that tail uncapped, giving
    asymmetrical (upper-tail-only) winsorization. Quantiles are computed on
    non-null values and NaNs are preserved.

    Returns:
        pd.DataFrame: sleep_data with the listed columns winsorized in place.
    """
    for column, (lower_q, upper_q) in WINSORIZE_LIMITS.items():
        lower = sleep_data[column].quantile(lower_q) if lower_q is not None else None
        upper = sleep_data[column].quantile(upper_q) if upper_q is not None else None
        sleep_data[column] = sleep_data[column].clip(lower=lower, upper=upper)
    return sleep_data


def assertions(sleep_data):
    """
    Run data quality checks on sleep_data.
    """
    # Check for duplicate participant_name + timeInterval_ID combinations
    duplicates = sleep_data.duplicated(subset=["participant_name", "timeInterval_ID"])
    
    assert not duplicates.any(), (
        f"Found {duplicates.sum()} duplicate rows based on "
        "['participant_name', 'timeInterval_ID']"
    )


def main():
    """
    Main pipeline for cleaning and validating sleep data.

    Steps:
    1. Load and preprocess time interval data.
    2. Load raw sleep data and align column names.
    3. Merge sleep data with time intervals to attach a normalized date.
    4. Select the sleep feature columns declared in feature_spec.yaml.
    5. Remove duplicates by choosing the row with the longest period of sleep per (participant_name, timeInterval_ID).
    6. Winsorize selected columns to their quantile bounds to cap outliers.
    7. Run data quality assertions to ensure integrity and consistency.
    8. Save the cleaned dataset to disk.

    Output:
        Writes cleaned sleep data to './data_cleaned/sleep_data.csv'.
    """
    time_intervals = load_in_time_intervals()
    sleep_data = pd.read_csv(PATH_TO_SLEEP)

    num_rows_start = sleep_data.shape[0]

    sleep_data.rename(columns={"time_interval": "timeInterval_ID"}, inplace = True)
    sleep_data = sleep_data.merge(time_intervals, on="timeInterval_ID", how="left")

    sleep_features = [
        col for col in load_sleep_feature_columns()
        if col not in DERIVED_SLEEP_COLUMNS
    ]
    # Retain the raw inputs that downstream derivations consume (not features
    # themselves, so absent from the YAML sleep section).
    sleep_features += [
        col for col in DERIVATION_INPUT_COLUMNS if col not in sleep_features
    ]
    missing = [col for col in sleep_features if col not in sleep_data.columns]
    assert not missing, (
        f"feature_spec.yaml sleep columns absent from raw sleep data: {missing}"
    )

    relevant_columns = ID_COLUMNS + sleep_features
    sleep_data = sleep_data[relevant_columns]

    # Keep, per (participant_name, timeInterval_ID), the row with the longest
    # total_sleep_time. Select by index rather than returning rows from
    # groupby.apply: pandas 3.x excludes the grouping columns from the applied
    # frame, so the apply form silently drops the ID columns.
    keep_idx = (
        sleep_data.groupby(['participant_name', 'timeInterval_ID'])['total_sleep_time']
        .idxmax()
    )
    sleep_data = sleep_data.loc[keep_idx].reset_index(drop=True)

    sleep_data = winsorize(sleep_data)

    assertions(sleep_data)

    num_rows_end = sleep_data.shape[0]

    print(f"{round(num_rows_end/num_rows_start * 100, 2)}% of observations retained.") # 86% retained

    sleep_data.to_csv(PATH_TO_CLEAN_DATA, index=False)

if __name__ == "__main__":
    main()