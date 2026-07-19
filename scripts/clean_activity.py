import pandas as pd
import yaml

IN_FILE = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_activity.csv"
OUT_FILE = "../data_cleaned/activity_data.csv"
PATH_TO_YAML = "../feature_spec.yaml"

# Non-feature key columns always kept in the cleaned output.
ID_COLUMNS = ["participant_name", "timeInterval_ID"]

# Only activity columns that are consumed downstream (i.e. have an entry in the
# `activity:` section of feature_spec.yaml) are winsorized here — winsorizing a
# column that never becomes a feature adds nothing. Column numbers below are the
# FEATURE_0NN suffixes from the activity feature dictionary (e.g. "feature 1"
# -> FEATURE_001). FEATURE_018 ("most active hour") is a circular time-of-day
# column and is intentionally left uncapped: quantile clipping is meaningless for
# circular values.
#
# IMPORTANT: if feature_spec.yaml is changed to feed a NEW activity column into
# the feature set, you must revisit WINSORIZE_LIMITS below and add it (or make a
# deliberate choice to leave it uncapped). This mapping does not track
# feature_spec.yaml automatically.
#
# Winsorization limits per column: (lower_quantile, upper_quantile).
# A None bound leaves that tail uncapped (asymmetrical winsorization).
WINSORIZE_LIMITS = {
    "FEATURE_001": (0.05, 0.95),
    "FEATURE_004": (None, 0.99),
    "FEATURE_005": (0.01, 0.99),
    "FEATURE_013": (None, 0.99),
    "FEATURE_015": (None, 0.99),
    "FEATURE_016": (None, 0.99),
}


def winsorize(activity_data):
    """
    Clip outliers in selected activity columns to their quantile bounds.

    For each column in WINSORIZE_LIMITS, values are clipped to the given
    lower/upper quantiles. A None bound leaves that tail uncapped, giving
    asymmetrical winsorization. Quantiles are computed on non-null values and
    NaNs are preserved.

    Returns:
        pd.DataFrame: activity_data with the listed columns winsorized in place.
    """
    for column, (lower_q, upper_q) in WINSORIZE_LIMITS.items():
        lower = activity_data[column].quantile(lower_q) if lower_q is not None else None
        upper = activity_data[column].quantile(upper_q) if upper_q is not None else None
        activity_data[column] = activity_data[column].clip(lower=lower, upper=upper)
    return activity_data


def load_activity_feature_columns():
    """
    Read the activity columns to keep in the cleaned dataset from
    feature_spec.yaml.

    feature_spec.yaml is the single source of truth. Two sections consume raw
    activity columns downstream, so both are unioned here:
      - the `activity:` section (basic aggregation features), and
      - any `activity.*` base signal in the catch22 `base_features` panel.

    Returns the column names de-duplicated, `activity:` section order first.
    """
    with open(PATH_TO_YAML, "r") as file:
        # safe_load to avoid executing arbitrary code inside the YAML.
        feature_spec = yaml.safe_load(file)

    columns = list(feature_spec.get("activity", {}).keys())

    catch22_config = feature_spec.get("catch22", {})
    for entry in catch22_config.get("base_features", []):
        stream, _, column = entry.get("signal", "").partition(".")
        if stream == "activity" and column and column not in columns:
            columns.append(column)

    return columns


activity_raw = pd.read_csv(IN_FILE)

activity_v1 = activity_raw[activity_raw["Script_Version"] == "v1"] # subset version 1

activity_duplicates_dropped = activity_v1.drop_duplicates() # drop all identical rows

activity_renamed = activity_duplicates_dropped.rename(columns={"SUBJECT_ID": "participant_name", "DATE_ID":"timeInterval_ID" })

# feature_spec.yaml is the single source of truth for which activity columns are
# kept; select the ID columns plus those features (DEVICE_ID_01, STREAM_ID, etc.
# not listed there are dropped by omission).
activity_features = load_activity_feature_columns()
missing = [col for col in activity_features if col not in activity_renamed.columns]
assert not missing, (
    f"feature_spec.yaml activity columns absent from raw data: {missing}"
)

activity_cleaned = activity_renamed[ID_COLUMNS + activity_features].copy()

activity_cleaned = winsorize(activity_cleaned) # cap outliers in selected columns to their quantile bounds

activity_cleaned.to_csv(OUT_FILE, index=False)
