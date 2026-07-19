import pandas as pd
import yaml

IN_FILE = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_steps.csv"
OUT_FILE = "../data_cleaned/steps_data.csv"
PATH_TO_YAML = "../feature_spec.yaml"

# Non-feature key columns always kept in the cleaned output.
ID_COLUMNS = ["participant_name", "timeInterval_ID"]

# Only steps columns that are consumed downstream (a basic `steps:` feature or a
# catch22 `base_features` signal in feature_spec.yaml) are winsorized here —
# winsorizing a column that never becomes a feature adds nothing. FEATURE_007 is
# catch22-only (not in the `steps:` section). Column numbers below are the
# FEATURE_0NN suffixes from the steps feature dictionary (e.g. "feature 1"
# -> FEATURE_001). FEATURE_022 ("most active hour") is a circular time-of-day
# column and is intentionally left uncapped: quantile clipping is meaningless for
# circular values.
#
# IMPORTANT: if feature_spec.yaml is changed to feed a NEW steps column into the
# feature set, you must revisit WINSORIZE_LIMITS below and add it (or make a
# deliberate choice to leave it uncapped). This mapping does not track
# feature_spec.yaml automatically.
#
# Winsorization limits per column: (lower_quantile, upper_quantile).
# A None bound leaves that tail uncapped (asymmetrical winsorization).
WINSORIZE_LIMITS = {
    "FEATURE_001": (0.25, 0.99),
    "FEATURE_019": (None, 0.99),
    "FEATURE_039": (0.25, 0.99),
    "FEATURE_040": (None, 0.99),
    "FEATURE_021": (None, 0.99),
    "FEATURE_007": (None, 0.99),
}


def winsorize(steps_data):
    """
    Clip outliers in selected steps columns to their quantile bounds.

    For each column in WINSORIZE_LIMITS, values are clipped to the given
    lower/upper quantiles. A None bound leaves that tail uncapped, giving
    asymmetrical winsorization. Quantiles are computed on non-null values and
    NaNs are preserved.

    Returns:
        pd.DataFrame: steps_data with the listed columns winsorized in place.
    """
    for column, (lower_q, upper_q) in WINSORIZE_LIMITS.items():
        lower = steps_data[column].quantile(lower_q) if lower_q is not None else None
        upper = steps_data[column].quantile(upper_q) if upper_q is not None else None
        steps_data[column] = steps_data[column].clip(lower=lower, upper=upper)
    return steps_data


def load_steps_feature_columns():
    """
    Read the steps columns to keep in the cleaned dataset from feature_spec.yaml.

    feature_spec.yaml is the single source of truth. Two sections consume raw
    steps columns downstream, so both are unioned here:
      - the `steps:` section (basic aggregation features), and
      - any `steps.*` base signal in the catch22 `base_features` panel.

    Returns the column names de-duplicated, `steps:` section order first.
    """
    with open(PATH_TO_YAML, "r") as file:
        # safe_load to avoid executing arbitrary code inside the YAML.
        feature_spec = yaml.safe_load(file)

    columns = list(feature_spec.get("steps", {}).keys())

    catch22_config = feature_spec.get("catch22", {})
    for entry in catch22_config.get("base_features", []):
        stream, _, column = entry.get("signal", "").partition(".")
        if stream == "steps" and column and column not in columns:
            columns.append(column)

    return columns


steps_raw = pd.read_csv(IN_FILE)

steps_v2 = steps_raw[steps_raw["Script_Version"] == "v2"]

steps_cols_renamed = steps_v2.rename(columns={"SUBJECT_ID": "participant_name", "DATE_ID":"timeInterval_ID" })

# feature_spec.yaml is the single source of truth for which steps columns are
# kept; select the ID columns plus those features (DEVICE_ID_01, STREAM_ID, etc.
# not listed there are dropped by omission).
steps_features = load_steps_feature_columns()
missing = [col for col in steps_features if col not in steps_cols_renamed.columns]
assert not missing, (
    f"feature_spec.yaml steps columns absent from raw data: {missing}"
)

steps_cleaned = steps_cols_renamed[ID_COLUMNS + steps_features].copy()

steps_cleaned = winsorize(steps_cleaned) # cap outliers in selected columns to their quantile bounds

steps_cleaned.to_csv(OUT_FILE, index=False)


