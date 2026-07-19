import pandas as pd
import yaml

IN_FILE = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_heart_rate.csv"
OUT_FILE = "../data_cleaned/heartrate_data.csv"
PATH_TO_YAML = "../feature_spec.yaml"

# Non-feature key columns always kept in the cleaned output.
ID_COLUMNS = ["participant_name", "timeInterval_ID"]

# Only heart-rate columns that are consumed downstream (i.e. have an entry in the
# `heartrate:` section of feature_spec.yaml) are winsorized here — winsorizing a
# column that never becomes a feature adds nothing. Column numbers below are the
# FEATURE_0NN suffixes from the heart-rate feature dictionary (e.g. "feature 17"
# -> FEATURE_017).
#
# IMPORTANT: if feature_spec.yaml is changed to feed a NEW heart-rate column into
# the feature set, you must revisit WINSORIZE_LIMITS below and add it (or make a
# deliberate choice to leave it uncapped). This mapping does not track
# feature_spec.yaml automatically.
#
# Winsorization limits per column: (lower_quantile, upper_quantile).
# A None bound leaves that tail uncapped (asymmetrical winsorization).
WINSORIZE_LIMITS = {
    "FEATURE_017": (0.01, 0.99),
    "FEATURE_006": (0.01, 0.99),
    "FEATURE_018": (0.01, 0.99),
    "FEATURE_005": (0.01, 0.99),
    "FEATURE_021": (0.01, 0.99),
    "FEATURE_030": (0.01, 0.99),
    "FEATURE_031": (None, 0.995),
    "FEATURE_012": (0.01, 0.99),
}


def winsorize(heartrate_data):
    """
    Clip outliers in selected heart-rate columns to their quantile bounds.

    For each column in WINSORIZE_LIMITS, values are clipped to the given
    lower/upper quantiles. A None bound leaves that tail uncapped, giving
    asymmetrical winsorization. Quantiles are computed on non-null values and
    NaNs are preserved.

    Returns:
        pd.DataFrame: heartrate_data with the listed columns winsorized in place.
    """
    for column, (lower_q, upper_q) in WINSORIZE_LIMITS.items():
        lower = heartrate_data[column].quantile(lower_q) if lower_q is not None else None
        upper = heartrate_data[column].quantile(upper_q) if upper_q is not None else None
        heartrate_data[column] = heartrate_data[column].clip(lower=lower, upper=upper)
    return heartrate_data


def load_heartrate_feature_columns():
    """
    Read the heart-rate columns to keep in the cleaned dataset from
    feature_spec.yaml.

    feature_spec.yaml is the single source of truth. Two sections consume raw
    heart-rate columns downstream, so both are unioned here:
      - the `heartrate:` section (basic aggregation features), and
      - any `heartrate.*` base signal in the catch22 `base_features` panel.

    Returns the column names de-duplicated, `heartrate:` section order first.
    """
    with open(PATH_TO_YAML, "r") as file:
        # safe_load to avoid executing arbitrary code inside the YAML.
        feature_spec = yaml.safe_load(file)

    columns = list(feature_spec.get("heartrate", {}).keys())

    catch22_config = feature_spec.get("catch22", {})
    for entry in catch22_config.get("base_features", []):
        stream, _, column = entry.get("signal", "").partition(".")
        if stream == "heartrate" and column and column not in columns:
            columns.append(column)

    return columns


heartrate_raw = pd.read_csv(IN_FILE)

heartrate_v2 = heartrate_raw[heartrate_raw["Script_Version"] == "v2"] # subset v2 of the data

heartrate_v2 = heartrate_v2.rename(columns={"SUBJECT_ID": "participant_name", "DATE_ID":"timeInterval_ID" })

# feature_spec.yaml is the single source of truth for which heart-rate columns
# are kept; select the ID columns plus those features (the QUALITY_METRIC,
# DEVICE_ID, etc. columns not listed there are dropped by omission).
heartrate_features = load_heartrate_feature_columns()
missing = [col for col in heartrate_features if col not in heartrate_v2.columns]
assert not missing, (
    f"feature_spec.yaml heart-rate columns absent from raw data: {missing}"
)

heartrate_cleaned = heartrate_v2[ID_COLUMNS + heartrate_features].copy()

heartrate_cleaned = winsorize(heartrate_cleaned) # cap outliers in selected columns to their quantile bounds

heartrate_cleaned.to_csv(OUT_FILE, index=False)



