import pandas as pd

IN_FILE = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_steps.csv"
OUT_FILE = "../data_cleaned/steps_data.csv"

steps_raw = pd.read_csv(IN_FILE)

steps_v2 = steps_raw[steps_raw["Script_Version"] == "v2"]

steps_missing_cols_dropped = steps_v2.dropna(how='all', axis=1) # subset columns that are completely missing data

steps_cols_renamed = steps_missing_cols_dropped.rename(columns={"SUBJECT_ID": "participant_name", "DATE_ID":"timeInterval_ID" }) 

columns_to_drop = ['DEVICE_ID_01', 'STREAM_ID', 'Last_updated','Script_Version']

steps_cleaned = steps_cols_renamed.drop(columns=columns_to_drop) # drop unneeded columns

steps_cleaned.to_csv(OUT_FILE, index=False)


