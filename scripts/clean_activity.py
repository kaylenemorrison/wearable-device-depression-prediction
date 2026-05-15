import pandas as pd

IN_FILE = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_activity.csv"
OUT_FILE = '../data_cleaned/activity_data.csv'

activity_raw = pd.read_csv(IN_FILE)

activity_v1 = activity_raw[activity_raw["Script_Version"] == "v1"] # subset version 1

activity_duplicates_dropped = activity_v1.drop_duplicates() # drop all identical rows

activity_missing_cols_dropped = activity_duplicates_dropped.dropna(how='all', axis=1) # drop all columns with 100% missing values

columns_to_drop = ['DEVICE_ID_01', 'STREAM_ID', 'Last_updated', 'Script_Version']

activity_cleaned = activity_missing_cols_dropped.drop(columns=columns_to_drop) # drop unneeded columns

activity_cleaned = activity_cleaned.rename(columns={"SUBJECT_ID": "participant_name", "DATE_ID":"timeInterval_ID" }) 

activity_cleaned.to_csv(OUT_FILE, index=False)
