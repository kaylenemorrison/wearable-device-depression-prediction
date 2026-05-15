import pandas as pd

IN_FILE = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_heart_rate.csv"
OUT_FILE = "../data_cleaned/heartrate_data.csv"


heartrate_raw = pd.read_csv(IN_FILE)

heartrate_v2 = heartrate_raw[heartrate_raw["Script_Version"] == "v2"] # subset v2 of the data

heartrate_v2 = heartrate_v2.dropna(how='all', axis=1) # drop all columns with completely missing data

cols_to_drop = [ 
    col for col in heartrate_v2.columns 
    if col.startswith('QUALITY_METRIC') and col != 'QUALITY_METRIC_007'
] + ['DEVICE_ID', 'STREAM_ID', 'Last_updated', 'Script_Version']    

heartrate_v2 = heartrate_v2.drop(columns=cols_to_drop) # drop all columns not in the data dictionary and unnecessary columns

heartrate_cleaned = heartrate_v2.rename(columns={"SUBJECT_ID": "participant_name", "DATE_ID":"timeInterval_ID" }) 

heartrate_cleaned.to_csv(OUT_FILE, index=False)



