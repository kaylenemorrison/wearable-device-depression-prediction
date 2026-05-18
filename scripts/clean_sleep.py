import pandas as pd

## DON'T CHOOSE SLEEP ROWS RANDOMLY!! CHANGE SCRIPT TO CHOOSE THE LONGEST SLEEP EPISODE PER DAY PER PERSON

PATH_TO_TIME_INTERVALS = "../data/radarV1_MDD/csv_files/timeIntervals.csv"
PATH_TO_SLEEP = "../data/radarV1_MDD/csv_files/dailyFeatures_fitbit_sleep.csv"
PATH_TO_CLEAN_DATA = '../data_cleaned/sleep_data.csv'

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
    4. Select relevant features for analysis.
    5. Remove duplicates by choosing the row with the longest period of sleep per (participant_name, timeInterval_ID).
    6. Run data quality assertions to ensure integrity and consistency.
    7. Save the cleaned dataset to disk.

    Output:
        Writes cleaned sleep data to './data_cleaned/sleep_data.csv'.
    """
    time_intervals = load_in_time_intervals()
    sleep_data = pd.read_csv(PATH_TO_SLEEP)

    num_rows_start = sleep_data.shape[0]

    sleep_data.rename(columns={"time_interval": "timeInterval_ID"}, inplace = True)
    sleep_data = sleep_data.merge(time_intervals, on="timeInterval_ID", how="left")

    relevant_columns = ["participant_name", "timeInterval_ID", "total_sleep_time", 
                         "time_in_bed", 'light_pct', 'deep_pct', 'REM_pct',
                         'awake_pct', 'NREM_pct', 'sleep_onset', 'sleep_offset', 'rem_latency',
                         'sleep_efficiency', 'awakenings', 'insomnia', 'hypersomnia']
    
    sleep_data = sleep_data[relevant_columns]

    sleep_data = (
        sleep_data.groupby(['participant_name', 'timeInterval_ID'], group_keys=False)
        .apply(lambda x: x.loc[x['total_sleep_time'].idxmax()])
        .reset_index(drop=True)
    )

    assertions(sleep_data)

    num_rows_end = sleep_data.shape[0]

    print(f"{round(num_rows_end/num_rows_start * 100, 2)}% of observations retained.") # 86% retained

    sleep_data.to_csv(PATH_TO_CLEAN_DATA, index=False)

if __name__ == "__main__":
    main()