
import pandas as pd
from datetime import datetime
import random
random.seed(42)

PATH_TO_TIME_INTERVALS = "../data/radarV1_MDD/csv_files/timeIntervals.csv"
PATH_TO_PHQ8 = "../data/radarV1_MDD/csv_files/phq8_data.csv"
PATH_TO_CLEAN_DATA = '../data_cleaned/phq8_data.csv'

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

def load_in_phq8():
    """
    Load PHQ-8 questionnaire data from CSV.

    Returns:
        pd.DataFrame: Raw PHQ-8 dataset.
    """
    return pd.read_csv(PATH_TO_PHQ8)

def calculate_total_phq8(phq8_data):
    """
    Compute total PHQ-8 score for each row.

    - Identifies columns 'value0' to 'value7'.
    - Sums them row-wise to create a 'total_score'.

    Args:
        phq8_data (pd.DataFrame): Raw PHQ-8 data.

    Returns:
        pd.DataFrame: Data with an added 'total_score' column.
    """
    cols = phq8_data.filter(regex=r'^value[0-7]$').columns
    phq8_data['phq8_score'] = phq8_data[cols].sum(axis=1) 
    return phq8_data

def eliminate_duplicates(max_range, phq8_data):
    """
    Remove inconsistent duplicate PHQ-8 entries within the same time interval.

    - Groups by participant and time interval.
    - Computes score range (max - min) within each group.
    - Filters out groups where variability exceeds 'max_range'.
    - Randomly samples one entry per remaining group.

    Args:
        max_range (int): Maximum allowed score variation within a group.
        phq8_data (pd.DataFrame): PHQ-8 data with total scores.

    Returns:
        pd.DataFrame: Deduplicated dataset.
    """
    phq8_data['score_range'] = (
    phq8_data.groupby(['participant_name', 'time_interval'])['phq8_score']
      .transform(lambda x: x.max() - x.min())
    )

    phq8_data = phq8_data[phq8_data['score_range'] <= max_range]
    phq8_data = (
    phq8_data.groupby(['participant_name', 'time_interval'], group_keys=False)
      .apply(lambda x: x.sample(n=1))
      .reset_index(drop=True)
    ) 
    return phq8_data

def merge_with_intervals(time_intervals, phq8_data):
    """
    Merge PHQ-8 data with corresponding time interval dates.

    - Renames 'time_interval' to match interval ID.
    - Performs a left join with time intervals.

    Args:
        time_intervals (pd.DataFrame): Processed time interval data.
        phq8_data (pd.DataFrame): PHQ-8 data.

    Returns:
        pd.DataFrame: Merged dataset with date information.
    """
    phq8_data = phq8_data.rename(columns={"time_interval": "timeInterval_ID"}) 
    phq8_data = phq8_data.merge(time_intervals, on="timeInterval_ID", how="left")
    return phq8_data

def make_windows_better(phq8_data, max_difference_from_reference):
    """
    Align PHQ-8 observations into regular 14-day windows per participant.

    Steps:
    - Determines each participant's start date.
    - Assigns each record to the nearest 14-day period index.
    - Computes a reference date for each period.
    - Calculates deviation from the reference date.
    - Filters records within a specified tolerance.
    - Keeps the closest observation per period.

    Args:
        phq8_data (pd.DataFrame): Merged PHQ-8 dataset with dates.
        max_difference_from_reference (int): Maximum allowed deviation (in days).

    Returns:
        pd.DataFrame: Windowed dataset with one observation per period.
    """
    # Get each participant's first date
    phq8_data['start_date'] = phq8_data.groupby('participant_name')['date'].transform('min')

    # Compute period index (nearest 14-day interval per participant)
    phq8_data['period_index'] = (
        ((phq8_data['date'] - phq8_data['start_date']).dt.days / 14)
        .round()
        .astype(int)
    )

    # Create reference date per participant
    phq8_data['ref_date'] = phq8_data['start_date'] + pd.to_timedelta(phq8_data['period_index'] * 14, unit='D')

    # Difference in days
    phq8_data['diff_days'] = (phq8_data['date'] - phq8_data['ref_date']).dt.days

    # Filter within ± max_difference_from_reference days
    phq8_data = phq8_data[phq8_data['diff_days'].abs() <= max_difference_from_reference]

    # If there are multiple dates that align to the same period, keep the date that is closest to the reference 
    phq8_data = (
    phq8_data.sort_values('diff_days', key=abs)
      .drop_duplicates(['participant_name', 'period_index'])
    )

    return phq8_data

def assertions(phq8_data):
    assert phq8_data['phq8_score'].between(0, 24).all(), "Invalid PHQ8 scores"

def main():
    """
    Execute the full PHQ-8 data processing pipeline.

    Workflow:
    - Load time intervals and PHQ-8 data.
    - Compute total PHQ-8 scores.
    - Remove inconsistent duplicates.
    - Merge with time interval dates.
    - Align data into 14-day windows.
    - Select relevant columns.
    - Save cleaned dataset to CSV.
    """
    time_intervals = load_in_time_intervals()
    phq8_data = load_in_phq8()

    num_rows_start = phq8_data.shape[0]

    phq8_data = calculate_total_phq8(phq8_data)

    max_range = 3
    phq8_data = eliminate_duplicates(max_range, phq8_data)

    phq8_data = merge_with_intervals(time_intervals, phq8_data)

    max_difference_from_reference = 2
    phq8_data = make_windows_better(phq8_data, max_difference_from_reference )
    
    relevant_columns = [
        'participant_name',   # Unique identifier for each participant
        'timeInterval_ID',    # ID linking the record to a specific time interval
        #'date',               # Date of the PHQ-8 assessment 
        'phq8_score',        # Sum of PHQ-8 item scores 
        #'start_date',         # First recorded assessment date for the participant (baseline)
        #'period_index',       # Index of the nearest 14-day window since start_date
        #'ref_date',           # Reference date for the 14-day window (start_date + 14 * period_index)
        #'diff_days'           # Difference (in days) between actual date and reference date
    ]
    
    phq8_data = phq8_data[relevant_columns]

    assertions(phq8_data)

    num_rows_end = phq8_data.shape[0]

    print(f"{round(num_rows_end/num_rows_start * 100, 2)}% of observations retained.") # 95% retained

    phq8_data.to_csv(PATH_TO_CLEAN_DATA, index=False)

if __name__ == "__main__":
    main()