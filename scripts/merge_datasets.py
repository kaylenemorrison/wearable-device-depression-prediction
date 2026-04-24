import pandas as pd

PATH_TO_PHQ8 = "../data_cleaned/phq8_data.csv"
PATH_TO_SLEEP = "../data_cleaned/sleep_data.csv"
PATH_TO_DEMOGRAPHICS = "../data_cleaned/totaldataset.csv"
PATH_TO_FINAL_DATASET = "../data_cleaned/final_dataset.csv"

def create_window(phq8, window_size):
    window = phq8[['participant_name', 'timeInterval_ID']].copy()

    window = window.loc[window.index.repeat(window_size)].copy()
    window['lag'] = list(range(1, window_size+1)) * len(phq8)

    # Compute the prior day IDs
    window['timeInterval_ID_window'] = window['timeInterval_ID'] - window['lag']
    return window

def remove_missingness(phq8, sleep, window_size, missing_threshold):
    sleep_obs = sleep[['participant_name', 'timeInterval_ID']].copy()
    sleep_obs['observed'] = 1

    sleep_obs = sleep_obs.rename(columns = {'timeInterval_ID': 'timeInterval_ID_sleep'})

    window = create_window(phq8, window_size)

    window = window.merge(
        sleep_obs,
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_sleep'],
        how='left'
    )

    window['missing'] = window['observed'].isna().astype(int)

    missing_counts = (
        window.groupby(['participant_name', 'timeInterval_ID'])
        ['missing']
        .sum()
        .reset_index(name=f'sleep_missing_last{window_size}')
    )

    phq8 = phq8.merge(
        missing_counts,
        on=['participant_name', 'timeInterval_ID'],
        how='left'
    )
    phq8 = phq8[phq8[f'sleep_missing_last{window_size}'] <= missing_threshold]

    return phq8

def merge_with_sleep(phq8, sleep, window_size):
    window = create_window(phq8, window_size)

    sleep = sleep.rename(columns = {'timeInterval_ID': 'timeInterval_ID_sleep'})

    window = window.merge(
        sleep[['participant_name', 'timeInterval_ID_sleep', 'total_sleep_time']],
        left_on=['participant_name', 'timeInterval_ID_window'],
        right_on=['participant_name', 'timeInterval_ID_sleep'],
        how='left'
    )

    agg = (
        window.groupby(['participant_name', 'timeInterval_ID'])['total_sleep_time']
        .agg(
            sleep_min='min',
            sleep_max='max',
            sleep_mean='mean',
            sleep_median='median',
            sleep_std='std',
            sleep_q25=lambda x: x.quantile(0.25),
            sleep_q75=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )

    # Compute IQR
    agg['sleep_iqr'] = agg['sleep_q75'] - agg['sleep_q25']

    # Drop intermediate quartiles 
    agg = agg.drop(columns=['sleep_q25', 'sleep_q75'])

    phq8_with_sleep_stats = phq8.merge(
        agg,
        on=['participant_name', 'timeInterval_ID'],
        how='left'
    )

    return phq8_with_sleep_stats

def main():
    phq8 = pd.read_csv(PATH_TO_PHQ8)
    sleep = pd.read_csv(PATH_TO_SLEEP)
    demographics = pd.read_csv(PATH_TO_DEMOGRAPHICS)

    window_size = 14
    missing_threshold = 7

    phq8_missingness_adjusted = remove_missingness(phq8, sleep, window_size, missing_threshold)

    phq8_merged_with_sleep = merge_with_sleep(phq8_missingness_adjusted, sleep, window_size)

    # merge with demographics
    final_dataset = phq8_merged_with_sleep.merge(
        demographics, 
        on = ['participant_name'], 
        how = 'left'
        )
    
    final_dataset.to_csv(PATH_TO_FINAL_DATASET, index = False)
    
if __name__ == "__main__":
    main()

