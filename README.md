# Predicting Depression Severity from Wearable Data

MSc thesis project predicting depression severity (PHQ-8 score) from Fitbit
wearable data (sleep, activity, heart rate, steps) and demographics, using
data from the RADAR-MDD study.

The modeling unit is one PHQ-8 assessment per participant per *N*-day window
(*N* ∈ {3, 5, 7, 14, 21, 28}), with sleep, activity, heart-rate, and step
features aggregated from the days preceding the assessment.

## Setup

```bash
conda env create -f environment.yml
conda activate thesis

# after adding a dependency:
conda env update --file environment.yml --prune
```

Key packages: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`,
`missforest`, `optuna`, `pycatch22`, `shap`, `pyarrow`.

> Raw and intermediate participant data (`data/`, `data_cleaned/`,
> `data_merged/`) are not included in this repository and are not tracked by
> git — RADAR-MDD data is governed by its own access/IRB restrictions.

## Pipeline

All scripts are run from the `scripts/` directory and use relative paths.

```bash
cd scripts

# 1. Clean each raw data source (run once per source)
python clean_phq8.py
python clean_sleep.py
python clean_demographics.py
python clean_activity.py
python clean_steps.py
python clean_heartrate.py

# 2. Merge cleaned sources into a modeling-ready dataset for one window size
python merge_datasets.py --window-size 14 --min-present-fraction 0.75

# 3. Run nested cross-validated modeling
python modeling.py \
  --feature-set simple \
  --window-size 14 \
  --min-present-fraction 0.75 \
  --imputation missforest \
  --model elastic_net \
  --n-repeats 100 \
  --n-jobs 8

# 4. Summarize completed runs into a heatmap grid
python summary_grid.py --metric rmse
```

## Repository Structure

```
feature_spec.yaml          hand-curated feature spec (aggregations, slope, catch22 panel)
environment.yml            conda environment
scripts/
  clean_*.py                raw -> data_cleaned/
  merge_datasets.py          data_cleaned/ -> data_merged/final_dataset_*.parquet
  modeling.py                 nested CV modeling, Optuna tuning, checkpointing
  summary_grid.py             results/ -> heatmap of held-out metrics
  run_*_grid.sh                predefined modeling.py sweeps
  classes/repeated_groupkfold.py   RepeatedGroupKFold used by modeling.py
*_eda.ipynb                 exploratory analysis notebooks (demographics, PHQ-8, sleep, missingness)
scratch/                    exploratory notebooks, gitignored
results/                    model outputs and plots, gitignored
```

`data/`, `data_cleaned/`, and `data_merged/` are gitignored and not present in
a fresh checkout; they are produced locally by the cleaning/merge steps above
from raw RADAR-MDD exports.

## Feature Families

Selected independently per run via `--feature-set` in `modeling.py`:

- **demographics** — age, gender, smoking status, AUDIT score
- **simple** — level/spread/extreme/total aggregates per wearable stream
- **slope** — within-window OLS trend per signal
- **catch22** — a fixed, axis-diverse panel of dynamical time-series features (windows ≥ 14 only)
- **adherence** — structural missingness indicators per stream

Full column definitions and feature counts are in `feature_spec.yaml`.
