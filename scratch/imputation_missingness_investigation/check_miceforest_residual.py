"""
Check residual missingness after miceforest imputation across all CV folds.
Run from the scripts/ directory:
    python ../scratch/imputation_missingness_investigation/check_miceforest_residual.py > ../scratch/imputation_missingness_investigation/miceforest_residual_missingness.txt
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import miceforest as mf
from sklearn.model_selection import GroupKFold
from modeling import prepare_data

SEED = 42
df = pd.read_parquet("../data_cleaned/final_dataset.parquet")
df["any_disorder"] = df["any_disorder"].astype("category")

categorical_cols = ["marital_status", "ethnicity_1", "accommodation", "mh_smoking", "audit_1"]
boolean_cols_demographics = ["children", "drop_school_1", "depression_medication", "gender"]

X, y, groups = prepare_data(df, categorical_cols, boolean_cols_demographics)

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=3)

total_residual = {}

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), 1):
    X_train = X.iloc[train_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    for inner_fold_idx, (itr_idx, ival_idx) in enumerate(inner_cv.split(X_train, y_train, groups_train), 1):
        X_itr = X_train.iloc[itr_idx].reset_index(drop=True)
        X_ival = X_train.iloc[ival_idx].reset_index(drop=True)

        print(f"Outer {fold_idx}, inner {inner_fold_idx}...", flush=True)

        kernel = mf.ImputationKernel(data=X_itr, num_datasets=1, random_state=SEED, mean_match_strategy="fast")
        kernel.mice(iterations=3)

        X_itr_imp = kernel.complete_data(dataset=0)
        X_ival_imp = kernel.impute_new_data(X_ival).complete_data(dataset=0)

        for name, imp in [("train", X_itr_imp), ("val", X_ival_imp)]:
            missing = imp.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                label = f"outer{fold_idx}_inner{inner_fold_idx}_{name}"
                print(f"  {label}: {missing.to_dict()}")
                for col, cnt in missing.items():
                    total_residual[col] = total_residual.get(col, 0) + cnt

print()
if total_residual:
    print("Residual NaN across all folds:", total_residual)
else:
    print("No residual NaN — miceforest fully imputes everything.")
