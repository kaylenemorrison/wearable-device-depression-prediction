"""
Check residual missingness after MissForest imputation across all CV folds.
Run from the scripts/ directory:
    python ../scratch/imputation_missingness_investigation/check_missforest_residual.py > ../scratch/imputation_missingness_investigation/missforest_residual_missingness.txt
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from missforest import MissForest
from sklearn.model_selection import GroupKFold
from modeling import prepare_data

SEED = 42
df = pd.read_parquet("../data_cleaned/final_dataset.parquet")
df["any_disorder"] = df["any_disorder"].astype("category")

categorical_cols = ["marital_status", "ethnicity_1", "accommodation", "mh_smoking", "audit_1"]
boolean_cols_demographics = ["children", "drop_school_1", "depression_medication", "gender"]

X, y, groups = prepare_data(df, categorical_cols, boolean_cols_demographics)

all_cat_cols = [col for col in X.columns if X[col].dtype.name == "category"]

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

        # MissForest's internal .any() call fails on pandas category dtype;
        # cast to float first (preserves NaN) while still telling the
        # imputer which columns are categorical so it uses a classifier.
        X_itr_mf = X_itr.copy()
        X_ival_mf = X_ival.copy()
        for col in all_cat_cols:
            X_itr_mf[col] = X_itr_mf[col].astype(float)
            X_ival_mf[col] = X_ival_mf[col].astype(float)

        imputer = MissForest(categorical=all_cat_cols, verbose=0)
        imputer.fit(X_itr_mf)
        X_itr_imp = imputer.transform(X_itr_mf)
        X_ival_imp = imputer.transform(X_ival_mf)

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
    print("No residual NaN — MissForest fully imputes everything.")
