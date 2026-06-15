"""Measure how long one MissForest imputation takes on the merged dataset,
full-column vs. correlation-pruned, and extrapolate to a full nested-CV run
(~24 imputations). Prints ONLY shapes and timings -- no participant rows.

Run from the scripts/ directory:
    python time_missforest.py --window-size 14 --threshold 0.9
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "classes")
from CorrelationPruner import CorrelationPruner  # noqa: E402
from modeling import missforest_impute, prepare_data  # noqa: E402

# Imputations per full nested-CV run (see modeling.py):
#   per outer fold: 3 inner-fold + 1 outer train/test = 4; x5 folds = 20
#   final model: 3 inner + 1 full = 4  -> 24 total
IMPUTATIONS_PER_RUN = 24

DEMOGRAPHIC_COLS = ["age", "gender", "mh_smoking", "audit_1", "depression_medication"]
CATEGORICAL_COLS = ["mh_smoking", "audit_1"]
BOOLEAN_COLS = ["depression_medication", "gender", "any_disorder"]


def time_one(X, categorical_features, label):
    t0 = time.time()
    missforest_impute(X, None, categorical_features, max_iter=3)
    dt = time.time() - t0
    print(f"  {label:<22} X={X.shape}  ->  {dt:6.1f}s "
          f"(~{dt * IMPUTATIONS_PER_RUN / 60:.1f} min for {IMPUTATIONS_PER_RUN} imputations)")
    return dt


def main(window_size, threshold):
    df = pd.read_parquet(f"../data_merged/final_dataset_w{window_size}.parquet")
    df["any_disorder"] = df["any_disorder"].astype("category")

    # Same encoding path as the missforest elastic-net run.
    X, _, _ = prepare_data(df, CATEGORICAL_COLS,
                           ["depression_medication", "gender", "any_disorder"])
    categorical_features = X.select_dtypes(include="category").columns.tolist()

    print("=" * 60)
    print("MISSFOREST TIMING")
    print("=" * 60)
    print(f"  full feature matrix: {X.shape}")
    print(f"  categorical cols:    {len(categorical_features)}")
    print()

    # 1) full-column imputation (prune-AFTER scenario: impute everything)
    t_full = time_one(X, categorical_features, "full (prune after)")

    # 2) prune first, then impute survivors (prune-BEFORE scenario)
    prune_cols = [c for c in X.select_dtypes(include=[np.number]).columns
                  if c not in DEMOGRAPHIC_COLS]
    pruner = CorrelationPruner(threshold=threshold)
    pruner.fit(X[prune_cols])
    survivors = list(pruner.keep_columns_)
    keep = survivors + [c for c in X.columns if c not in prune_cols]
    X_pruned = X[keep]
    cat_pruned = [c for c in categorical_features if c in keep]
    print(f"\n  prune @ threshold={threshold}: "
          f"{len(prune_cols)} -> {len(survivors)} numeric cols kept")
    t_pruned = time_one(X_pruned, cat_pruned, "pruned (prune before)")

    print()
    print(f"  speedup from pruning first: {t_full / t_pruned:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=14, choices=[3, 5, 7, 14, 21, 28])
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    main(args.window_size, args.threshold)
