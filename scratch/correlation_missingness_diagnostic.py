"""Diagnose whether pre-imputation correlation pruning is safe on the merged
dataset. Prints ONLY aggregate statistics (no participant-level rows/values):
how much of the pairwise-correlation matrix is undefined due to missingness, and
how sparse the joint overlap between numeric feature pairs is.

Run from the scripts/ directory:
    python correlation_missingness_diagnostic.py --window-size 14
"""

import argparse

import numpy as np
import pandas as pd

# Demographic columns are never pruned (see modeling.py); excluded here too.
DEMOGRAPHIC_COLS = [
    "age",
    "gender",
    "mh_smoking",
    "audit_1",
    "depression_medication",
]

NON_FEATURE_COLS = ["phq8_score", "participant_name"]


def main(window_size, method):
    path = f"../data_merged/final_dataset_w{window_size}.parquet"
    df = pd.read_parquet(path)

    # Numeric, non-demographic, non-target columns = the prune candidates.
    drop = set(DEMOGRAPHIC_COLS) | set(NON_FEATURE_COLS)
    numeric = df.select_dtypes(include=[np.number])
    prune_cols = [c for c in numeric.columns if c not in drop]
    X = df[prune_cols]

    n_rows, n_cols = X.shape
    n_pairs = n_cols * (n_cols - 1) // 2

    # --- per-column missingness (aggregate only) ---------------------------
    miss_frac = X.isna().mean()

    # --- pairwise joint overlap (rows where BOTH columns are present) ------
    mask = X.notna().to_numpy().astype(np.int32)
    overlap = mask.T @ mask              # n_cols x n_cols matrix of joint counts
    iu = np.triu_indices(n_cols, k=1)    # off-diagonal upper triangle
    overlap_pairs = overlap[iu]

    # A Spearman/Pearson correlation is undefined when <2 rows jointly present.
    n_undefined_overlap = int((overlap_pairs < 2).sum())

    # --- fraction of NaN entries in the actual correlation matrix ----------
    corr = X.corr(method=method)
    corr_vals = corr.to_numpy()[iu]
    n_nan_corr = int(np.isnan(corr_vals).sum())

    print("=" * 60)
    print("CORRELATION-MISSINGNESS DIAGNOSTIC")
    print("=" * 60)
    print(f"  dataset:            {path}")
    print(f"  corr method:        {method}")
    print(f"  rows:               {n_rows}")
    print(f"  numeric prune cols: {n_cols}  (demographics excluded)")
    print(f"  unique pairs:       {n_pairs}")
    print()
    print("  Per-column missingness (fraction):")
    print(f"    min / median / max: "
          f"{miss_frac.min():.3f} / {miss_frac.median():.3f} / {miss_frac.max():.3f}")
    print()
    print("  Pairwise joint overlap (rows present in BOTH columns):")
    print(f"    min:    {int(overlap_pairs.min())}")
    print(f"    median: {int(np.median(overlap_pairs))}")
    print(f"    max:    {int(overlap_pairs.max())}")
    print(f"    pairs with <2 joint rows (corr undefined): "
          f"{n_undefined_overlap} ({n_undefined_overlap / n_pairs:.2%})")
    print()
    print("  Correlation matrix:")
    print(f"    NaN entries (off-diagonal): "
          f"{n_nan_corr} ({n_nan_corr / n_pairs:.2%})")
    print("=" * 60)
    print("\n  Interpretation:")
    print("  - High NaN% or low min-overlap => pairwise correlations are")
    print("    estimated on inconsistent/sparse subsets; pre-impute pruning")
    print("    may miss genuinely-redundant pairs (conservative: keeps too much).")
    print("  - Near-0 NaN% and large min-overlap => dense data; pre-impute")
    print("    pruning is safe and close to what post-impute pruning would give.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=14, choices=[3, 5, 7, 14, 21, 28])
    parser.add_argument("--method", default="spearman",
                        choices=["spearman", "pearson", "kendall"])
    args = parser.parse_args()
    main(args.window_size, args.method)
