"""Data-driven importance ranking of sleep features for predicting PHQ-8.

Two complementary, leakage-aware methods over the merged modeling dataset:

  1. Univariate Spearman |rho| between each sleep feature and phq8_score
     (monotonic, robust to outliers, pairwise-complete).
  2. Grouped-CV permutation importance: a LightGBM regressor is trained on the
     SLEEP features only inside a 5-fold GroupKFold (participants never split
     across folds). For each held-out fold, every feature is permuted and the
     increase in test MSE is recorded. Importance is the mean MSE increase
     across folds (held-out, so it reflects genuine predictive contribution).

Results are also rolled up to the base sleep construct (e.g. all
total_sleep_time_* aggregations -> "total_sleep_time").

Only aggregate rankings are printed; no participant-level rows are emitted.
"""
import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

SEED = 42

# Columns that are NOT sleep features (demographics, outcome, group, bookkeeping)
DEMOGRAPHIC_COLS = [
    "age", "gender", "marital_status", "children", "ethnicity_1",
    "drop_school_1", "accommodation", "mh_smoking", "audit_1",
    "depression_medication",
]
NON_FEATURE_COLS = ["phq8_score", "participant_name"]


def base_construct(col):
    """Map an aggregated sleep column back to its base sleep construct."""
    # order matters: check the more specific prefixes first
    checks = [
        ("total_sleep_time", "total_sleep_time"),
        ("time_in_bed", "time_in_bed"),
        ("light_restorative_ratio", "stage_ratio"),
        ("rem_deep_ratio", "stage_ratio"),
        ("light_pct", "light_pct"),
        ("deep_pct", "deep_pct"),
        ("REM_pct", "REM_pct"),
        ("awake_pct", "awake_pct"),
        ("NREM_pct", "NREM_pct"),
        ("rem_latency", "rem_latency"),
        ("awakenings", "awakenings"),
        ("insomnia", "insomnia/hypersomnia"),
        ("hypersomnia", "insomnia/hypersomnia"),
        ("any_disorder", "insomnia/hypersomnia"),
        ("sleep_onset", "sleep_timing"),
        ("sleep_offset", "sleep_timing"),
        ("sleep_midpoint", "sleep_timing"),
        ("sleep_phase_instability", "sleep_timing"),
        ("avg_time_in_bed_window", "sleep_timing"),
    ]
    for prefix, construct in checks:
        if col.startswith(prefix):
            return construct
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-size", type=int, default=14, choices=[3, 5, 7, 14])
    args = ap.parse_args()

    df = pd.read_parquet(f"../data_merged/final_dataset_w{args.window_size}.parquet")

    # any_disorder may be carried as an object/category flag; coerce to numeric
    if "any_disorder" in df.columns:
        df["any_disorder"] = pd.to_numeric(df["any_disorder"], errors="coerce")

    exclude = set(DEMOGRAPHIC_COLS + NON_FEATURE_COLS)
    exclude |= {c for c in df.columns if c.startswith("sleep_missing_last")}

    sleep_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    y = df["phq8_score"]
    groups = df["participant_name"]
    keep = y.notna() & groups.notna()
    df, y, groups = df.loc[keep], y.loc[keep], groups.loc[keep]
    X = df[sleep_cols]

    print("=" * 70)
    print(f"DATA-DRIVEN SLEEP-FEATURE IMPORTANCE (window={args.window_size})")
    print("=" * 70)
    print(f"  Rows (PHQ-8 assessments): {len(df)}")
    print(f"  Participants:             {groups.nunique()}")
    print(f"  Sleep features:           {len(sleep_cols)}")

    # ---- Method 1: univariate Spearman -------------------------------------
    spear = {}
    for c in sleep_cols:
        m = X[c].notna()
        if m.sum() < 30 or X.loc[m, c].nunique() < 2:
            spear[c] = np.nan
            continue
        rho, _ = spearmanr(X.loc[m, c], y.loc[m])
        spear[c] = rho
    spear = pd.Series(spear)

    # ---- Method 2: grouped-CV permutation importance -----------------------
    gkf = GroupKFold(n_splits=5)
    perm = np.zeros(len(sleep_cols))
    fold_r2 = []
    for tr, te in gkf.split(X, y, groups):
        model = LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        fold_r2.append(model.score(X.iloc[te], y.iloc[te]))
        r = permutation_importance(
            model, X.iloc[te], y.iloc[te],
            scoring="neg_mean_squared_error",
            n_repeats=10, random_state=SEED, n_jobs=-1,
        )
        perm += r.importances_mean
    perm = pd.Series(perm / gkf.get_n_splits(), index=sleep_cols)

    print(f"  LightGBM held-out R² per fold: "
          f"{', '.join(f'{v:.3f}' for v in fold_r2)} "
          f"(mean {np.mean(fold_r2):.3f})")

    # ---- combine -----------------------------------------------------------
    res = pd.DataFrame({
        "feature": sleep_cols,
        "construct": [base_construct(c) for c in sleep_cols],
        "spearman_rho": spear.values,
        "abs_rho": spear.abs().values,
        "perm_importance": perm.values,
    })

    print("\n" + "-" * 70)
    print("TOP 20 INDIVIDUAL SLEEP FEATURES (by grouped-CV permutation importance)")
    print("-" * 70)
    print(f"  {'feature':<40}{'perm_imp':>12}{'rho':>10}")
    top = res.sort_values("perm_importance", ascending=False).head(20)
    for _, r in top.iterrows():
        rho = f"{r['spearman_rho']:+.3f}" if pd.notna(r["spearman_rho"]) else "   nan"
        print(f"  {r['feature']:<40}{r['perm_importance']:>12.5f}{rho:>10}")

    print("\n" + "-" * 70)
    print("TOP 15 INDIVIDUAL SLEEP FEATURES (by univariate |Spearman rho|)")
    print("-" * 70)
    print(f"  {'feature':<40}{'rho':>10}")
    for _, r in res.sort_values("abs_rho", ascending=False).head(15).iterrows():
        print(f"  {r['feature']:<40}{r['spearman_rho']:>+10.3f}")

    print("\n" + "-" * 70)
    print("RANKING BY BASE SLEEP CONSTRUCT")
    print("-" * 70)
    grp = res.groupby("construct").agg(
        n_features=("feature", "size"),
        perm_total=("perm_importance", "sum"),
        perm_max=("perm_importance", "max"),
        best_abs_rho=("abs_rho", "max"),
    ).sort_values("perm_total", ascending=False)
    print(f"  {'construct':<22}{'n':>4}{'perm_total':>12}{'perm_max':>11}{'best|rho|':>11}")
    for name, r in grp.iterrows():
        print(f"  {name:<22}{int(r['n_features']):>4}{r['perm_total']:>12.5f}"
              f"{r['perm_max']:>11.5f}{r['best_abs_rho']:>11.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
