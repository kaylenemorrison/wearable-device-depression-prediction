"""Quick estimate of how much CorrelationPruner shrinks the numeric feature
block, and the implied MissForest imputation speedup. MissForest cost scales
with (#columns that have missingness) x (column width), so we report both.
Prints only counts/timings -- no participant rows. Flushed so progress shows.

Run from scripts/ (thesis env):
    PYTHONPATH=. python ../scratch/prune_reduction.py
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "classes")
from CorrelationPruner import CorrelationPruner  # noqa: E402

DEMO = ["age", "gender", "mh_smoking", "audit_1", "depression_medication"]
NONFEAT = ["phq8_score", "participant_name"]

df = pd.read_parquet("../data_merged/final_dataset_w14.parquet")
numeric = df.select_dtypes(include=[np.number])
drop = set(DEMO) | set(NONFEAT)
prune_cols = [c for c in numeric.columns if c not in drop]
X = df[prune_cols]

n_full = len(prune_cols)
miss_full = [c for c in prune_cols if X[c].isna().any()]
print(f"numeric prune candidates: {n_full}", flush=True)
print(f"  with missingness (one MissForest model each): {len(miss_full)}", flush=True)
print("", flush=True)

for thr in [0.95, 0.9, 0.8]:
    t0 = time.time()
    p = CorrelationPruner(threshold=thr)
    p.fit(X)
    surv = list(p.keep_columns_)
    miss_surv = [c for c in surv if X[c].isna().any()]
    dt = time.time() - t0
    model_ratio = len(miss_full) / max(len(miss_surv), 1)
    width_ratio = n_full / max(len(surv), 1)
    print(f"threshold {thr}: {n_full} -> {len(surv)} survivors "
          f"({len(miss_surv)} with missingness)", flush=True)
    print(f"    est. imputation speedup ~{model_ratio * width_ratio:.1f}x "
          f"(model-count {model_ratio:.2f}x * width {width_ratio:.2f}x)  "
          f"[prune fit {dt:.0f}s]", flush=True)
    print("", flush=True)
