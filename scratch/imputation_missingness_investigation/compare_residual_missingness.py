"""
Compare residual missingness after miceforest vs MissForest imputation.
Run from the scripts/ directory (or anywhere, paths are relative to scratch/):
    python ../scratch/imputation_missingness_investigation/compare_residual_missingness.py > ../scratch/imputation_missingness_investigation/residual_missingness_comparison.txt

Reads:
    ../scratch/imputation_missingness_investigation/miceforest_residual_missingness.txt
    ../scratch/imputation_missingness_investigation/missforest_residual_missingness.txt
"""
import ast
import re
from pathlib import Path

SCRATCH = Path(__file__).parent

FILES = {
    "miceforest": SCRATCH / "miceforest_residual_missingness.txt",
    "missforest":  SCRATCH / "missforest_residual_missingness.txt",
}

# Regex for per-fold lines:  outer1_inner2_train: {'col': 3, ...}
FOLD_RE  = re.compile(r"^\s+(outer\d+_inner\d+_\w+): (\{.+\})")
# Regex for the summary line
TOTAL_RE = re.compile(r"Residual NaN across all folds: (\{.+\})")
NONE_RE  = re.compile(r"No residual NaN")


def parse_file(path: Path):
    fold_records = {}   # label -> {col: count}
    total = {}
    fully_imputed = False

    with open(path) as fh:
        for line in fh:
            m = FOLD_RE.match(line)
            if m:
                fold_records[m.group(1)] = ast.literal_eval(m.group(2))
                continue
            m = TOTAL_RE.search(line)
            if m:
                total = ast.literal_eval(m.group(1))
                continue
            if NONE_RE.search(line):
                fully_imputed = True

    return fold_records, total, fully_imputed


def all_cols(*parsed):
    cols = set()
    for _, total, _ in parsed:
        cols.update(total.keys())
    return sorted(cols)


results = {}
for name, path in FILES.items():
    if not path.exists():
        print(f"[WARN] {path} not found — skipping {name}")
        continue
    results[name] = parse_file(path)

if len(results) < 2:
    print("Need both output files present to compare.")
    raise SystemExit(1)

mice_folds, mice_total, mice_ok = results["miceforest"]
miss_folds, miss_total, miss_ok = results["missforest"]

print("=" * 60)
print("RESIDUAL MISSINGNESS COMPARISON")
print("=" * 60)

# --- Per-fold summary ---
all_labels = sorted(set(mice_folds) | set(miss_folds))
if all_labels:
    print("\nFolds with residual NaN:")
    print(f"  {'Label':<35} {'miceforest':>15} {'missforest':>15}")
    print("  " + "-" * 65)
    for label in all_labels:
        mc = sum(mice_folds.get(label, {}).values())
        mf = sum(miss_folds.get(label, {}).values())
        print(f"  {label:<35} {mc:>15} {mf:>15}")
else:
    print("\nNo folds had residual NaN in either method.")

# --- Column-level totals ---
cols = all_cols(results["miceforest"], results["missforest"])
print("\nTotal residual NaN per column (across all folds):")
if cols:
    print(f"  {'Column':<35} {'miceforest':>12} {'missforest':>12}")
    print("  " + "-" * 59)
    for col in cols:
        mc = mice_total.get(col, 0)
        mf = miss_total.get(col, 0)
        print(f"  {col:<35} {mc:>12} {mf:>12}")
else:
    print("  (none)")

# --- Overall summary ---
print("\nOverall:")
print(f"  miceforest  — {'fully imputed' if mice_ok else f'total residual NaN: {sum(mice_total.values())}'}")
print(f"  missforest  — {'fully imputed' if miss_ok else f'total residual NaN: {sum(miss_total.values())}'}")
print("=" * 60)
