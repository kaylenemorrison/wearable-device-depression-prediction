"""Build a two-panel summary grid from modeling.py result files.

Scans the per-model text outputs under ../results/{model}/text/, parses the
held-out Summary metric from each run, and renders two heatmaps:

  * XGBoost
  * ElasticNet + MissForest

x-axis = window size (3, 7, 14, 28); y-axis = feature set. Each cell is the
metric of the most recent matching run (latest timestamp wins); combinations
with no run yet are left grey. Re-run this after new modeling runs to refresh.

    cd scripts
    python summary_grid.py                       # Test RMSE, most-populated fraction
    python summary_grid.py --metric r2
    python summary_grid.py --min-present-fraction 0.8
"""
import argparse
import glob
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "../results"

# x-axis
WINDOWS = [3, 7, 14, 28]

# y-axis: (feature_set_tag as written by modeling.py, display label). The tag is
# the fixed-order "+".join produced by parse_args (family_order = demographics,
# simple, catch22, adherence), so these must match that ordering exactly.
FEATURE_SETS = [
    ("demographics", "demographics"),
    ("demographics+simple", "demo+simple"),
    ("demographics+simple+adherence", "demo+simple+adherence"),
    ("demographics+simple+catch22+adherence", "demo+simple+catch22+adherence"),
    ("catch22", "catch22"),
]

# (panel title, text subdir, model token, required imputation token or None)
PANELS = [
    ("XGBoost", "xgboost", "xgboost", None),
    ("ElasticNet + MissForest", "elastic_net", "elastic_net", "missforest"),
]

# metric -> (label in the Summary block, lower_is_better)
METRICS = {
    "rmse": ("RMSE", True),
    "mae": ("MAE", True),
    "r2": ("Test R²", False),
}

# Filename: {model}_{imp}_{featureset}_w{N}_mp{frac}_{c22|noc22}_{YYYYmmdd_HHMMSS}.txt
# feature_set_tag has no underscores (families joined by '+'), so a non-greedy
# capture between the imputation token and the _w<window> anchor is unambiguous.
FILENAME_RE = re.compile(
    r"^(?P<model>elastic_net|xgboost)_"
    r"(?P<imp>simple_imputation|missforest|none)_"
    r"(?P<fs>.+?)_"
    r"w(?P<window>\d+)_"
    r"mp(?P<mp>[0-9.]+)_"
    r"(?P<c22>c22|noc22)_"
    r"(?P<ts>\d{8}_\d{6})\.txt$"
)


def parse_metric(path, summary_label):
    """Return the mean of `summary_label` from a run's Summary block, or None.

    The Summary lines look like '  RMSE:        4.8210 ± 0.3100'; we take the
    first number (the mean). '±' is unicode, so read the file as utf-8.
    """
    label = re.escape(summary_label)
    pat = re.compile(rf"^\s*{label}:\s*([0-9.]+)")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                return float(m.group(1))
    return None


def collect(panel_subdir, model_token, imp_token, summary_label, mp_filter):
    """Latest-wins metric per (feature_set_tag, window, mp) for one panel.

    Returns {(fs_tag, window, mp): value}. mp is part of the key, so runs at
    different present-fractions never overwrite each other -- latest-wins is
    scoped WITHIN a single mp. Scans only files whose model/imputation match the
    panel; when mp_filter is set, restricts to that present-fraction.
    """
    best = {}   # (fs, window, mp) -> (timestamp, value)
    text_dir = os.path.join(RESULTS_DIR, panel_subdir, "text")
    for path in glob.glob(os.path.join(text_dir, "*.txt")):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        if m["model"] != model_token:
            continue
        if imp_token is not None and m["imp"] != imp_token:
            continue
        mp = float(m["mp"])
        if mp_filter is not None and mp != mp_filter:
            continue

        fs, window = m["fs"], int(m["window"])
        if fs not in {f for f, _ in FEATURE_SETS} or window not in WINDOWS:
            continue

        key = (fs, window, mp)
        prev = best.get(key)
        if prev is not None and m["ts"] <= prev[0]:
            continue   # keep the newer run
        value = parse_metric(path, summary_label)
        if value is not None:
            best[key] = (m["ts"], value)

    return {k: v for k, (_, v) in best.items()}


def build_matrix(cell_values):
    """(len(FEATURE_SETS), len(WINDOWS)) masked array; missing cells masked."""
    data = np.full((len(FEATURE_SETS), len(WINDOWS)), np.nan)
    for i, (fs_tag, _) in enumerate(FEATURE_SETS):
        for j, w in enumerate(WINDOWS):
            v = cell_values.get((fs_tag, w))
            if v is not None:
                data[i, j] = v
    return np.ma.masked_invalid(data)


def draw_panel(ax, matrix, title, cmap, norm, lower_is_better):
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=12, pad=10)

    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([f"w{w}" for w in WINDOWS])
    ax.set_yticks(range(len(FEATURE_SETS)))
    ax.set_yticklabels([label for _, label in FEATURE_SETS])
    ax.set_xlabel("window size")

    # 2px surface gap between cells (dataviz mark-spacer convention).
    ax.set_xticks(np.arange(-0.5, len(WINDOWS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FEATURE_SETS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Annotate each populated cell; white text on dark fills, dark on light.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix.mask[i, j]:
                continue
            v = matrix[i, j]
            r, g, b, _ = cmap(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if lum < 0.5 else "#111111",
                    fontsize=10, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRICS), default="rmse",
                    help="Summary metric to grid. Default: rmse.")
    ap.add_argument("--min-present-fraction", type=float, default=None,
                    help="Restrict the grid to runs at this mp; "
                         "default: auto-pick the most-populated fraction.")
    ap.add_argument("--outfile", default=None,
                    help="Output PNG path. Default: ../results/summary_grid_<metric>.png")
    args = ap.parse_args()

    summary_label, lower_is_better = METRICS[args.metric]

    # Collect (fs, window, mp)-keyed cells per panel; mp is preserved so runs at
    # different fractions can't clobber each other.
    panel_cells = []
    for title, subdir, model_token, imp_token in PANELS:
        cells = collect(subdir, model_token, imp_token, summary_label,
                        args.min_present_fraction)
        panel_cells.append((title, cells))

    # The grid renders a single present-fraction so every cell shares the same
    # floor. If --min-present-fraction was given, collect already restricted to
    # it; otherwise auto-pick the fraction with the most populated cells across
    # both panels (tie -> higher fraction).
    if args.min_present_fraction is not None:
        target_mp = args.min_present_fraction
    else:
        counts = {}
        for _, cells in panel_cells:
            for (_, _, mp) in cells:
                counts[mp] = counts.get(mp, 0) + 1
        target_mp = max(counts, key=lambda mp: (counts[mp], mp)) if counts else None
        if len(counts) > 1:
            print(f"Multiple present-fractions found {sorted(counts)}; "
                  f"showing mp={target_mp:g} (most populated). "
                  f"Pass --min-present-fraction to pick another.")

    # Reduce each panel to (fs, window) at the chosen fraction for rendering.
    matrices = []
    for title, cells in panel_cells:
        grid = {(fs, w): v for (fs, w, mp), v in cells.items() if mp == target_mp}
        matrices.append((title, build_matrix(grid)))

    # Shared color scale across both panels so cells are comparable model-to-model.
    finite = np.concatenate([m.compressed() for _, m in matrices]) \
        if any(m.count() for _, m in matrices) else np.array([])
    if finite.size == 0:
        print("No matching result files found — nothing to plot yet.")
        print("Run modeling.py for these (model, feature-set, window) combos first.")
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:                      # single value -> pad so the ramp is visible
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Sequential single hue; orient so the *better* end is the darker (salient) one.
    cmap = plt.get_cmap("Blues_r" if lower_is_better else "Blues").copy()
    cmap.set_bad("#e0e0e0")               # missing cells -> grey

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for ax, (title, matrix) in zip(axes, matrices):
        draw_panel(ax, matrix, title, cmap, norm, lower_is_better)

    better = "lower is better" if lower_is_better else "higher is better"
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(f"{summary_label}  ({better})")

    mp_note = f"  |  mp={target_mp:g}" if target_mp is not None else ""
    fig.suptitle(f"Held-out {summary_label} by feature set × window{mp_note}",
                 fontsize=14)

    outfile = args.outfile or f"{RESULTS_DIR}/summary_grid_{args.metric}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Console recap: how many cells populated per panel.
    print(f"Metric: {summary_label} ({better})")
    for title, matrix in matrices:
        print(f"  {title:<28} {matrix.count():>2}/{matrix.size} cells populated")
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
